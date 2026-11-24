"""FLAMINGO simulations (Schaye et al 2023, Kugel et al 2023).

Data is fetched from the FLAMINGO public data release (Helly et al 2026,
https://dataweb.cosma.dur.ac.uk:8443/flamingo/), using the ``hdfstream``
module (``pip install hdfstream``; not on conda-forge). The data release
allows individual HDF5 datasets ("columns") to be downloaded, without
downloading entire files. This is essential, since the full snapshots are
huge (e.g. a single L1_m8 snapshot is ~3 TB).

Local cache layout: every file ends up under $KSZX_DATA/flamingo/<relpath>,
where <relpath> is the path below /FLAMINGO on the server, e.g.::

   flamingo/L1_m8/L1_m8/snapshots/flamingo_0014/swift_snapshot_0014/flamingo_0014.{0..479}.hdf5
   flamingo/L1_m8/L1_m8/SOAP-HBT/halo_properties_0014.hdf5

Each local file is a column-subset of the corresponding remote file: the
requested datasets are copied "as is" (same HDF5 paths, dtypes, attributes,
and units), along with small metadata groups (Header, Cosmology, Units, etc.)
Additional columns can be added later, by calling download() again (the new datasets are
appended to the existing local files). Use show() to see which columns have been downloaded
(and their disk usage), and delete() to delete columns (repacking files to reclaim space).

Notes:

  - FLAMINGO units are (Mpc, 1e10 Msol, km/s), with no h-factors. Positions
    are comoving. Since files are downloaded "as is", conversion to kszx
    units (e.g. Msol, c=1) is the responsibility of the reader. Each dataset
    has attributes describing its units (e.g. 'Conversion factor to CGS
    (not including cosmological corrections)', 'a-scale exponent').

  - Each snapshot is split into many chunk files (480 for L1_m8). A single
    chunk file does not correspond to a simple spatial region -- it is a union
    of cells (see the 'Cells' group), which can be scattered through the box.

  - We don't download the "virtual" snapshot file flamingo_XXXX.hdf5. (It
    contains HDF5 virtual datasets pointing to the chunk files, including
    chunk files that we don't download, which HDF5 silently replaces by fill
    values.)

  - According to the data release documentation, the FOFGroupIDs and X-ray
    columns in the swift_snapshot_XXXX chunk files are outdated (corrected
    values are in the membership_XXXX and xray_XXXX chunk files, which
    download() doesn't currently support).

  - Dark matter particles do NOT all have the same mass, so PartType1/Masses
    must be downloaded along with Coordinates. In L1_m8, DM masses range
    from 6.886e8 to 7.235e8 Msol (vs Header/InitialMassTable[1] = 7.063e8 Msol),
    with identical min/max at z=3 and z=4 (masses are fixed at the initial
    conditions).

  - Download speed is limited to ~13 MB/s per connection (server-side), so
    download() uses multiple processes, one remote file per process. The
    data release documentation asks users to limit themselves to 2-4
    simultaneous requests.
"""

import os
import sys
import glob
import time
import multiprocessing
import numpy as np
import h5py

from . import io_utils


_server = 'cosma'      # hdfstream alias for https://dataweb.cosma.dur.ac.uk:8443/hdfstream
_remote_root = '/FLAMINGO'

# Keyword argument name (in download()) -> HDF5 group name in snapshot files.
_part_types = {
    'gas': 'PartType0',
    'dm': 'PartType1',
    'stars': 'PartType4',
    'bh': 'PartType5',
    'neutrinos': 'PartType6',
}

# Small groups copied to each local file (if present in the remote file).
_metadata_groups = [ 'Header', 'Cosmology', 'Units', 'Parameters', 'Code', 'Cells' ]

# Large datasets are downloaded in requests of (approximately) this size.
_request_nbytes = 2**28

_nretries = 5
_retry_delay = 30   # seconds


def download(sim, z, *, gas=(), dm=(), stars=(), bh=(), neutrinos=(), halos=(), files=None, nproc=4):
    r"""Downloads selected columns from a FLAMINGO snapshot and/or SOAP halo catalog.

    Function args:

      - ``sim`` (str): simulation name, e.g. 'L1_m8', 'L1_m8_DMO', or 'L1_m9/fgas+2sigma'.
        This is a path below /FLAMINGO on the server. If the name is unambiguous,
        then the first path component (e.g. 'L1_m9/') may be omitted.

      - ``z`` (float): redshift. Must be one of the snapshot redshifts. (If particle
        columns are requested, then the snapshot must be a full snapshot -- the error
        message will list the full snapshots if not.)

      - ``gas``, ``dm``, ``stars``, ``bh``, ``neutrinos`` (lists of str): dataset names
        to download from the snapshot chunk files, for each particle type
        (PartType0, PartType1, PartType4, PartType5, PartType6 respectively).
        For example, ``gas=['Coordinates','Masses']``.

      - ``halos`` (list of str): dataset names to download from the SOAP halo catalog
        (SOAP-HBT/halo_properties_XXXX.hdf5), for example
        ``halos=['InputHalos/HaloCentre', 'SO/200_crit/TotalMass', 'InputHalos/IsCentral']``.

      - ``files`` (list of int, optional): restrict snapshot download to a subset of
        chunk files (default is all chunk files). Intended for testing.

      - ``nproc`` (int): number of processes (= simultaneous requests to the server).
        Each process downloads one remote file at a time.

    Datasets which are already present locally are skipped (re-running is safe, and
    an interrupted download can be resumed). A dataset is written under a temporary
    name ('<name>.partial'), and renamed when complete. (After an interrupted download,
    the deleted partial dataset leaves unused space in the HDF5 file. See :func:`show()`
    to view disk usage, and :func:`delete()` to delete columns and reclaim space.)

    Can be called from command line: ``python -m kszx download_flamingo ...``.
    """

    hdfstream = _import_hdfstream()

    halos = list(halos)
    pcols = [ ]
    for k, cols in [ ('gas',gas), ('dm',dm), ('stars',stars), ('bh',bh), ('neutrinos',neutrinos) ]:
        pcols += [ f'{_part_types[k]}/{c}' for c in cols ]

    if (len(pcols) == 0) and (len(halos) == 0):
        print('kszx.flamingo.download(): no columns were specified, nothing to do')
        return

    root = hdfstream.open(_server, _remote_root)
    simpath = _resolve_sim(root, sim)
    snap = _resolve_snap(root, simpath, z)

    # List of (relpath, dataset_names) pairs, one per remote file.
    # The SOAP file goes first, since it's a single large file (better load-balancing).
    tasks = [ ]

    if len(halos) > 0:
        tasks.append((f'{simpath}/SOAP-HBT/halo_properties_{snap:04d}.hdf5', halos))

    if len(pcols) > 0:
        snapdir = f'{simpath}/snapshots/flamingo_{snap:04d}'
        if f'flamingo_{snap:04d}' not in root[f'{simpath}/snapshots']:
            raise RuntimeError(f'kszx.flamingo.download(): {sim=} has no full snapshot at {z=}.'
                               + f' Available redshifts: {_full_snapshot_redshifts(root, simpath)}')

        swdir = f'{snapdir}/swift_snapshot_{snap:04d}'
        nfiles = len([ f for f in root[swdir] if f.endswith('.hdf5') ])
        files = range(nfiles) if (files is None) else files

        for i in files:
            if not (0 <= i < nfiles):
                raise RuntimeError(f'kszx.flamingo.download(): file index {i} out of range (expected 0 <= i < {nfiles})')
            tasks.append((f'{swdir}/flamingo_{snap:04d}.{i}.hdf5', pcols))

    local_root = os.path.join(io_utils.get_data_dir(), 'flamingo')
    nproc = min(nproc, len(tasks))
    args = [ (local_root, relpath, dsets) for (relpath, dsets) in tasks ]

    print(f'kszx.flamingo.download(): {len(tasks)} remote file(s), {nproc} process(es)')
    t0 = time.time()
    nbytes_tot = 0

    if nproc <= 1:
        results = map(_download_file, args)
        pool = None
    else:
        pool = multiprocessing.Pool(nproc)
        results = pool.imap_unordered(_download_file, args)

    try:
        for i, (relpath, nbytes) in enumerate(results):
            nbytes_tot += nbytes
            dt = max(time.time() - t0, 1.0e-3)
            status = 'done' if (nbytes > 0) else 'already present'
            print(f'[{i+1}/{len(tasks)}] {relpath} {status}. Total {nbytes_tot/1e9:.2f} GB downloaded'
                  + f' in {dt:.0f} s ({nbytes_tot/dt/1e6:.1f} MB/s)', flush=True)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


def show(sim=None, z=None):
    r"""Prints a summary of the locally downloaded FLAMINGO data: which columns are present, in how many
    files, and how much disk space they use.

    Function args:

      - ``sim`` (str, optional): restrict to one simulation (same naming as :func:`download()`).
      - ``z`` (float, optional): restrict to one redshift.

    Only reads local files (no server access). Can be called from command line:
    ``python -m kszx download_flamingo --show [sim [z]]``.
    """

    local_root = os.path.join(io_utils.get_data_dir(), 'flamingo')
    simpaths = _local_sims(local_root, sim)
    if len(simpaths) == 0:
        print(f'No FLAMINGO data found in {local_root}')
        return

    grand_total = 0
    for simpath in simpaths:
        print(f'{simpath}  ({os.path.join(local_root, simpath)})')
        for p in _local_products(local_root, simpath, z):
            sizes, ncols, nbytes_cols, npartial = 0, { }, { }, 0
            for fn in p['files']:
                fsize, dsets, partial = _scan_file(fn)
                sizes += fsize
                npartial += (len(partial) > 0)
                for d, nb in dsets.items():
                    ncols[d] = ncols.get(d, 0) + 1
                    nbytes_cols[d] = nbytes_cols.get(d, 0) + nb
            grand_total += sizes

            if p['kind'] == 'snapshot':
                nexp = f' (of {p["nfiles_expected"]})' if (p['nfiles_expected'] != len(p['files'])) else ''
                print(f'  Snapshot {p["snap"]} (z={p["z"]:.2f}): {len(p["files"])} chunk files{nexp}, {_fmt_bytes(sizes)} on disk')
            else:
                print(f'  SOAP halo catalog {p["snap"]} (z={p["z"]:.2f}): {_fmt_bytes(sizes)} on disk')

            ptype_names = { v: k for k, v in _part_types.items() }
            for d in sorted(ncols):
                top = d.split('/')[0]
                label = f'{ptype_names[top]:>9s}: {d.split("/",1)[1]}' if (top in ptype_names) else f'  {d}'
                nfiles = f'{ncols[d]}/{len(p["files"])} files, ' if (p['kind'] == 'snapshot') else ''
                print(f'      {label:<45s} {nfiles}{_fmt_bytes(nbytes_cols[d])}')

            other = sizes - sum(nbytes_cols.values())
            print(f'      {"(metadata + unused space)":<45s} {_fmt_bytes(other)}')
            if npartial > 0:
                print(f'      WARNING: {npartial} file(s) contain incomplete (.partial) datasets from an interrupted download')

    print(f'Total: {_fmt_bytes(grand_total)}')


def delete(sim, z, *, gas=(), dm=(), stars=(), bh=(), neutrinos=(), halos=(), files=None, nproc=4, confirm=True):
    r"""Deletes selected columns from locally downloaded FLAMINGO files, and repacks the files to reclaim disk space.

    Function args are the same as :func:`download()`: ``sim``, ``z``, column lists ``gas``, ``dm``, ``stars``,
    ``bh``, ``neutrinos``, ``halos``, optional chunk-file subset ``files``, and ``nproc`` (number of files
    repacked in parallel). If ``confirm`` is True (the default), prints a summary of what will be deleted
    and asks for confirmation.

    Repacking rewrites each modified file (HDF5 does not return space to the filesystem when a dataset is
    deleted), so temporarily needs extra disk space equal to the size of the largest file being repacked,
    times ``nproc``. Files with no remaining columns are removed entirely. Don't run this while a download
    of the same files is in progress.

    Only reads/writes local files (no server access). Can be called from command line:
    ``python -m kszx download_flamingo <sim> <z> --delete [column flags]``.
    """

    pcols = [ ]
    for k, cols in [ ('gas',gas), ('dm',dm), ('stars',stars), ('bh',bh), ('neutrinos',neutrinos) ]:
        pcols += [ f'{_part_types[k]}/{c}' for c in cols ]
    halos = list(halos)

    if (len(pcols) == 0) and (len(halos) == 0):
        raise RuntimeError('kszx.flamingo.delete(): no columns were specified')
    for d in pcols + halos:
        if d.split('/')[0] in _metadata_groups:
            raise RuntimeError(f'kszx.flamingo.delete(): refusing to delete metadata group {d}')

    local_root = os.path.join(io_utils.get_data_dir(), 'flamingo')
    simpaths = _local_sims(local_root, sim)
    if len(simpaths) != 1:
        raise RuntimeError(f'kszx.flamingo.delete(): {sim=} not found (or ambiguous) in {local_root}')
    products = { p['kind']: p for p in _local_products(local_root, simpaths[0], z) }

    # List of (filename, [datasets to delete]) pairs.
    tasks = [ ]
    for kind, dsets in [ ('halos', halos), ('snapshot', pcols) ]:
        if len(dsets) == 0:
            continue
        if kind not in products:
            print(f'kszx.flamingo.delete(): no local {kind} files for {simpaths[0]} at z={z}, skipping {dsets}')
            continue
        fnames = products[kind]['files']
        if (kind == 'snapshot') and (files is not None):
            fnames = [ fn for fn in fnames if _chunk_index(fn) in set(files) ]
        tasks += [ (fn, dsets) for fn in fnames ]

    # Summary of what will be deleted.
    nfiles, nbytes = { }, { }
    todo = [ ]
    for fn, dsets in tasks:
        _, present, partial = _scan_file(fn)
        dd = [ d for d in dsets if (d in present) or (d + '.partial' in partial) ]
        for d in dd:
            nfiles[d] = nfiles.get(d, 0) + 1
            nbytes[d] = nbytes.get(d, 0) + present.get(d, 0) + partial.get(d + '.partial', 0)
        if len(dd) > 0:
            todo.append((fn, dd))

    for d in pcols + halos:
        if d not in nfiles:
            print(f'  {d}: not present locally, nothing to delete')
    if len(todo) == 0:
        return

    print(f'kszx.flamingo.delete(): will delete from {simpaths[0]} at z={z}:')
    for d in nfiles:
        print(f'  {d}: {nfiles[d]} file(s), {_fmt_bytes(nbytes[d])}')
    print(f'  Total: {_fmt_bytes(sum(nbytes.values()))} in {len(todo)} file(s), which will then be repacked')

    if confirm:
        if not sys.stdin.isatty():
            raise RuntimeError('kszx.flamingo.delete(): stdin is not a terminal, so cannot ask for confirmation'
                               + ' (use confirm=False, or -y on the command line)')
        if input('Proceed? [y/N] ').strip().lower() not in ('y', 'yes'):
            print('Aborted')
            return

    nproc = max(1, min(nproc, len(todo)))
    freed = 0
    t0 = time.time()

    if nproc <= 1:
        results = map(_delete_worker, todo)
        pool = None
    else:
        pool = multiprocessing.Pool(nproc)
        results = pool.imap_unordered(_delete_worker, todo)

    try:
        for i, (fn, size0, size1, removed) in enumerate(results):
            freed += size0 - size1
            status = 'removed (no columns left)' if removed else f'repacked, {_fmt_bytes(size0)} -> {_fmt_bytes(size1)}'
            print(f'[{i+1}/{len(todo)}] {os.path.relpath(fn, local_root)}: {status}. Total freed {_fmt_bytes(freed)}'
                  + f' ({time.time()-t0:.0f} s)', flush=True)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


####################################################################################################


def _fmt_bytes(n):
    for unit, scale in [ ('TB', 1e12), ('GB', 1e9), ('MB', 1e6), ('KB', 1e3) ]:
        if abs(n) >= scale:
            return f'{n/scale:.2f} {unit}'
    return f'{n} B'


def _chunk_index(filename):
    """E.g. '.../flamingo_0014.123.hdf5' -> 123."""
    return int(os.path.basename(filename).split('.')[-2])


def _local_sims(local_root, sim=None):
    """Returns list of locally present simulation paths (e.g. 'L1_m8/L1_m8'), optionally restricted to 'sim'."""

    ret = [ ]
    for d in sorted(glob.glob(f'{local_root}/*/*')):
        if os.path.isdir(f'{d}/snapshots') or os.path.isdir(f'{d}/SOAP-HBT'):
            ret.append(os.path.relpath(d, local_root))

    if sim is not None:
        sim = sim.strip('/')
        ret = [ s for s in ret if (s == sim) or (s.split('/')[-1] == sim) ]
    return ret


def _header_redshift(filename):
    with h5py.File(filename, 'r') as f:
        return float(f['Header'].attrs['Redshift'][0])


def _local_products(local_root, simpath, z=None):
    """Returns list of dicts describing local snapshots and SOAP catalogs for one simulation.

    Each dict has keys 'kind' ('snapshot' or 'halos'), 'snap', 'z', 'files', and (for snapshots) 'nfiles_expected'.
    If z is specified, only products at that redshift are returned.
    """

    ret = [ ]
    for d in sorted(glob.glob(f'{local_root}/{simpath}/snapshots/flamingo_*')):
        snap = int(os.path.basename(d).split('_')[1])
        fnames = sorted(glob.glob(f'{d}/swift_snapshot_{snap:04d}/flamingo_{snap:04d}.*.hdf5'), key=_chunk_index)
        if len(fnames) == 0:
            continue
        with h5py.File(fnames[0], 'r') as f:
            zs = float(f['Header'].attrs['Redshift'][0])
            nexp = int(f['Header'].attrs['NumFilesPerSnapshot'][0])
        ret.append({ 'kind': 'snapshot', 'snap': snap, 'z': zs, 'files': fnames, 'nfiles_expected': nexp })

    for fn in sorted(glob.glob(f'{local_root}/{simpath}/SOAP-HBT/halo_properties_*.hdf5')):
        snap = int(os.path.basename(fn).split('_')[-1].split('.')[0])
        ret.append({ 'kind': 'halos', 'snap': snap, 'z': _header_redshift(fn), 'files': [fn] })

    if z is not None:
        ret = [ p for p in ret if abs(p['z'] - z) < 0.005 ]
    return sorted(ret, key=lambda p: (p['snap'], p['kind'] != 'snapshot'))


def _scan_file(filename):
    """Returns (file size, {dataset: storage bytes}, {partial dataset: storage bytes}) for non-metadata datasets."""

    dsets, partial = { }, { }
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and (name.split('/')[0] not in _metadata_groups):
            (partial if name.endswith('.partial') else dsets)[name] = obj.id.get_storage_size()

    with h5py.File(filename, 'r') as f:
        f.visititems(visit)
    return os.path.getsize(filename), dsets, partial


def _delete_worker(args):
    """Worker function for delete(): deletes datasets from one file, then repacks (or removes) it.
    Returns (filename, size before, size after, removed)."""

    filename, dsets = args
    size0 = os.path.getsize(filename)

    with h5py.File(filename, 'a') as f:
        for d in dsets:
            for name in (d, d + '.partial'):
                if name in f:
                    del f[name]
            # Remove parent groups which are now empty (e.g. 'PartType1' after deleting all DM columns).
            parts = d.split('/')
            for i in range(len(parts)-1, 0, -1):
                g = '/'.join(parts[:i])
                if (g in f) and (len(f[g]) == 0):
                    del f[g]

    _, remaining, remaining_partial = _scan_file(filename)
    if len(remaining) + len(remaining_partial) == 0:
        os.remove(filename)
        return filename, size0, 0, True

    _repack(filename)
    return filename, size0, os.path.getsize(filename), False


def _repack(filename):
    """Rewrites an HDF5 file (copying all objects and attributes), to reclaim space from deleted datasets."""

    tmp = filename + '.repack.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)   # leftover from an interrupted repack
    with h5py.File(filename, 'r') as src, h5py.File(tmp, 'w') as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        for name in src:
            src.copy(src[name], dst, name=name)
    os.replace(tmp, filename)


def _import_hdfstream():
    try:
        import hdfstream
    except ImportError as e:
        raise ImportError("kszx.flamingo: the 'hdfstream' module is needed to download FLAMINGO data"
                          + " (pip install hdfstream)") from e
    return hdfstream


def _resolve_sim(root, sim):
    """Returns path below /FLAMINGO, e.g. 'L1_m8' -> 'L1_m8/L1_m8'."""

    sim = sim.strip('/')

    if '/' in sim:
        box, name = sim.split('/', 1)
        if (box in root) and (name in root[box]):
            return sim
        candidates = [ ]
    else:
        candidates = [ f'{box}/{sim}' for box in root if sim in root[box] ]
        if len(candidates) == 1:
            return candidates[0]

    all_sims = [ f'{box}/{name}' for box in root for name in root[box] ]
    if len(candidates) > 1:
        raise RuntimeError(f'kszx.flamingo: {sim=} is ambiguous (candidates: {candidates})')
    raise RuntimeError(f'kszx.flamingo: {sim=} not found (available: {all_sims})')


def _snapshot_redshifts(root, simpath):
    """Returns list of redshifts, indexed by snapshot number (parsed from output_list.txt)."""

    with root[f'{simpath}/output_list.txt'].open('r') as f:
        lines = [ line.strip() for line in f.read().splitlines() ]
    return [ float(line) for line in lines if (len(line) > 0) and not line.startswith('#') ]


def _full_snapshot_redshifts(root, simpath):
    zlist = _snapshot_redshifts(root, simpath)
    snaps = sorted(int(d.split('_')[1]) for d in root[f'{simpath}/snapshots'] if d.startswith('flamingo_'))
    return [ zlist[s] for s in snaps ]


def _resolve_snap(root, simpath, z):
    zlist = _snapshot_redshifts(root, simpath)
    snaps = [ i for i, zi in enumerate(zlist) if abs(zi - z) < 0.005 ]
    if len(snaps) != 1:
        raise RuntimeError(f'kszx.flamingo: {z=} is not a snapshot redshift for {simpath} (snapshot redshifts: {zlist})')
    return snaps[0]


def _retry(f, what):
    for attempt in range(_nretries):
        try:
            return f()
        except Exception as e:
            if attempt == _nretries-1:
                raise
            print(f'kszx.flamingo: {what} failed ({e!r}), retrying in {_retry_delay} s', flush=True)
            time.sleep(_retry_delay)


def _copy_attrs(src, dst):
    for k, v in src.attrs.items():
        if isinstance(v, np.ndarray) and (v.dtype == object):
            v = np.array([ str(x) for x in v.ravel() ], dtype=h5py.string_dtype()).reshape(v.shape)
        dst.attrs[k] = v


def _download_file(args):
    """Worker function: downloads all missing datasets from one remote file. Returns (relpath, nbytes)."""

    local_root, relpath, dsets = args
    local_filename = os.path.join(local_root, relpath)
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)

    hdfstream = _import_hdfstream()
    hdfstream.disable_progress(True)   # tqdm progress bars from multiple processes would be unreadable

    remote = None
    nbytes = 0

    with h5py.File(local_filename, 'a') as out:
        # Remove leftovers from an interrupted download.
        partial = [ ]
        out.visit(lambda name: partial.append(name) if name.endswith('.partial') else None)
        for name in partial:
            if name in out:   # could have been deleted already, as a member of a partial group
                del out[name]

        todo_groups = [ g for g in _metadata_groups if g not in out ]
        todo_dsets = [ d for d in dsets if d not in out ]

        if (len(todo_dsets) == 0) and ('Header' in out):
            return relpath, 0

        remote = _retry(lambda: hdfstream.open(_server, f'{_remote_root}/{relpath}'), f'opening {relpath}')

        for g in todo_groups:
            if g in remote:
                remote.copy(g, out, name=f'{g}.partial')
                out.move(f'{g}.partial', g)

        for d in todo_dsets:
            if d not in remote:
                raise RuntimeError(f'kszx.flamingo: dataset {d} not found in {relpath}')
            nbytes += _download_dataset(remote, out, d, relpath)

    return relpath, nbytes


def _download_dataset(remote, out, name, relpath):
    """Copies remote dataset 'name' to local h5py.File 'out'. Returns number of bytes downloaded."""

    t0 = time.time()
    rds = remote[name]

    # Create parent groups (copying attributes from remote groups).
    parts = name.split('/')
    for i in range(1, len(parts)):
        gname = '/'.join(parts[:i])
        if gname not in out:
            _copy_attrs(remote[gname], out.create_group(gname))

    tmpname = name + '.partial'
    shape, dtype = tuple(rds.shape), rds.dtype
    nbytes = int(np.prod(shape)) * dtype.itemsize

    if (len(shape) == 0) or (nbytes <= _request_nbytes):
        data = _retry(lambda: rds[...], f'reading {relpath}:{name}')
        dst = out.create_dataset(tmpname, data=data)
    else:
        dst = out.create_dataset(tmpname, shape=shape, dtype=dtype)
        row_nbytes = int(np.prod(shape[1:])) * dtype.itemsize
        nrows = max(1, _request_nbytes // row_nbytes)
        buf = np.empty((nrows,) + shape[1:], dtype=dtype)

        for i in range(0, shape[0], nrows):
            j = min(i + nrows, shape[0])
            b = buf[:(j-i)]
            _retry(lambda: rds.read_direct(b, source_sel=np.s_[i:j]), f'reading {relpath}:{name}[{i}:{j}]')
            dst[i:j] = b

    _copy_attrs(rds, dst)
    out.move(tmpname, name)

    dt = time.time() - t0
    print(f'    {relpath}:{name}: {nbytes/1e9:.2f} GB in {dt:.0f} s ({nbytes/dt/1e6:.1f} MB/s)', flush=True)
    return nbytes
