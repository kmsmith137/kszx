"""kszx/gadget_utils.py — Reader for Gadget-II/III binary FoF halo catalogs.

This is a general-purpose reader for FoF group catalogs written by Gadget-II/III.
It handles single-file and multi-file catalogs (group_tab_NNN.0, group_tab_NNN.1, ...).

Supports both plain binary format (no record markers) and Fortran-record-wrapped format.
The format is auto-detected from the file contents.

All arrays are returned in raw Gadget units — unit conversions are handled by the
calling module (e.g. quijote.py).
"""

import os
import struct
import numpy as np


####################################################################################################
# Low-level helpers


def read_fortran_record(f, dtype, count):
    """Read a Fortran-style record: [4-byte len] data [4-byte len]."""
    reclen = struct.unpack('I', f.read(4))[0]
    expected = np.dtype(dtype).itemsize * count
    assert reclen == expected, f'Record length mismatch: {reclen} != {expected}'
    arr = np.fromfile(f, dtype=dtype, count=count)
    reclen2 = struct.unpack('I', f.read(4))[0]
    assert reclen == reclen2
    return arr


def skip_fortran_record(f):
    """Skip a Fortran-style record without reading its contents."""
    reclen = struct.unpack('I', f.read(4))[0]
    f.seek(reclen, 1)  # seek relative to current position
    reclen2 = struct.unpack('I', f.read(4))[0]
    assert reclen == reclen2


####################################################################################################
# FoF halo catalogs


def read_fof(catalog_dir):
    """Read a Gadget FoF group_tab catalog (possibly multi-file).

    Args:
        catalog_dir (str): Directory containing group_tab_NNN.X files.

    Returns:
        dict with keys (all in raw Gadget units):
            'npart': int32 array — particle count per halo
            'mass': float32 array — mass in 10^10 M_sun/h
            'pos': float32 array, shape (N,3) — positions in kpc/h
            'vel': float32 array, shape (N,3) — velocities in km/s * a
    """
    # Discover the group_tab file prefix.
    # Files are named group_tab_NNN.X where NNN is the snapshot number.
    files = sorted([f for f in os.listdir(catalog_dir) if f.startswith('group_tab_')])
    if len(files) == 0:
        raise RuntimeError(f'No group_tab files found in {catalog_dir}')

    # Get the snapshot number from the first filename
    # e.g. 'group_tab_004.0' -> prefix = 'group_tab_004'
    prefix = files[0].rsplit('.', 1)[0]

    # Read file 0 to get header info (TotNgroups, NFiles)
    file0 = os.path.join(catalog_dir, f'{prefix}.0')
    data0, tot_ngroups, nfiles = _read_fof_file(file0)

    if nfiles == 1:
        return data0

    # Multi-file: read remaining files and concatenate
    chunks = [data0]
    for i in range(1, nfiles):
        filename = os.path.join(catalog_dir, f'{prefix}.{i}')
        chunk, _, _ = _read_fof_file(filename)
        chunks.append(chunk)

    return {
        'npart': np.concatenate([c['npart'] for c in chunks]),
        'mass': np.concatenate([c['mass'] for c in chunks]),
        'pos': np.concatenate([c['pos'] for c in chunks], axis=0),
        'vel': np.concatenate([c['vel'] for c in chunks], axis=0),
    }


def _detect_format(filename):
    """Detect whether a group_tab file uses Fortran record markers or plain binary.

    Returns 'fortran' or 'plain'.
    """
    fsize = os.path.getsize(filename)
    with open(filename, 'rb') as f:
        first_word = struct.unpack('<I', f.read(4))[0]

    # In Fortran-record format, first word is the header record length (16 bytes = 4 uint32).
    # In plain format, first word is Ngroups.
    # Heuristic: if first_word == 16, it's Fortran-record format.
    # Otherwise, check if plain format is consistent with the file size.
    if first_word == 16:
        return 'fortran'

    # For plain format: header is 6 uint32 (24 bytes), then Ngroups * 84 bytes of data.
    # (GroupLen + GroupOffset + GroupMass + GroupPos + GroupVel + GroupLenType + GroupMassType
    #  = 4 + 4 + 4 + 12 + 12 + 24 + 24 = 84 bytes per halo)
    ngroups = first_word
    expected_plain = 24 + ngroups * 84
    if fsize == expected_plain:
        return 'plain'

    # Default to Fortran
    return 'fortran'


def _read_fof_file(filename):
    """Read a single FoF group_tab file. Auto-detects format.

    Returns:
        (dict, tot_ngroups, nfiles) where dict has keys 'npart', 'mass', 'pos', 'vel'
    """
    fmt = _detect_format(filename)
    if fmt == 'plain':
        return _read_fof_file_plain(filename)
    else:
        return _read_fof_file_fortran(filename)


def _read_fof_file_plain(filename):
    """Read a plain binary (no Fortran records) FoF group_tab file.

    File layout:
      Header:         Ngroups(u32) TotNgroups(u32) Nids(u32) TotNids(u32) ?(u32) NFiles(u32)
      GroupLen:        int32[Ngroups]
      GroupOffset:     int32[Ngroups]
      GroupMass:       float32[Ngroups]
      GroupPos:        float32[Ngroups*3]
      GroupVel:        float32[Ngroups*3]
      GroupLenType:    int32[Ngroups*6]
      GroupMassType:   float32[Ngroups*6]
    """
    with open(filename, 'rb') as f:
        header = np.fromfile(f, dtype=np.uint32, count=6)
        ngroups = int(header[0])
        tot_ngroups = int(header[1])
        nfiles = int(header[5])

        if ngroups == 0:
            result = {
                'npart': np.array([], dtype=np.int32),
                'mass': np.array([], dtype=np.float32),
                'pos': np.zeros((0, 3), dtype=np.float32),
                'vel': np.zeros((0, 3), dtype=np.float32),
            }
            return result, tot_ngroups, nfiles

        npart = np.fromfile(f, dtype=np.int32, count=ngroups)        # GroupLen
        np.fromfile(f, dtype=np.int32, count=ngroups)                 # GroupOffset (skip)
        mass = np.fromfile(f, dtype=np.float32, count=ngroups)        # GroupMass
        pos = np.fromfile(f, dtype=np.float32, count=ngroups * 3)     # GroupPos
        vel = np.fromfile(f, dtype=np.float32, count=ngroups * 3)     # GroupVel
        # GroupLenType and GroupMassType are not needed — skip.

    result = {
        'npart': npart,
        'mass': mass,
        'pos': pos.reshape(ngroups, 3),
        'vel': vel.reshape(ngroups, 3),
    }
    return result, tot_ngroups, nfiles


def _read_fof_file_fortran(filename):
    """Read a Fortran-record-wrapped FoF group_tab file.

    File layout (Fortran-record-wrapped):
      Header:       Ngroups(u32) Nids(u32) TotNgroups(u32) NFiles(u32)
      GroupLen:      int32[Ngroups]
      GroupOffset:   int32[Ngroups]     (skipped)
      GroupMass:     float32[Ngroups]
      GroupPos:      float32[Ngroups*3]
      GroupVel:      float32[Ngroups*3]
    """
    with open(filename, 'rb') as f:
        header = read_fortran_record(f, np.uint32, 4)
        ngroups, nids, tot_ngroups, nfiles = header

        if ngroups == 0:
            result = {
                'npart': np.array([], dtype=np.int32),
                'mass': np.array([], dtype=np.float32),
                'pos': np.zeros((0, 3), dtype=np.float32),
                'vel': np.zeros((0, 3), dtype=np.float32),
            }
            return result, int(tot_ngroups), int(nfiles)

        npart = read_fortran_record(f, np.int32, ngroups)
        skip_fortran_record(f)  # GroupOffset
        mass = read_fortran_record(f, np.float32, ngroups)
        pos = read_fortran_record(f, np.float32, ngroups * 3)
        vel = read_fortran_record(f, np.float32, ngroups * 3)

    result = {
        'npart': npart,
        'mass': mass,
        'pos': pos.reshape(ngroups, 3),
        'vel': vel.reshape(ngroups, 3),
    }
    return result, int(tot_ngroups), int(nfiles)
