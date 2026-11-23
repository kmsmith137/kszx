"""Filenames (and optional downloading) for Legacy Survey DR10/DR11 randoms and sweep catalogs.

Data model: https://www.legacysurvey.org/dr11/files/ (and .../dr10/files/).

Note: the photo-z sweeps (suffix='pz') contain IDs and Z_PHOT_* columns, but no RA/DEC. They are
row-matched to the main sweeps (suffix=None), which must also be read to get sky locations.

Note: the full DR11 sweeps are large (photo-z ~380 GB, main ~2 TB, north+south). At NERSC, the
files are already on disk, and can be used without downloading via a symlink, e.g.:

   mkdir -p $KSZX_DATA_DIR/desils
   ln -s /global/cfs/cdirs/cosmo/data/legacysurvey/dr11 $KSZX_DATA_DIR/desils/dr11
"""

import os
from . import io_utils


def download_randoms(dr):
    for i in range(20):
        _desils_path(f'randoms/randoms-1-{i}.fits', dr, download=True)


def download_south_randoms(dr):
    for i in range(20):
        _desils_path(f'south/randoms/randoms-south-1-{i}.fits', dr, download=True)


def sweep_filename(brickname, region, dr, suffix=None, download=False, dlfunc=None):
    """Returns the filename of one sweep file, e.g. sweep_filename('000m005-005p000', 'south', 11, 'pz').

    Function arguments:

      - ``brickname`` (str): '<brickmin>-<brickmax>', e.g. '000m005-005p000'.
      - ``region`` (str): either 'north' or 'south' (DR10 is south-only).
      - ``dr`` (int): either 10 or 11.
      - ``suffix`` (str or None): None for the main sweeps, 'pz' for photo-z,
        'ex' for extra columns, or 'lc' for light curves.
      - ``download`` (boolean): if True, then the file will be auto-downloaded.
    """

    assert region in ('north', 'south')
    subdirs = { None: '', 'pz': '-photo-z', 'ex': '-extra', 'lc': '-lightcurves' }
    assert suffix in subdirs

    subdir = _sweep_version(dr) + subdirs[suffix]
    basename = f'sweep-{brickname}.fits' if (suffix is None) else f'sweep-{brickname}-{suffix}.fits'
    return _desils_path(f'{region}/sweep/{subdir}/{basename}', dr, download=download, dlfunc=dlfunc)


def list_sweep_bricknames(region, dr, suffix=None):
    """Returns sorted list of '<brickmin>-<brickmax>' names, for sweep files found on local disk (no downloading)."""

    dirname = os.path.dirname(sweep_filename('x', region, dr, suffix))
    tail = '.fits' if (suffix is None) else f'-{suffix}.fits'
    filenames = [ f for f in os.listdir(dirname) if f.startswith('sweep-') and f.endswith(tail) ]
    return sorted(f[len('sweep-'):-len(tail)] for f in filenames)


def _sweep_version(dr):
    versions = { 10: '10.1', 11: '11.0' }
    assert dr in versions
    return versions[dr]


def _desils_path(relpath, dr, download=False, dlfunc=None):
    """Example: _desils_path('randoms/randoms-1-0.fits', dr=11).

    Intended to be called through wrapper, e.g. sweep_filename().

    Here and in other parts of kszx, the 'dlfunc' argument gives the name of a transitive caller that
    expects the file to be present, and has a 'download=False' optional argument. This information is
    only used when generating exception-text (to tell the user how to download the file).
    """

    assert dr in (10, 11)
    relpath = os.path.join(f'dr{dr}', relpath)

    desils_base_dir = os.path.join(io_utils.get_data_dir(), 'desils')
    abspath = os.path.join(desils_base_dir, relpath)

    if io_utils.do_download(abspath, download, dlfunc):
        url = f'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(...) after downloading

    return abspath
