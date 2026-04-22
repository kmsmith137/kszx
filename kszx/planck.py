"""Planck data products supported by kszx: HFI galactic masks (PR2) and PR3 beam window functions."""

import os
import fitsio
import healpy
import numpy as np
from astropy.io import fits

from . import io_utils

    
def read_hfi_galmask(sky_percentage, apodization=0, dtype=None, download=False):
    """Returns an nside=2048 healpix map in RING ordering and Galactic coordinates.

    Default dtype is either uint8 or float32 (depending on whether apodization > 0), 
    but this can be changed with the ``dtype`` argument.

    Allowed ``sky_percentage`` values: 20, 40, 60, 70, 80, 90, 97, 99
    Allowed ``apodization`` values: 0, 2, 5 (degrees)
    If ``download`` is True, then data files will be auto-downloaded.

    Note that we use Planck release 2, since release 3 doesn't seem to have HFI
    foreground masks (it does have other masks).
    
    To apply a Planck mask to ACT data, you'll need to rotate/pixellize the mask::

       pixell_mask = pixell.reproject.healpix2map(
          healpix_mask, shape, wcs, 
          rot='gal,equ',                # NOTE coordinate rotation!!
          method='spline', order=0)     # NOTE method='spline', not method='harm'!
    """

    assert sky_percentage in [ 20, 40, 60, 70, 80, 90, 97, 99 ]
    
    abspath = _hfi_galmask_filename(apodization, download, dlfunc='kszx.planck.read_hfi_galmask')
    col_name = f'GAL0{sky_percentage}'
    print(f'Reading {abspath} ({col_name=})\n', end='')
    
    with fitsio.FITS(abspath) as f:
        if col_name not in f[1].get_colnames():
            raise RuntimeError(f'{abspath}: column name {col_name} not found')
        mask = f[1][col_name].read()

    # Convert from Healpix NESTED (in fits file) to RING (assumed by pixell)
    mask = healpy.pixelfunc.reorder(mask, n2r=True)

    if dtype is not None:
        mask = np.asarray(mask, dtype)
    elif apodization > 0:
        # Convert from big-endian float32 (numpy dtype '>f4') to native float32
        mask = np.asarray(mask, np.float32)
        
    return mask


def read_beam(freq, freq2=None, release=3, *, pol='T', lmax=None, download=False):
    r"""Returns the Planck PR3 effective-beam window function as a 1-d numpy array of length (lmax+1).

    Function args:

      - ``freq`` (integer): frequency channel in GHz.
        Allowed: 30, 44, 70 (LFI) or 100, 143, 217, 353, 545, 857 (HFI).
      - ``freq2`` (integer or None): if specified, returns the cross-beam for freq x freq2.
        Both frequencies must be LFI, or both HFI (LFI x HFI crosses are not provided in PR3).
        LFI does not provide cross-channel beams; freq2 must equal freq in the LFI case.
        The 353p channel (the 353 GHz polarization-sensitive bolometer subset) is also available,
        but only crossed with {100, 353, 545, 857} in PR3 (use freq or freq2 = '353p' as a string).
      - ``release`` (integer): Planck release number. Currently only 3 (PR3, 2018) is supported.
      - ``pol`` (string): 'T', 'E', or 'B'. Default 'T'. 'E'/'B' are only available for HFI
        PSB-containing channels (100-353 GHz, also 353p) in PR3.
        (Note: 'TE' is NOT a column in the Bl_TEB_* files, so it's not accepted here; it lives in
        the separate Wl_* beam-matrix products, which are out of scope for read_beam().)
      - ``lmax`` (integer): if None, returns the full range from the file (4000 for HFI,
        2048 for LFI). If specified, the returned array is truncated to length ``lmax+1``.
      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded
        (HFI is ~87MB tarball, unpacked + cached; LFI is a ~770kB single FITS file).

    Returns:
      1-d numpy.ndarray of length (lmax+1), indexed by multipole. b[0] is normalized to 1.
      (HFI files are already normalized at the source; LFI 'BL' values in the RIMO are close to
      but not exactly 1 at ell=0 due to residual calibration factors, and we divide by b[0] to
      match the ACT and HFI convention.)
    """
    if release != 3:
        raise RuntimeError(f'Planck release={release} not supported (only PR3 is currently implemented)')
    if freq2 is None:
        freq2 = freq

    filename, col = _beam_filename(freq, freq2, pol, download=download, dlfunc='kszx.planck.read_beam')
    print(f'Reading {filename} ({col=})\n', end='')

    with fits.open(filename) as h:
        if isinstance(col, str):
            # HFI: single HDU (index 1) with one or three named columns.
            b_ell = np.asarray(h[1].data[col], dtype=float)
        else:
            # LFI: tuple (extname, colname). Select the HDU by EXTNAME.
            extname, colname = col
            b_ell = np.asarray(h[extname].data[colname], dtype=float)
            b_ell = b_ell / b_ell[0]   # LFI BL is not self-normalized to 1; fix to match ACT.

    if lmax is not None:
        assert lmax < len(b_ell), f'lmax={lmax} exceeds max available multipole {len(b_ell)-1}'
        b_ell = b_ell[:(lmax+1)]
    return b_ell


def download():
    r"""Downloads Planck galmasks and beam window functions.

    Can be called from command line: ``python -m kszx download_planck``."""

    for apodization in [0, 2, 5]:
        _hfi_galmask_filename(apodization, download=True)

    # HFI beams: downloading one inside-the-tarball file triggers download+unpack of the whole tarball.
    _beam_filename(100, 100, 'T', download=True)
    # LFI beams: single RIMO file contains all three frequencies.
    _beam_filename(30, 30, 'T', download=True)


####################################################################################################


def _planck_path(relpath, *, aux_relpath=None, download=False, dlfunc=None):
    """IRSA-backed Planck downloader. Returns absolute path to ``relpath`` under the
    kszx data directory, downloading if needed.

    If ``aux_relpath`` is given, it's the path (relative to the same IRSA Planck root) of a
    tarball/zip archive that contains ``relpath``; we download+unpack that instead of
    downloading ``relpath`` directly. Mirrors the ``kszx.act._act_path`` pattern.

    Here and in other parts of kszx, the 'dlfunc' argument gives the name of a transitive caller that
    expects the file to be present, and has a 'download=False' optional argument. This information is
    only used when generating exception-text (to tell the user how to download the file).
    """

    planck_base_dir = os.path.join(io_utils.get_data_dir(), 'planck')
    abspath = os.path.join(planck_base_dir, relpath)

    if not io_utils.do_download(abspath, download, dlfunc):
        return abspath

    if aux_relpath is None:
        url = f'https://irsa.ipac.caltech.edu/data/Planck/{relpath}'
        io_utils.wget(abspath, url)
    else:
        aux_abspath = _planck_path(aux_relpath, download=download, dlfunc=dlfunc)
        io_utils.unpack(aux_abspath, expected_dstfile=abspath)

    return abspath


def _hfi_galmask_filename(apodization, download=False, dlfunc=None):
    """Allowed apodization values: 0, 2, 5 (degrees).

    Note that we use Planck release 2, since release 3 doesn't seem to have HFI
    foreground masks (it does have other masks).
    """

    assert apodization in [0, 2, 5]
    relpath = f'release_2/ancillary-data/masks/HFI_Mask_GalPlane-apo{apodization}_2048_R2.00.fits'
    return _planck_path(relpath, download=download, dlfunc=dlfunc)


# HFI channels with polarized PSBs (support Bl_TEB) -- "353p" handled as a string sentinel.
_HFI_FREQS      = (100, 143, 217, 353, '353p', 545, 857)
_HFI_PSB_FREQS  = (100, 143, 217, 353, '353p')
_LFI_FREQS      = (30, 44, 70)

# Not every HFI cross is provided in PR3. Explicitly whitelist (f_lo, f_hi) pairs (after sort).
_HFI_PAIRS_PROVIDED = {
    (100, 100), (100, 143), (100, 217), (100, 353), (100, '353p'), (100, 545), (100, 857),
    (143, 143), (143, 217), (143, 353),                            (143, 545), (143, 857),
                (217, 217), (217, 353),                            (217, 545), (217, 857),
                            (353, 353),                            (353, 545), (353, 857),
                                        ('353p', '353p'),          ('353p', 545), ('353p', 857),
                                                                   (545, 545),   (545, 857),
                                                                                 (857, 857),
}


def _beam_pair_sorted(freq, freq2):
    """Sort (freq, freq2) to canonical ordering used in the Planck filenames. Treats '353p' > 353."""
    def _key(f):
        return (int(f[:-1]) + 0.5) if (isinstance(f, str) and f.endswith('p')) else float(f)
    return tuple(sorted([freq, freq2], key=_key))


def _beam_filename(freq, freq2, pol, download=False, dlfunc=None):
    """Returns (filename, column_selector) for the requested Planck PR3 beam.

    For HFI, ``column_selector`` is a column name (str) to read from HDU 1.
    For LFI, it's a tuple (extname, colname) since the RIMO FITS holds many HDUs.
    """

    if pol not in ('T', 'E', 'B'):
        raise RuntimeError(f'pol={pol!r} is not supported (must be "T", "E", or "B")')

    is_hfi = (freq in _HFI_FREQS) and (freq2 in _HFI_FREQS)
    is_lfi = (freq in _LFI_FREQS) and (freq2 in _LFI_FREQS)

    if is_hfi:
        f_lo, f_hi = _beam_pair_sorted(freq, freq2)
        if (f_lo, f_hi) not in _HFI_PAIRS_PROVIDED:
            raise RuntimeError(f'HFI cross beam ({f_lo} x {f_hi}) is not provided in PR3')

        if pol == 'T':
            sub, col = 'T', 'TEMPERATURE'
        else:  # 'E' or 'B'
            if (freq not in _HFI_PSB_FREQS) or (freq2 not in _HFI_PSB_FREQS):
                raise RuntimeError(f'pol={pol!r} only available for HFI PSB channels (100,143,217,353,353p)')
            sub, col = 'TEB', pol

        relpath = (f'release_3/ancillary-data/BeamWf_HFI_R3.01/'
                   f'Bl_{sub}_R3.01_fullsky_{f_lo}x{f_hi}.fits')
        aux_relpath = 'release_3/ancillary-data/HFI_RIMO_BEAMS_R3.01.tar.gz'
        filename = _planck_path(relpath, aux_relpath=aux_relpath, download=download, dlfunc=dlfunc)
        return filename, col

    if is_lfi:
        if pol != 'T':
            raise RuntimeError(f'LFI RIMO only provides temperature beams (got pol={pol!r})')
        if freq != freq2:
            raise RuntimeError('LFI cross-channel beams are not provided in PR3')
        relpath = 'release_3/ancillary-data/LFI_RIMO_R3.31.fits'
        filename = _planck_path(relpath, download=download, dlfunc=dlfunc)
        extname = f'BEAMWF_{freq:03d}X{freq:03d}'
        return filename, (extname, 'BL')

    raise RuntimeError(f'No PR3 beam for (freq={freq!r}, freq2={freq2!r}) '
                       '(mixed LFI x HFI crosses are not provided)')
