"""Planck data products supported by kszx:

  - HFI Galactic-plane masks (PR2) -- see read_hfi_galmask().
  - Frequency sky maps (PR4 / NPIPE by default, PR3 optional) -- see read_cmb().
  - Component-separated CMB maps (SMICA / NILC / COMMANDER / SEVEM, PR3) -- see read_cmb().
  - Effective beam window functions for all of the above -- see read_beam().

Hosting split:

  - PR2 and PR3 are mirrored on IRSA (irsa.ipac.caltech.edu/data/Planck/).
  - PR4 (NPIPE) is hosted on the NERSC community portal
    (portal.nersc.gov/cfs/cmb/planck2020/). IRSA does not mirror PR4.
  - Both are plain HTTPS (wget-friendly, no Globus or NERSC login).

Local cache layout: every product ends up under $KSZX_DATA/planck/<relpath>,
where <relpath> is the URL path below the host root. That gives separate
trees for release_2/, release_3/, and planck2020/ with no extra branching.
"""

import os
import fitsio
import healpy
import numpy as np
from astropy.io import fits

from . import io_utils


_IRSA_BASE  = 'https://irsa.ipac.caltech.edu/data/Planck'
_NERSC_BASE = 'https://portal.nersc.gov/cfs/cmb'


def read_hfi_galmask(sky_percentage, apodization=0, dtype=None, download=False):
    """Returns an nside=2048 healpix map in RING ordering and Galactic coordinates.

    Default dtype is either uint8 or float32 (depending on whether apodization > 0),
    but this can be changed with the ``dtype`` argument.

    Allowed ``sky_percentage`` values: 20, 40, 60, 70, 80, 90, 97, 99
    Allowed ``apodization`` values: 0, 2, 5 (degrees)
    If ``download`` is True, then data files will be auto-downloaded.

    Note that we use Planck release 2 (PR2) for the HFI Galactic-plane masks,
    regardless of which Planck release is selected elsewhere in ``kszx.planck``.
    Neither PR3 nor PR4/NPIPE re-release the percentile Galactic-plane masks,
    and the sky foreground they describe is not pipeline-dependent.

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


def read_cmb(freq, release=4, *, pol='T', download=False):
    r"""Returns a Planck sky map in thermodynamic microkelvin (μK_CMB), as a
    1-D HEALPix array in RING order and Galactic coordinates.

    Function args:

      - ``freq``: one of
          - an integer frequency channel in GHz: one of {30, 44, 70} (LFI)
            or {100, 143, 217, 353, 545, 857} (HFI), OR
          - a string CMB-solution name: one of {'smica','nilc','commander','sevem'}.

      - ``release`` (integer): Planck release number. 4 = PR4 / NPIPE (default,
        hosted on NERSC); 3 = PR3 (IRSA). For the CMB-solution strings the file
        is the PR3 product regardless of ``release`` (NPIPE has no official
        component-separation release).

      - ``pol`` (string): 'T', 'Q', or 'U'. Default 'T'. 'Q'/'U' are available
        for LFI, HFI 100-353, and all CMB solutions; HFI 545 and 857 are
        intensity-only under both PR3 and PR4.

      - ``download`` (boolean): if True, auto-download the FITS file from IRSA
        (PR3) or NERSC (PR4). Individual NPIPE frequency maps are ~500 MB (LFI,
        HFI 545/857) up to ~2 GB (HFI 100-353), so beware of disk/bandwidth.

    Returns:
      1-D numpy.ndarray of HEALPix pixels in μK_CMB (thermodynamic). float32,
      native endianness, RING order. nside is 1024 for LFI frequency maps;
      2048 for HFI frequency maps and all CMB solutions.

    Unit handling (internal):
      - PR4: NPIPE writes every channel in K_CMB on disk (including 545 and 857,
        which were MJy/sr in PR3). Convert with × 1e6.
      - PR3 LFI / HFI 100-353 / CMB solutions: natively K_CMB; × 1e6.
      - PR3 HFI 545/857: natively MJy/sr; × (1e6 / Kcmb_MJy[freq]) using the
        Planck 2018 bandpass-integrated factors.

    Bad pixels: NPIPE and PR3 both store invalid pixels as ``BAD_DATA = -1.6375e+30``
    (the HEALPix UNSEEN convention). Callers who compute statistics should mask
    these out explicitly.

    Planck sky maps are in Galactic coordinates. To reproject onto a pixell CAR
    map (e.g. for stacking on ACT data) with a coordinate rotation::

        pixell_map = pixell.reproject.healpix2map(
            healpix_map, shape, wcs,
            rot='gal,equ',                # NOTE coordinate rotation
            method='spline', order=0)
    """

    if release not in (3, 4):
        raise RuntimeError(f'Planck release={release} not supported (expected 3 or 4)')

    filename, col = _cmb_filename(freq, pol, release, download=download,
                                  dlfunc='kszx.planck.read_cmb')
    print(f'Reading {filename} ({col=})\n', end='')

    with fitsio.FITS(filename) as f:
        if col not in f[1].get_colnames():
            raise RuntimeError(f'{filename}: column name {col!r} not found '
                               f'(available columns: {f[1].get_colnames()})')
        m = f[1][col].read()

    # NPIPE stores HEALPix maps in '1024E' chunked format (each row is 1024
    # pixels), which fitsio returns as a 2-D (Nrows, 1024) ndarray. PR3 used
    # '1E' (one pixel per row) and comes back 1-D. Flatten to 1-D unconditionally.
    m = np.ravel(m)

    # Planck FITS is NESTED; convert to RING (matches pixell / ACT conventions).
    m = healpy.pixelfunc.reorder(m, n2r=True)
    m = np.asarray(m, dtype=np.float32)

    # Convert to thermodynamic μK_CMB. The only case that needs a non-1e6 factor
    # is PR3 HFI 545/857, which is natively MJy/sr.
    if release == 3 and freq in _HFI_TONLY_MAP_FREQS:
        m *= np.float32(1.0e6 / _KCMB_TO_MJY_PER_SR[freq])
    else:
        m *= np.float32(1.0e6)

    return m


def read_beam(freq, freq2=None, release=4, *, pol='T', lmax=None, download=False):
    r"""Returns a Planck effective-beam window function as a 1-d numpy array of length (lmax+1).

    Function args:

      - ``freq``: one of
          - an integer frequency channel in GHz: {30, 44, 70} (LFI) or
            {100, 143, 217, 353, 545, 857} (HFI), OR
          - a string CMB-solution name: one of {'smica','nilc','commander','sevem'}.

      - ``freq2`` (integer or None): only meaningful for integer ``freq``. If
        specified, returns the cross-beam for (freq × freq2). Default: same as
        ``freq`` (auto-beam). For CMB-solution strings ``freq2`` must be None.

        PR4 (NPIPE) provides auto + cross for all 81 pairs in T, and T+E+B for
        the 49 pairs covering {30..353} GHz. LFI × HFI crosses are available
        (new vs PR3). PR3 provides HFI auto + cross for T/TEB restricted to
        some sub-set, and LFI auto only; LFI × HFI crosses are not provided.

      - ``release`` (integer): Planck release number. 4 = PR4 / NPIPE (default,
        NERSC); 3 = PR3 (IRSA). Ignored for CMB-solution strings (always PR3).

      - ``pol`` (string): 'T', 'E', or 'B'. Default 'T'. For frequency channels,
        E/B are only available for 30-353 GHz in PR4 and for HFI 100-353 (and
        353p) in PR3. For CMB-solution strings, 'T' reads the ``INT_BEAM``
        column and 'E'/'B' both read ``POL_BEAM`` (the file stores a single
        common polarization TF).

      - ``lmax`` (integer): if None, returns the full range from the file.
        Full ranges: PR4 LFI auto 4096, HFI auto 8192, LFI × HFI cross 4096;
        PR3 LFI 2048, PR3 HFI 4000; CMB-solution beams 4096.

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded
        (PR4 RIMO is a ~224 MB tarball, unpacked + cached; PR3 HFI is ~87MB tarball;
        PR3 LFI is a ~770kB single FITS file). CMB-solution beams ride inside the
        same ~2 GB FITS file that ``read_cmb`` uses, and download with it.

    Returns:
      1-d numpy.ndarray of length (lmax+1), indexed by multipole. For frequency
      channels ``b[0] = 1``. For CMB-solution strings ``b[0] = 0`` -- the
      component-separation pipelines don't reconstruct the monopole, so the
      stored transfer function is deliberately zero at ℓ=0. **Do not divide by
      b[0]** for CMB-solution beams; apply the TF to C_ell directly.

    Note on SEVEM: the SEVEM ``COM_CMB_IQU`` file ships an empty BEAM_TF table
    (NAXIS2=0, LMAX_I=-1). Per Planck 2018 IV, the SEVEM combined CMB map has
    a 5' FWHM Gaussian effective beam; this function substitutes that analytic
    Gaussian (also with b[0]=0) when the table is empty.
    """
    if release not in (3, 4):
        raise RuntimeError(f'Planck release={release} not supported (expected 3 or 4)')

    is_solution = isinstance(freq, str) and freq.lower() in _CMB_SOLUTIONS
    if not is_solution and freq2 is None:
        freq2 = freq

    filename, col = _beam_filename(freq, freq2, pol, release,
                                   download=download, dlfunc='kszx.planck.read_beam')
    print(f'Reading {filename} ({col=})\n', end='')

    with fits.open(filename) as h:
        if isinstance(col, tuple):
            # Two-element tuple identifying (HDU, column):
            #   - PR3 LFI uses string EXTNAME (e.g. 'BEAMWF_030X030') + 'BL'.
            #   - CMB-solution branch uses int HDU index 2 + 'INT_BEAM' / 'POL_BEAM'
            #     (the COM_CMB_IQU files do not assign an EXTNAME to HDU 2).
            extname_or_idx, colname = col
            b_ell = np.asarray(h[extname_or_idx].data[colname], dtype=float)
            if (release == 3 and isinstance(extname_or_idx, str)
                    and extname_or_idx.startswith('BEAMWF_')):
                # PR3 LFI 'BL' values in the RIMO are close to but not exactly 1 at ell=0
                # due to residual calibration factors; renormalize to match the ACT and HFI
                # convention b[0] = 1.
                b_ell = b_ell / b_ell[0]
            elif is_solution and len(b_ell) == 0:
                # SEVEM's COM_CMB_IQU file ships an empty BEAM_TF table (NAXIS2=0,
                # LMAX_I=-1). Per Planck 2018 IV / the PLA wiki, the SEVEM combined
                # CMB map has a 5' FWHM Gaussian effective beam; substitute that.
                # Match the b[0]=0 monopole convention used by the other solutions.
                lmax_full = int(h[extname_or_idx].header.get('LMAX_I', 4096))
                if lmax_full < 0:
                    lmax_full = 4096
                ell_full = np.arange(lmax_full + 1)
                sigma_rad = (5.0 * np.pi / (180.0 * 60.0)) / np.sqrt(8.0 * np.log(2.0))
                b_ell = np.exp(-0.5 * ell_full * (ell_full + 1) * sigma_rad**2)
                b_ell[0] = 0.0
                print(f'  [{freq}] BEAM_TF table is empty; substituting analytic '
                      f'5\' FWHM Gaussian (Planck 2018 IV).\n', end='')
        else:
            # HFI (PR3 or PR4): single HDU at index 1 with one or three named columns,
            # already self-normalized to b[0] = 1.
            b_ell = np.asarray(h[1].data[col], dtype=float)

    if lmax is not None:
        assert lmax < len(b_ell), f'lmax={lmax} exceeds max available multipole {len(b_ell)-1}'
        b_ell = b_ell[:(lmax+1)]
    return b_ell


def download():
    r"""Downloads every Planck product read by default ``read_*`` calls.

    Total size ~18 GB::

      - PR2 HFI Galactic-plane masks (3 apodizations)     ~384 MB
      - PR4 NPIPE RIMO beam tarball (all beam products)    224 MB
      - PR4 NPIPE frequency maps (9 channels)            ~10.5 GB
      - PR3 CMB component-separation maps (4 methods)      ~7 GB

    Intended to prime the kszx cache for offline use. Beams for the four
    CMB component-separation products live inside HDU 2 of the same files
    as the maps, so this one pass also caches them.

    PR3 frequency maps and PR3 beams are NOT prefetched (the user will get
    them on-demand via ``read_*(..., release=3, download=True)`` if needed).
    Split products (ring-half, detector-set, LFI year) are also deferred.

    Can be called from command line: ``python -m kszx download_planck``.
    """

    # PR2 HFI Galactic-plane masks.
    for apodization in [0, 2, 5]:
        _hfi_galmask_filename(apodization, download=True)

    # PR4 NPIPE RIMO tarball: a single 224 MB fetch gives every beam file
    # (81 T + 49 TEB + 49 Wl). One _beam_filename call triggers the tarball
    # download + unpack.
    _beam_filename(100, 100, 'T', release=4, download=True)

    # PR4 NPIPE frequency maps (all 9 channels).
    for freq in _LFI_FREQS + _HFI_MAP_FREQS:
        _cmb_filename(freq, 'T', release=4, download=True,
                      dlfunc='kszx.planck.download')

    # PR3 CMB component-separation maps (all 4 methods). Beams for these
    # products live in HDU 2 of the same FITS files.
    for method in _CMB_SOLUTIONS:
        _cmb_filename(method, 'T', release=3, download=True,
                      dlfunc='kszx.planck.download')


####################################################################################################


def _planck_path(relpath, *, base_url, aux_relpath=None, download=False, dlfunc=None):
    """Generic Planck downloader, base_url-parameterized.

    - URL resolved as ``f'{base_url}/{relpath}'``.
    - Local path is always ``$KSZX_DATA/planck/{relpath}``, independent of
      base_url. The relpath prefix (``release_2/``, ``release_3/``,
      ``planck2020/``) naturally separates releases on disk.
    - ``aux_relpath``, if given, names a tarball/zip (relative to the same
      base_url) that contains relpath; we download+unpack that instead of
      fetching relpath directly. Mirrors the ``kszx.act._act_path`` pattern.

    Here and in other parts of kszx, the 'dlfunc' argument gives the name of a transitive caller that
    expects the file to be present, and has a 'download=False' optional argument. This information is
    only used when generating exception-text (to tell the user how to download the file).
    """

    abspath = os.path.join(io_utils.get_data_dir(), 'planck', relpath)

    if not io_utils.do_download(abspath, download, dlfunc):
        return abspath

    if aux_relpath is None:
        io_utils.wget(abspath, f'{base_url}/{relpath}')
    else:
        aux_abspath = _planck_path(aux_relpath, base_url=base_url,
                                   download=download, dlfunc=dlfunc)
        io_utils.unpack(aux_abspath, expected_dstfile=abspath)

    return abspath


def _irsa_path(relpath, **kwargs):
    """IRSA-hosted Planck file (PR2 / PR3)."""
    return _planck_path(relpath, base_url=_IRSA_BASE, **kwargs)


def _nersc_path(relpath, **kwargs):
    """NERSC-hosted Planck file (PR4 / NPIPE)."""
    return _planck_path(relpath, base_url=_NERSC_BASE, **kwargs)


def _hfi_galmask_filename(apodization, download=False, dlfunc=None):
    """Allowed apodization values: 0, 2, 5 (degrees).

    Uses the PR2 galmask; neither PR3 nor PR4 re-release these files.
    """

    assert apodization in [0, 2, 5]
    relpath = f'release_2/ancillary-data/masks/HFI_Mask_GalPlane-apo{apodization}_2048_R2.00.fits'
    return _irsa_path(relpath, download=download, dlfunc=dlfunc)


####################################################################################################
# Constants.

# Planck CMB-component-separation products (PR3; NPIPE has no official equivalent).
_CMB_SOLUTIONS        = ('smica', 'nilc', 'commander', 'sevem')

# Planck PR3/PR4 sky-map frequency channels.
_LFI_FREQS            = (30, 44, 70)
_HFI_MAP_FREQS        = (100, 143, 217, 353, 545, 857)
_HFI_TONLY_MAP_FREQS  = (545, 857)                                # no Q/U columns

# PR3 beam products: HFI channels with polarized PSBs support Bl_TEB.
# "353p" is the 353 GHz polarization-sensitive bolometer subset; handled
# as a string sentinel (and only meaningful in PR3 beam filenames).
_PR3_HFI_BEAM_FREQS    = (100, 143, 217, 353, '353p', 545, 857)
_PR3_HFI_PSB_FREQS     = (100, 143, 217, 353, '353p')

# Planck 2018 bandpass-integrated K_CMB -> MJy/sr conversion factors.
# Only 545/857 are actually needed by read_cmb (for PR3 only, where those
# channels are natively MJy/sr). The full set is kept for reference; these
# match the Kcmb_MJy array in halomodel_cib_tsz_cibxtsz/Cell_tSZ.py.
_KCMB_TO_MJY_PER_SR = {100: 244.1, 143: 371.74, 217: 483.69, 353: 287.45, 545: 58.04, 857: 2.27}


####################################################################################################
# Helpers for read_cmb().


def _cmb_filename(freq, pol, release, download=False, dlfunc=None):
    """Returns (filename, colname) for the requested Planck sky map.

    ``colname`` is the name of the HEALPix column in HDU 1 to read (e.g.
    'I_STOKES'). HDU structure is standard Planck NESTED HEALPix; the caller
    (``read_cmb``) reorders to RING.
    """
    if pol not in ('T', 'Q', 'U'):
        raise RuntimeError(f'pol={pol!r} is not supported (must be "T", "Q", or "U")')
    col = {'T': 'I_STOKES', 'Q': 'Q_STOKES', 'U': 'U_STOKES'}[pol]

    # CMB-solution strings: always PR3 on IRSA, regardless of `release`.
    if isinstance(freq, str) and freq.lower() in _CMB_SOLUTIONS:
        relpath = (f'release_3/all-sky-maps/maps/component-maps/cmb/'
                   f'COM_CMB_IQU-{freq.lower()}_2048_R3.00_full.fits')
        return _irsa_path(relpath, download=download, dlfunc=dlfunc), col

    # 545/857 are intensity-only under both PR3 and PR4.
    if pol != 'T' and freq in _HFI_TONLY_MAP_FREQS:
        raise RuntimeError(f'HFI {freq} GHz map is intensity-only; no Q/U data')

    if release == 4:
        if freq in _LFI_FREQS:
            relpath = (f'planck2020/frequency_maps/Single-frequency/'
                       f'LFI_SkyMap_{freq:03d}_1024_R4.00_full.fits')
        elif freq in _HFI_MAP_FREQS:
            relpath = (f'planck2020/frequency_maps/Single-frequency/'
                       f'HFI_SkyMap_{freq:03d}_2048_R4.00_full.fits')
        else:
            raise RuntimeError(
                f'freq={freq!r} is not a valid Planck channel or CMB-solution name. '
                f'Expected an integer in {_LFI_FREQS + _HFI_MAP_FREQS} '
                f'or one of {_CMB_SOLUTIONS}.')
        return _nersc_path(relpath, download=download, dlfunc=dlfunc), col

    if release == 3:
        if freq in _LFI_FREQS:
            relpath = f'release_3/all-sky-maps/maps/LFI_SkyMap_{freq:03d}_1024_R3.00_full.fits'
        elif freq in _HFI_MAP_FREQS:
            relpath = f'release_3/all-sky-maps/maps/HFI_SkyMap_{freq:03d}_2048_R3.01_full.fits'
        else:
            raise RuntimeError(
                f'freq={freq!r} is not a valid Planck channel or CMB-solution name. '
                f'Expected an integer in {_LFI_FREQS + _HFI_MAP_FREQS} '
                f'or one of {_CMB_SOLUTIONS}.')
        return _irsa_path(relpath, download=download, dlfunc=dlfunc), col

    raise RuntimeError(f'Planck release={release} not supported (expected 3 or 4)')


####################################################################################################
# Helpers for read_beam().


# PR4 NPIPE beam tarball + inner directory (within the tarball / cache).
_NPIPE_BEAM_ROOT = 'planck2020/misc'
_NPIPE_BEAM_TAR  = f'{_NPIPE_BEAM_ROOT}/PLANCK_RIMO_TF_R4.00.tar.gz'
_NPIPE_BEAM_DIR  = (f'{_NPIPE_BEAM_ROOT}/simulated_maps/npipe_aux/'
                    f'beam_window_functions/full_frequency')

# PR4 channels with polarized beams (Bl_TEB is provided for these, auto + cross).
# Note: NPIPE provides TEB for LFI too, unlike PR3 which had TEB only for HFI PSBs.
_NPIPE_TEB_FREQS = _LFI_FREQS + (100, 143, 217, 353)

# Not every HFI cross is provided in PR3. Explicitly whitelist (f_lo, f_hi) pairs (after sort).
_PR3_HFI_PAIRS_PROVIDED = {
    (100, 100), (100, 143), (100, 217), (100, 353), (100, '353p'), (100, 545), (100, 857),
    (143, 143), (143, 217), (143, 353),                            (143, 545), (143, 857),
                (217, 217), (217, 353),                            (217, 545), (217, 857),
                            (353, 353),                            (353, 545), (353, 857),
                                        ('353p', '353p'),          ('353p', 545), ('353p', 857),
                                                                   (545, 545),   (545, 857),
                                                                                 (857, 857),
}


def _pr3_beam_pair_sorted(freq, freq2):
    """Sort (freq, freq2) to canonical ordering used in PR3 HFI beam filenames.

    Treats '353p' > 353. Only used for PR3; NPIPE provides both orderings as
    separate files, so no canonicalization is required there.
    """
    def _key(f):
        return (int(f[:-1]) + 0.5) if (isinstance(f, str) and f.endswith('p')) else float(f)
    return tuple(sorted([freq, freq2], key=_key))


def _beam_filename(freq, freq2, pol, release, download=False, dlfunc=None):
    """Returns (filename, column_selector) for the requested Planck beam.

    ``column_selector`` is one of:
      - a column name (str), for HFI Bl files (PR3 or PR4) at HDU index 1;
      - a tuple (extname, colname), for the PR3 LFI RIMO (selects by EXTNAME)
        and for the CMB-solution beams (selects the BEAM_TF HDU).
    """

    if pol not in ('T', 'E', 'B'):
        raise RuntimeError(f'pol={pol!r} is not supported (must be "T", "E", or "B")')

    # CMB-solution beams: HDU 2 (no EXTNAME, addressed by index) inside the same
    # FITS file read_cmb uses. PR3 product; release= is ignored here. For SEVEM
    # the BEAM_TF table is empty; read_beam falls back to an analytic 5' Gaussian.
    if isinstance(freq, str) and freq.lower() in _CMB_SOLUTIONS:
        if freq2 is not None and freq2 != freq:
            raise RuntimeError(f'freq2={freq2!r} is not supported for CMB-solution beams '
                               '(cross-method beams are not a Planck data product)')
        colname = 'INT_BEAM' if pol == 'T' else 'POL_BEAM'
        relpath = (f'release_3/all-sky-maps/maps/component-maps/cmb/'
                   f'COM_CMB_IQU-{freq.lower()}_2048_R3.00_full.fits')
        return _irsa_path(relpath, download=download, dlfunc=dlfunc), (2, colname)

    if release == 4:
        return _pr4_beam_filename(freq, freq2, pol, download=download, dlfunc=dlfunc)
    if release == 3:
        return _pr3_beam_filename(freq, freq2, pol, download=download, dlfunc=dlfunc)

    raise RuntimeError(f'Planck release={release} not supported (expected 3 or 4)')


def _pr4_beam_filename(freq, freq2, pol, download=False, dlfunc=None):
    """PR4 / NPIPE beam dispatch. Returns (filename, colname) for HDU 1."""

    valid_freqs = _LFI_FREQS + _HFI_MAP_FREQS
    if (freq not in valid_freqs) or (freq2 not in valid_freqs):
        raise RuntimeError(f'PR4 beam: (freq={freq!r}, freq2={freq2!r}) not in '
                           f'{valid_freqs}')

    if pol == 'T':
        sub, colname = 'Bl', 'TEMPERATURE'
    else:  # 'E' or 'B'
        if (freq not in _NPIPE_TEB_FREQS) or (freq2 not in _NPIPE_TEB_FREQS):
            raise RuntimeError(f'PR4 TEB beams only available for channels in '
                               f'{_NPIPE_TEB_FREQS} (got freq={freq!r}, freq2={freq2!r})')
        sub, colname = 'Bl_TEB', pol

    # NPIPE provides both orderings (_030GHzx143GHz and _143GHzx030GHz are both
    # present); no canonicalization needed.
    relpath = (f'{_NPIPE_BEAM_DIR}/'
               f'{sub}_npipe6v20_{freq:03d}GHzx{freq2:03d}GHz.fits')
    filename = _nersc_path(relpath, aux_relpath=_NPIPE_BEAM_TAR,
                           download=download, dlfunc=dlfunc)
    return filename, colname


def _pr3_beam_filename(freq, freq2, pol, download=False, dlfunc=None):
    """PR3 beam dispatch. Returns (filename, col) where col is either a string
    (HFI) or a (extname, colname) tuple (LFI RIMO)."""

    is_hfi = (freq in _PR3_HFI_BEAM_FREQS) and (freq2 in _PR3_HFI_BEAM_FREQS)
    is_lfi = (freq in _LFI_FREQS) and (freq2 in _LFI_FREQS)

    if is_hfi:
        f_lo, f_hi = _pr3_beam_pair_sorted(freq, freq2)
        if (f_lo, f_hi) not in _PR3_HFI_PAIRS_PROVIDED:
            raise RuntimeError(f'HFI cross beam ({f_lo} x {f_hi}) is not provided in PR3')

        if pol == 'T':
            sub, col = 'T', 'TEMPERATURE'
        else:  # 'E' or 'B'
            if (freq not in _PR3_HFI_PSB_FREQS) or (freq2 not in _PR3_HFI_PSB_FREQS):
                raise RuntimeError(f'pol={pol!r} only available for HFI PSB channels (100,143,217,353,353p)')
            sub, col = 'TEB', pol

        relpath = (f'release_3/ancillary-data/BeamWf_HFI_R3.01/'
                   f'Bl_{sub}_R3.01_fullsky_{f_lo}x{f_hi}.fits')
        aux_relpath = 'release_3/ancillary-data/HFI_RIMO_BEAMS_R3.01.tar.gz'
        filename = _irsa_path(relpath, aux_relpath=aux_relpath,
                              download=download, dlfunc=dlfunc)
        return filename, col

    if is_lfi:
        if pol != 'T':
            raise RuntimeError(f'PR3 LFI RIMO only provides temperature beams (got pol={pol!r})')
        if freq != freq2:
            raise RuntimeError('PR3 LFI cross-channel beams are not provided')
        relpath = 'release_3/ancillary-data/LFI_RIMO_R3.31.fits'
        filename = _irsa_path(relpath, download=download, dlfunc=dlfunc)
        extname = f'BEAMWF_{freq:03d}X{freq:03d}'
        return filename, (extname, 'BL')

    raise RuntimeError(f'No PR3 beam for (freq={freq!r}, freq2={freq2!r}) '
                       '(mixed LFI x HFI crosses are not provided in PR3)')
