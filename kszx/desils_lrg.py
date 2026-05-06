"""
The ``kszx.desils_lrg`` module is based on data products from
Zhou et al 2023, "DESI LRG samples for cross-correlation",
https://arxiv.org/abs/2309.06443.

Data files are mostly found here:
  https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/.

With the exception of the randoms, which are here:
  https://data.desi.lbl.gov/public/ets/target/catalogs/dr9/0.49.0/randoms/
"""

import os
import yaml
import fitsio
import healpy
import warnings
import functools
import numpy as np

from . import Catalog
from . import io_utils
    

def read_galaxies(extended, download=False):
    r"""Reads LRGs with no imaging weights or quality cuts, and returns a Catalog object.

    After calling this function, you'll probably want to call :func:`~kszx.desils_lrg.compute_imaging_weights`
    and/or :func:`~kszx.desils_lrg.apply_quality_cuts`. You'll also want to call :func:`Catalog.apply_redshift_cut()`
    to retrict to an appropriate redshift range.
    
    Function arguments:

      - ``extended`` (boolean): determines whether "extended" or "main" catalog is read.
      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    Returns a :class:`kszx.Catalog` object, with the following columns::
    
      ra_deg, dec_deg, z, zerr,  # sky location, redshift, redshift error
      pz_bin,                    # in {1,2,3,4}
      nobs_{g,r,z}, ebv, lrg_mask, maskbits      # for quality cuts
      photsys, galdepth_{g,r,z}, psfsize_{g,r,z} # for imaging weights
    """

    # Note: LRG catalog is split between a few FITS files (see _catalog_filename docstring).

    # First, the "main" FITS file.
    # cols = list of pairs (my_col_name, fits_col_name).
    cols = [ ('ra_deg','RA'), ('dec_deg','DEC'), ('z','Z_PHOT_MEDIAN') ]
    cols += [ (f'nobs_{x}', f'PIXEL_NOBS_{x.upper()}') for x in ['g','r','z'] ]
    cols += [ ('ebv','EBV'), ('lrg_mask','lrg_mask'), ('pz_bin','pz_bin') ]
    cols += [ ('maskbits','MASKBITS'), ('photsys','PHOTSYS') ]
    filename = _catalog_filename(extended, download=download, dlfunc='kszx.desils_lrg.read_galaxies')
    catalog = Catalog.from_fits(filename, cols)

    # Read zerr column (photo-z error) from the "pz" FITS file.
    filename = _catalog_filename(extended, suffix='pz', download=download, dlfunc='kszx.desils_lrg.read_galaxies')
    ctmp = Catalog.from_fits(filename, [('zerr','Z_PHOT_STD')])
    catalog.absorb_columns(ctmp, destructive=True)

    # Read GALDEPTH_{G,R,Z} and PSFSIZE_{G,R,Z} cols from the "more_2" FITS file.
    cols = [ (f'galdepth_{x}', f'GALDEPTH_{x.upper()}') for x in ['g','r','z'] ]
    cols += [ (f'psfsize_{x}', f'PSFSIZE_{x.upper()}') for x in ['g','r','z'] ]
    filename = _catalog_filename(extended, suffix='more_2', download=download, dlfunc='kszx.desils_lrg.read_galaxies')
    ctmp = Catalog.from_fits(filename, cols)
    catalog.absorb_columns(ctmp, destructive=True)

    return catalog


def read_randoms(ix_list, download=False):
    r"""Reads LRG random catalog with no quality cuts, and returns a Catalog object.

    After calling this function, you'll probably want to call :func:`~kszx.desils_lrg.apply_quality_cuts`.
    Note that DESILS-LRG randoms do not have redshifts!
    
    Function arguments:

      - ``ix_list`` (list): a list of integers $0 \le i < 200$. The DESILS-LRG randoms
        are split across 200 source files, and this argument determines which source files
        will be read.
    
      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    Returns a :class:`kszx.Catalog` object, obtained by combining randoms from all
    source files in ``ix_list``, with the following columns::

      ra_deg, dec_deg,           # sky location but no redshift
      nobs_{g,r,z}, ebv, lrg_mask, maskbits  # for quality cuts
    """

    assert len(ix_list) >= 1
    assert len(set(ix_list)) == len(ix_list)  # no duplicates
    
    catalog_list = [ ]
    
    for ix in ix_list:
        assert 0 <= ix < 200   # generalize later

        # List of pairs (my_col_name, fits_col_name).
        cols = [ ('ra_deg','RA'), ('dec_deg','DEC'), ('ebv','EBV') ]
        cols += [ (f'nobs_{x}', f'NOBS_{x.upper()}') for x in ['g','r','z'] ]
        cols += [ ('maskbits','MASKBITS') ]
    
        filename = _random_filename(ix, mflag=False, download=download, dlfunc='kszx.desils_lrg.read_galaxies')
        catalog = Catalog.from_fits(filename, cols)

        # lrg_mask
        filename2 = _random_filename(ix, mflag=True, download=download, dlfunc='kszx.desils_lrg.read_galaxies')
        ctmp = Catalog.from_fits(filename2, [('lrg_mask','lrg_mask')])
        catalog.add_column('lrg_mask', ctmp.lrg_mask)

        catalog_list.append(catalog)

    return Catalog.concatenate(catalog_list, name='DESILS-LRG randoms', destructive=True)


def compute_imaging_weights(gcat, extended, ebv=True, download=False, col_name='weight'):
    r"""Adds an imaging weights column to the Catalog (usually called just after read_galaxies()).

    Note that this function can be called on the output of :func:`~kszx.desils_lrg.read_galaxies`,
    but not the output of :func:`~kszx.desils_lrg.read_randoms`.

    Function arguments:

      - ``gcat``: a :class:`~kszx.Catalog` object, obtained by calling
        :func:`kszx.desils_lrg.read_galaxies`.

      - ``extended`` (boolean): Slightly different imaging weights should be used for the
        "main" and "extended" DESILS-LRG samples, and this argument selects between them.
        (Note that ``extended`` is also an argument to :func:`~kszx.desils_lrg.read_galaxies`.)

      - ``ebv`` (boolean): The DESILS-LRG data products define two sets of imaging weights,
        which do or do not use the E(B-V) column. This argument selects between them.

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

      - ``col_name`` (string): name of the column to add to the Catalog. Defaults to ``'weight'``.
        Useful if you want to keep the imaging weights in a separate column from other weights
        (e.g. ``col_name='imaging_weight'``).

    Reference:

      https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/compute_imaging_weights.py

    Note: in addition to the imaging weights column, this function also adds ``galdepth_gmag_ebv``,
    ``galdepth_rmag_ebv``, and ``galdepth_zmag_ebv`` columns to the Catalog (following the python
    script referenced above.)
    """

    # Weights array that will be added to Catalog at the end.
    weight = np.zeros(gcat.size)

    # Load weights (linear_coeffs yaml file)
    weights_path = _linear_coeffs_filename(extended, ebv, download, dlfunc='kszx.desils_lrg.read_galaxies')
    with open(weights_path, "r") as f:
        linear_coeffs = yaml.safe_load(f)

    # Convert depths to units of magnitude
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gcat.add_column('galdepth_gmag_ebv', -2.5*(np.log10((5/np.sqrt(gcat.galdepth_g)))-9) - 3.214*gcat.ebv)
        gcat.add_column('galdepth_rmag_ebv', -2.5*(np.log10((5/np.sqrt(gcat.galdepth_r)))-9) - 2.165*gcat.ebv)
        gcat.add_column('galdepth_zmag_ebv', -2.5*(np.log10((5/np.sqrt(gcat.galdepth_z)))-9) - 1.211*gcat.ebv)

    # List of column names in linear regression ('EBV', 'PSFSIZE_G', 'galdepth_gmag_ebv', etc.)
    xnames_fit = list(linear_coeffs['south_bin_1'].keys())
    xnames_fit.remove('intercept')

    # Assign zero weights to objects with invalid imaging properties
    # (their fraction should be negligibly small)
    mask_bad = np.full(gcat.size, False)
    for xname in xnames_fit:
        col = getattr(gcat, xname.lower())
        mask_bad |= ~np.isfinite(col)
    print(f'kszx.desils_lrg.compute_imaging_mask: {np.sum(mask_bad)} invalid objects')
    
    for field in ['north', 'south']:
        photsys = field[0].upper()   # 'N' or 'S'

        for bin_index in range(1, 5):  # 4 bins
            mask_bin = (gcat.photsys == photsys)
            mask_bin &= (gcat.pz_bin == bin_index)
            mask_bin &= ~mask_bad

            # wt = intercept + sum(coeff * col)
            bin_str = f'{field}_bin_{bin_index}'
            wt = linear_coeffs[bin_str]['intercept']
            
            for xname in xnames_fit:
                coeff = linear_coeffs[bin_str][xname]
                col = getattr(gcat, xname.lower())
                wt += coeff * col[mask_bin]

            weight[mask_bin] = 1.0 / wt    # 1/predicted_density as weights for objects

    gcat.add_column(col_name, weight)


def apply_quality_cuts(catalog, min_nobs=2, max_ebv=0.15, max_stardens=2500, lrg_mask=True, maskbits=True, island_mask=True, mask_negative_zerr=True, download=True):
    r"""Applies LRG quality cuts in-place to a catalog (usually called just after read_galaxies() or read_randoms()).

    Function arguments:

      - ``catalog``: a :class:`~kszx.Catalog` object, obtained by calling
        :func:`kszx.desils_lrg.read_galaxies` or :func:`kszx.desils_lrg.read_randoms`.

      - ``min_nobs`` (integer): min allowed value for NOBS_G, NOBS_R, NOBS_Z.
        Default value (2) comes from the references below.

      - ``max_ebv`` (float): max allowed value for E(B-V).
        Default value (0.15) comes from the references below.

      - ``max_stardens`` (float): max allowed stellar density (in deg^{-2})).
        Note that stellar density isn't in the catalogs, and will be inferred from an external
        data product. Default value (2500) comes from the references below.
    
      - ``lrg_mask`` (boolean): whether to apply 'lrg_mask' column from catalog.
        (See third reference below.)

      - ``maskbits`` (boolean): whether to apply DESILS mask bits 1 (BRIGHT), 12 (GALAXY),
        and 13 (CLUSTER). This is strictly weaker than ``lrg_mask``. (See fourth reference
        below.)
    
      - ``island_mask`` (boolean): If True, then some islands in the NGC will be masked.

      - ``mask_negative_zerr`` (boolean): If True, galaxies with photo-z error < 0
        will be masked. (This flag was added by Kendrick, and is not in Rongpu's original
        script. Note that there are ~3000 such galaxies in the extended catalog, and only
        2 of them pass the other qualtiy cuts.)

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    Reference:
    
       - https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/quality_cuts.py
       - https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/randoms_quality_cuts.py
       - https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/lrgmask_v1.1/README.txt
       - https://www.legacysurvey.org/dr10/bitmasks/
    """

    if island_mask:
        mask = ~((catalog.dec_deg < -10.5) & (catalog.ra_deg > 120) & (catalog.ra_deg < 260))
        catalog.apply_boolean_mask(mask, name = 'DESILS-LRG quality cut: remove NGC islands')

    if (min_nobs is not None) and (min_nobs > 0):
        mask = (catalog.nobs_g >= min_nobs) & (catalog.nobs_r >= min_nobs) & (catalog.nobs_z >= min_nobs)
        catalog.apply_boolean_mask(mask, name = f'DESILS-LRG quality cut: nobs >= {min_nobs}')

    if lrg_mask:
        bits = (1<<1) | (1<<12) | (1<<13)
        mask = np.logical_and(catalog.lrg_mask == 0, (catalog.maskbits & bits) == 0)
        catalog.apply_boolean_mask(mask, name = 'DESILS-LRG quality cut: LRG maskbits == 0')

    if max_ebv is not None:
        mask = (catalog.ebv < max_ebv)
        catalog.apply_boolean_mask(mask, name = f'DESILS-LRG quality cut: EBV < {max_ebv}')

    if max_stardens is not None:
        m = read_stardens_map(download=download)
        nside = healpy.npix2nside(len(m))
        ix = healpy.ang2pix(nside, catalog.ra_deg, catalog.dec_deg, lonlat=True)
        stardens = m[ix]
        mask = (stardens < max_stardens)
        catalog.apply_boolean_mask(mask, name = f'DESILS-LRG quality cut: stardens < {max_stardens}')

    if mask_negative_zerr and ('zerr' in catalog.col_names):
        mask = (catalog.zerr >= 0)
        catalog.apply_boolean_mask(mask, name = f'DESILS-LRG quality cut: zerr >= 0')


def download(*, main=False, extended=False, randoms=0,
             imaging_weights=False, stardens=False):
    r"""Pre-fetch DESILS-LRG data products (Zhou et al 2023).

    All flags default to False / 0; pass keywords to select what to download.

      - ``main`` (bool): main galaxy catalog -- the three FITS files actually
        read by :func:`~kszx.desils_lrg.read_galaxies` (suffixes ``None``,
        ``'more_2'``, ``'pz'``).
      - ``extended`` (bool): same three FITS files for the extended sample.
      - ``randoms`` (int): download the first ``randoms`` random source files
        (each provides one ETS FITS and one lrgmask FITS, so 2x files).
      - ``imaging_weights`` (bool): the four linear-coeffs YAML files
        (main/extended x ebv/no_ebv) consumed by
        :func:`~kszx.desils_lrg.compute_imaging_weights`.
      - ``stardens`` (bool): nside=64 stellar density map used by
        :func:`~kszx.desils_lrg.apply_quality_cuts`.

    Already-present files are skipped (re-running is safe).

    Can be called from command line: ``python -m kszx download_desils_lrg ...``.
    """

    dlfunc = 'kszx.desils_lrg.download'

    for ext_flag, do_it in [(False, main), (True, extended)]:
        if not do_it:
            continue
        for suffix in [None, 'more_2', 'pz']:
            _catalog_filename(extended=ext_flag, suffix=suffix,
                              download=True, dlfunc=dlfunc)

    assert randoms >= 0
    if randoms > 200:
        raise RuntimeError(f'kszx.desils_lrg.download: randoms={randoms} exceeds the 200 available source files')
    for ix in range(randoms):
        _random_filename(ix, mflag=False, download=True, dlfunc=dlfunc)
        _random_filename(ix, mflag=True,  download=True, dlfunc=dlfunc)

    if imaging_weights:
        for ext_flag in [False, True]:
            for ebv_flag in [False, True]:
                _linear_coeffs_filename(extended=ext_flag, ebv=ebv_flag,
                                        download=True, dlfunc=dlfunc)

    if stardens:
        _desils_lrg_path('misc/pixweight-dr7.1-0.22.0_stardens_64_ring.fits',
                         download=True, dlfunc=dlfunc)


@functools.cache
def read_stardens_map(download=False):
    """Returns nside=64 ring-ordered healpix map, cached between calls to read_stardens_map().

    Intended as a helper for apply_quality_cuts(), but may be independently useful."""
    
    filename = _desils_lrg_path('misc/pixweight-dr7.1-0.22.0_stardens_64_ring.fits', download=True, dlfunc='kszx.desils_lrg.read_galaxies')
    
    print(f'Reading {filename}\n', end='')
    f = fitsio.read(filename)

    nside = 64
    npix = 12 * nside**2
    
    assert np.all(f['HPXPIXEL'] == np.arange(npix))
    return np.asarray(f['STARDENS'], dtype=np.float32)  # convert big-endian float32 -> native


####################################################################################################


def _catalog_filename(extended, suffix=None, download=False, dlfunc=None):
    """The 'extended' argument should be True or False.
    
    Catalogs are split across multiple FITS files, distinguished by 'suffix':

        suffix=None      TARGETID, RA, DEC, EBV, PIXEL_NOBS_{G,R,Z}, MASKBITS, 
                         PHOTSYS, Z_PHOT_MEDIAN, lrg_mask, pz_bin

        suffix='more_1'  GAIA_PHOT_{BP,RP}_MEAN_MAG, GAIA_ASTROMETRIC_EXCESS_NOISE,
                         FITBITS, FRACFLUX_{G,R,Z,W1,W2}, FRACMASKED_{G,R,Z},
                         FRACIN_{G,R,Z}, FIBERTOTFLUX_G, SHAPE_{R,R_IVAR},
                         SHAPE_{E1,E2}, SERSIC, DCHISSQ

        suffix='more_2'  GALDEPTH_{G,R,Z}, PSFSIZE_{G,R,Z,W1,W2}, PSFSIZE_{G,R,Z}

        suffix='photom'  TYPE, EBC, FLUX_{G,R,Z,W1,W2}, FLUX_IVAR_{G,R,Z,W1,W2},
                         MW_TRANSMISSION_{G,R,Z,W1,W2}, FIBERFLUX_{G,R,Z},
                         FIBERTOTFLUX_{R,Z}, GAIA_PHOT_G_MEAN_MAG

        suffix='pz'      Z_PHOT_{MEAN,MEDIAN,STD,L68,U68,L95,U95}, 
                         Z_SPEC, SURVEY, TRAINING

        suffix='pzbins-weights'          weight
        suffix='pzbins-weights_no_ebv'   weight

    For descriptions of some of these columns, see:

        https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/README.txt        
        https://www.legacysurvey.org/dr9/files/#sweep-catalogs-region-sweep
    
    Here and in other parts of kszx, the 'dlfunc' argument gives the name of a transitive caller that
    expects the file to be present, and has a 'download=False' optional argument. This information is
    only used when generating exception-text (to tell the user how to download the file).
    """

    lrg = 'extended_lrg' if extended else 'lrg'
    slist = [ None, 'more_1', 'more_2', 'photom', 'pz', 'pzbins-weights', 'pzbins-weights_no_ebv' ]
    
    if suffix not in slist:
        raise RuntimeError(f"kszx.desils_lrg._catalog_filename: suffix='{suffix}' invalid."
                           + f" Valid suffixes are: {slist}")

    if suffix is None:
        return _desils_lrg_path(f'catalogs/dr9_{lrg}_pzbins.fits', download=download, dlfunc=dlfunc)
    else:
        return _desils_lrg_path(f'catalogs/more/dr9_{lrg}_{suffix}.fits', download=download, dlfunc=dlfunc)


def _random_filename(ix, mflag, download=False, dlfunc=None):
    """The index 'ix' satisfies 0 <= ix < 200.

    The randoms are split across two FITS files, distinguished through the boolean 'mflag' arg:
      - mflag=False   ETS file containing RA, DEC, NOBS_{G,R,Z}, EBV  (+ more cols I don't use)
      - mflag=True    LRG mask file containing boolean 'lrg_mask' col
    """

    assert 0 <= ix < 200
    s = f'{(ix//20)+1}-{(ix%20)}'   # E.g. (ix=31) -> '2-11'

    if mflag:
        return _desils_lrg_path(f'catalogs/lrgmask_v1.1/randoms-{s}-lrgmask_v1.1.fits.gz', download=download, dlfunc=dlfunc)
    else:
        return _desils_ets_path(f'target/catalogs/dr9/0.49.0/randoms/resolve/randoms-{s}.fits', download=download, dlfunc=dlfunc)


def _linear_coeffs_filename(extended, ebv, download=False, dlfunc=None):
    """The 'extended' and 'ebv' arguments should be True or False."""

    prefix = 'extended' if extended else 'main'
    suffix = '' if ebv else '_no_ebv'
    return _desils_lrg_path(f'catalogs/imaging_weights/{prefix}_lrg_linear_coeffs_pz{suffix}.yaml', download=download, dlfunc=dlfunc)

    
def _desils_lrg_path(relpath, download=False, dlfunc=None):
    """Example: _desils_lrs_path('catalogs/dr9_lrg_pzbins.fits').
        -> local filename /data/desils/lrg_xcorr_2023/v1/catalogs/dr9_lrg_pzbins.fits
        -> url https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/dr9_lrg_pzbins.fits

    Intended to be called through wrapper, e.g. _catalog_filename()."""
    
    relpath = os.path.join('lrg_xcorr_2023/v1', relpath)
    desils_base_dir = os.path.join(io_utils.get_data_dir(), 'desils')
    abspath = os.path.join(desils_base_dir, relpath)

    if io_utils.do_download(abspath, download, dlfunc):
        url = f'https://data.desi.lbl.gov/public/papers/c3/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(...) after downloading
    
    return abspath


def _desils_ets_path(relpath, download=False, dlfunc=None):
    """Example: _desils_ets_path('target/catalogs/dr9/0.49.0/randoms/resolve/randoms-1-0.fits')
        -> local filename /data/desils/ets/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-1-0.fits
        -> url https://data.desi.lbl.gov/public/ets/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-1-0.fits

    Intended to be called through wrapper, e.g. _random_filename()."""

    relpath = os.path.join('ets', relpath)
    desils_base_dir = os.path.join(io_utils.get_data_dir(), 'desils')
    abspath = os.path.join(desils_base_dir, relpath)

    if io_utils.do_download(abspath, download, dlfunc):
        url = f'https://data.desi.lbl.gov/public/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(...) after downloading
    
    return abspath
