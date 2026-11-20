"""
The ``kszx.desils_lrg`` module is based on data products from
Zhou et al 2023, "DESI LRG samples for cross-correlation",
https://arxiv.org/abs/2309.06443.

Data files are mostly found here:
  https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/.

With the exception of the randoms, which are here:
  https://data.desi.lbl.gov/public/ets/target/catalogs/dr9/0.49.0/randoms/

DR11 (``dr=11`` argument): the Legacy Survey DR11 LRG catalogs (Rongpu Zhou, target
selection 5.1.0) and the corresponding randoms + lrg_mask are not public yet, so they
can't be auto-downloaded. At NERSC, create these symlinks (one-time setup)::

   mkdir -p $KSZX_DATA_DIR/desils
   ln -s /global/cfs/cdirs/desi/users/rongpu/targets/dr11.0/5.1.0/resolve $KSZX_DATA_DIR/desils/lrg_dr11
   ln -s /global/cfs/cdirs/desi/target/catalogs/dr11/5.1.0/randoms/resolve $KSZX_DATA_DIR/desils/ets_dr11_randoms
   ln -s /global/cfs/cdirs/desi/users/rongpu/desi_mask/randoms_dr11/lrgmask_v1.1 $KSZX_DATA_DIR/desils/lrgmask_dr11_randoms

(If $KSZX_DATA_DIR is not set, replace it by $HOME/kszx_data.)
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
    

def read_galaxies(extended=False, download=False, dr=9):
    r"""Reads LRGs with no imaging weights or quality cuts, and returns a Catalog object.

    After calling this function, you'll probably want to call :func:`~kszx.desils_lrg.compute_imaging_weights`
    and/or :func:`~kszx.desils_lrg.apply_quality_cuts`. You'll also want to call :func:`Catalog.apply_redshift_cut()`
    to retrict to an appropriate redshift range.

    Function arguments:

      - ``extended`` (boolean): determines whether "extended" or "main" catalog is read.
        (Only the "main" catalog is available for DR11.)
      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.
        (Not supported for DR11, see module docstring.)
      - ``dr`` (integer): Legacy Survey data release, either 9 (default) or 11.

    Returns a :class:`kszx.Catalog` object, with the following columns::

      ra_deg, dec_deg, z, zerr,  # sky location, redshift, redshift error
      pz_bin,                    # in {1,2,3,4}
      nobs_{g,r,z}, ebv, lrg_mask, maskbits      # for quality cuts
      photsys, galdepth_{g,r,z}, psfsize_{g,r,z} # for imaging weights

    For DR11, the ``pz_bin``, ``galdepth_*`` and ``psfsize_*`` columns are not available
    (and :func:`~kszx.desils_lrg.compute_imaging_weights` is not supported).
    """

    if dr == 11:
        return _read_galaxies_dr11(extended)
    assert dr == 9, f'kszx.desils_lrg.read_galaxies: {dr=} is not supported (expected 9 or 11)'

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


def read_randoms(ix_list, download=False, dr=9):
    r"""Reads LRG random catalog with no quality cuts, and returns a Catalog object.

    After calling this function, you'll probably want to call :func:`~kszx.desils_lrg.apply_quality_cuts`.
    Note that DESILS-LRG randoms do not have redshifts!

    Function arguments:

      - ``ix_list`` (list): a list of integers $0 \le i < 200$ (DR9) or $0 \le i < 20$ (DR11).
        The DESILS-LRG randoms are split across 200 (DR9) or 20 (DR11) source files, and this
        argument determines which source files will be read.

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.
        (Not supported for DR11, see module docstring.)

      - ``dr`` (integer): Legacy Survey data release, either 9 (default) or 11.

    Returns a :class:`kszx.Catalog` object, obtained by combining randoms from all
    source files in ``ix_list``, with the following columns::

      ra_deg, dec_deg,           # sky location but no redshift
      nobs_{g,r,z}, ebv, lrg_mask, maskbits  # for quality cuts
    """

    assert dr in (9, 11), f'kszx.desils_lrg.read_randoms: {dr=} is not supported (expected 9 or 11)'
    assert len(ix_list) >= 1
    assert len(set(ix_list)) == len(ix_list)  # no duplicates

    catalog_list = [ ]

    for ix in ix_list:
        # List of pairs (my_col_name, fits_col_name).
        cols = [ ('ra_deg','RA'), ('dec_deg','DEC'), ('ebv','EBV') ]
        cols += [ (f'nobs_{x}', f'NOBS_{x.upper()}') for x in ['g','r','z'] ]
        cols += [ ('maskbits','MASKBITS') ]

        if dr == 11:
            filename = _random_filename_dr11(ix, mflag=False)
            filename2 = _random_filename_dr11(ix, mflag=True)
        else:
            filename = _random_filename(ix, mflag=False, download=download, dlfunc='kszx.desils_lrg.read_galaxies')
            filename2 = _random_filename(ix, mflag=True, download=download, dlfunc='kszx.desils_lrg.read_galaxies')

        catalog = Catalog.from_fits(filename, cols)

        # lrg_mask (row-matched to the randoms file)
        ctmp = Catalog.from_fits(filename2, [('lrg_mask','lrg_mask')])
        catalog.add_column('lrg_mask', ctmp.lrg_mask)

        catalog_list.append(catalog)

    return Catalog.concatenate(catalog_list, name='DESILS-LRG randoms', destructive=True)


def compute_imaging_weights(gcat, extended, ebv=True, download=False):
    r"""Adds a 'weights' column to the Catalog (usually called just after read_galaxies()).

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
    
    Reference:
    
      https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/compute_imaging_weights.py

    Note: in addition to the ``weights`` column, this function also adds ``galdepth_gmag_ebv``,
    ``galdepth_rmag_ebv``, and ``galdepth_zmag_ebv`` columns to the Catalog (following the python
    script referenced above.)
    """

    if 'pz_bin' not in gcat.col_names:
        raise NotImplementedError('kszx.desils_lrg.compute_imaging_weights: imaging weights are only'
                                  + ' available for DR9 (catalog has no pz_bin column, e.g. DR11)')

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

    gcat.add_column('weight', weight)


def apply_quality_cuts(catalog, min_nobs=2, max_ebv=0.15, max_stardens=2500, lrg_mask=True, maskbits=True, island_mask=None, mask_negative_zerr=True, download=True, dr=9):
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
        If None (default), then True for DR9 and False for DR11 (where these regions are
        covered by DECam data).

      - ``mask_negative_zerr`` (boolean): If True, galaxies with photo-z error < 0
        will be masked. (This flag was added by Kendrick, and is not in Rongpu's original
        script. Note that there are ~3000 such galaxies in the extended catalog, and only
        2 of them pass the other qualtiy cuts.)

      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

      - ``dr`` (integer): Legacy Survey data release, either 9 (default) or 11. Only used
        to choose the default value of ``island_mask``. (The stellar density map is full-sky
        and is used for both.)

    Reference:

       - https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/quality_cuts.py
       - https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/randoms_quality_cuts.py
       - https://data.desi.lbl.gov/public/papers/c3/lrg_xcorr_2023/v1/catalogs/lrgmask_v1.1/README.txt
       - https://www.legacysurvey.org/dr10/bitmasks/
    """

    assert dr in (9, 11), f'kszx.desils_lrg.apply_quality_cuts: {dr=} is not supported (expected 9 or 11)'

    if island_mask is None:
        island_mask = (dr == 9)

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


def _read_galaxies_dr11(extended):
    """Helper for read_galaxies(dr=11). The DR11 LRG catalog is split between row-matched FITS files:

        dr11_lrg_5.1.0_basic.fits              RA, DEC, NOBS_{G,R,Z}, MASKBITS, PHOTSYS, TARGETID, ...
        dr11_lrg_5.1.0_basic_pz.fits           TARGETID, Z_PHOT_{MEAN,MEDIAN,STD,...}, Z_SPEC, ...
        dr11_lrg_5.1.0_photom.fits             EBV, FLUX_*, MW_TRANSMISSION_*, ...
        dr11_lrg_5.1.0_lrgmask_v1.1.fits.gz    lrg_mask
    """

    if extended:
        raise RuntimeError('kszx.desils_lrg.read_galaxies: extended=True is not available for dr=11')

    cols = [ ('ra_deg','RA'), ('dec_deg','DEC'), ('targetid','TARGETID') ]
    cols += [ (f'nobs_{x}', f'NOBS_{x.upper()}') for x in ['g','r','z'] ]
    cols += [ ('maskbits','MASKBITS'), ('photsys','PHOTSYS') ]
    catalog = Catalog.from_fits(_catalog_filename_dr11('basic'), cols, name='DESILS-LRG DR11 galaxies')

    # Photo-z (check row-matching with TARGETID).
    ctmp = Catalog.from_fits(_catalog_filename_dr11('basic_pz'), [('z','Z_PHOT_MEDIAN'), ('zerr','Z_PHOT_STD'), ('targetid_pz','TARGETID')])
    assert np.array_equal(catalog.targetid, ctmp.targetid_pz), 'DR11 LRG basic and basic_pz files are not row-matched'
    ctmp.remove_column('targetid_pz')
    catalog.absorb_columns(ctmp, destructive=True)

    ctmp = Catalog.from_fits(_catalog_filename_dr11('photom'), [('ebv','EBV')])
    catalog.absorb_columns(ctmp, destructive=True)

    ctmp = Catalog.from_fits(_catalog_filename_dr11('lrgmask_v1.1', ext='.fits.gz'), [('lrg_mask','lrg_mask')])
    catalog.absorb_columns(ctmp, destructive=True)

    return catalog


def _catalog_filename_dr11(suffix, ext='.fits'):
    assert suffix in [ 'basic', 'basic_pz', 'photom', 'lrgmask_v1.1', 'stellar_mass' ]
    return _desils_dr11_path('lrg_dr11', f'dr11_lrg_5.1.0_{suffix}{ext}')


def _random_filename_dr11(ix, mflag):
    """The index 'ix' satisfies 0 <= ix < 20. Same 'mflag' convention as _random_filename()."""

    assert 0 <= ix < 20

    if mflag:
        return _desils_dr11_path('lrgmask_dr11_randoms', f'randoms-1-{ix}-lrgmask_v1.1.fits.gz')
    else:
        return _desils_dr11_path('ets_dr11_randoms', f'randoms-1-{ix}.fits')


def _desils_dr11_path(subdir, basename):
    """DR11 files can't be downloaded (not public yet), they are accessed through symlinks (see module docstring)."""

    abspath = os.path.join(io_utils.get_data_dir(), 'desils', subdir, basename)

    if not os.path.exists(abspath):
        raise RuntimeError(f'File {abspath} not found. DR11 LRG files can not be auto-downloaded:'
                           + ' see the kszx.desils_lrg module docstring for the symlinks to create at NERSC.')

    return abspath


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
