"""
The ``kszx.act`` module contains functions for downloading/parsing ACT data products.

References: 
  - https://lambda.gsfc.nasa.gov/product/act/index.html
  - https://lambda.gsfc.nasa.gov/product/act/actadv_prod_table.html (DR5)
  - https://lambda.gsfc.nasa.gov/product/act/act_dr6.02 (DR6)
  - https://portal.nersc.gov/project/act (cluster masks from https://arxiv.org/abs/2307.01258)
"""

import os
import zipfile
import numpy as np
import pixell.enmap
import astropy.wcs

from . import io_utils


# (default_shape, default_wcs): this (shape, wcs) is used for most ACT maps.
# It's convenient to have it available as a global variable, rather than needing to read ACT maps to get it.

default_shape = (10320, 43200)
default_wcs = astropy.wcs.WCS(naxis=2)
default_wcs.wcs.ctype = ['RA---CAR', 'DEC--CAR']
default_wcs.wcs.cdelt = [-0.008333, 0.008333]
default_wcs.wcs.crval = [0.0, 0.0]
default_wcs.wcs.crpix = [21601.0, 7560.5]


def read_cmb(freq, dr, *, night=False, download=False):
    r"""Returns a pixell map. We currently only support act+planck srcfree maps.

    Function args:
    
      - ``freq``: either 90, 150, or 220 (integer), or 'nilc' (string).
      - ``dr`` (integer): currently, ``dr=5`` and ``dr=6`` are supported.
      - ``night`` (boolean): either True (for night) or False (for daynight).
      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.

    (Note that for ACT DR6, there are two types of ACT+Planck coadds: DR4+DR6+Planck and
    DR6+Planck. According to Mat, "I would probably go with the latter since it's more homogenous,
    and the dr4 data isn't going to make a huge difference in sensitivity", so I'm using DR6+Planck.)
    """
    
    filename = _cmb_filename(freq, dr, night=night, download=download, dlfunc='kszx.act.read_cmb')
    return _read_map(filename)


def read_ivar(freq, dr, *, night=False, download=False):
    r"""Returns a pixell map. We currently only support act+planck srcfree maps.

    Function args:
    
      - ``freq`` (integer): either 90, 150, or 220.
      - ``dr`` (integer): currently, only ``dr=5`` and ``dr=6`` are supported.
      - ``night`` (boolean): either True (for night) or False (for daynight).
      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.
    """
    
    filename = _ivar_filename(freq, dr, night=night, download=download, dlfunc='kszx.act.read_ivar')
    return _read_map(filename)


def read_beam(freq, dr, lmax=None, *, night=False, download=False):
    r"""Returns a 1-d numpy array of length (lmax+1). We currently only support act+planck srcfree maps.

    Function args:
    
      - ``freq`` (integer): either 90, 150, or 220.
      - ``dr`` (integer): currently, only ``dr=5`` and ``dr=6`` is supported.
      - ``lmax`` (integer): if None, then a large lmax will be used.
      - ``night`` (boolean): either True (for night) or False (for daynight).
      - ``download`` (boolean): if True, then all needed data files will be auto-downloaded.
    """

    filename = _beam_filename(freq, dr, night=night, download=download, dlfunc='kszx.act.read_beam')
    
    print(f'Reading {filename}\n', end='')
    a = np.loadtxt(filename)

    if dr == 5:
        assert (a.ndim == 2) and (a.shape[1] == 2)
        assert np.all(a[:, 0] == np.arange(len(a)))
        a = a[:, 1]
    elif dr == 6:
        a = a[:, 1] / a[0, 1]

    if lmax is not None:
        assert lmax < len(a)
        a = a[:(lmax+1)]
        
    return a


def read_cluster_mask(download=False):
    r"""Returns the cluster mask from https://arxiv.org/abs/2307.01258 as a pixell map."""
    
    filename = _cluster_mask_filename(download, dlfunc='kszx.act.read_cluster_mask')
    print(f'Reading {filename}\n', end='')
    return pixell.enmap.read_map(filename)


def read_nilc_wide_mask(download=False):
    r"""Returns the Galactic mask from https://arxiv.org/abs/2307.01258 as a pixell map."""
    
    filename = _nilc_wide_mask_filename(download, dlfunc='kszx.act.read_nilc_wide_mask')
    print(f'Reading {filename}\n', end='')
    return pixell.enmap.read_map(filename)


def download(dr, freq_list=None, night=False, cmb=True, ivar=True, beams=True):
    r"""Downloads ACT data products (cmb, ivar, beam) for a given survey.
        
    Can be called from command line: ``python -m kszx download_act``."""

    if freq_list is None:
        freq_list = [ 90, 150, 220 ]

    for freq in freq_list:
        _beam_filename(freq, dr, night=night, download=beams)
        _cmb_filename(freq, dr, night=night, download=cmb)
        _ivar_filename(freq, dr, night=night, download=ivar)


####################################################################################################
    
                     
def _act_path(relpath, dr, download=False, aux_relpath=None, dlfunc=None):
    """Intended to be called through wrapper such as _cmb_filename(), _ivar_filename(), etc.

    Examples (note that directory structure is a little different for DR5/DR6):

      _act_path(relpath = 'act_planck_dr5.01_s08s18_AA_f090_daynight_ivar.fits', dr=5)
      _act_path(relpath = 'maps/published/act-planck_dr6.02_coadd_AA_daynight_f090_map_srcfree.fits', dr=6)

    The 'aux_relpath' arg is used in a situation where the target file is contained in a tar.gz
    or zip file. Examples:

       aux_relpath = 'act_dr5.01_auxilliary.zip'           # DR5 example
       aux_relpath = 'beams/act_dr6.02_main_beams.tar.gz'  # DR6 example
    
    Here and in other parts of kszx, the 'dlfunc' argument gives the name of a transitive caller that
    expects the file to be present, and has a 'download=False' optional argument. This information is
    only used when generating exception-text (to tell the user how to download the file).
    """

    if dr == 5:
        dr_dir = 'dr5.01'
        url_base = 'https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/'
    elif dr == 6:
        dr_dir = 'dr6.02'
        url_base = 'https://lambda.gsfc.nasa.gov/data/act/'
    else:
        raise RuntimeError(f'ACT {dr=} is not supported (currently support dr=5 and dr=6)')

    act_base_dir = os.path.join(io_utils.get_data_dir(), 'act', dr_dir)
    abspath = os.path.join(act_base_dir, relpath)

    if not io_utils.do_download(abspath, download, dlfunc):
        return abspath

    if aux_relpath is None:
        # Download case 1: target file is not contained in a tar.gz or zip.
        io_utils.wget(abspath, url_base + relpath)
    else:
        # Download case 2: need to download the auxfile (tar.gz or zip) first.
        # We do this by calling _act_path() recursively (with aux_relpath=None).
        aux_abspath = _act_path(aux_relpath, dr, download=download, dlfunc=dlfunc)
        io_utils.unpack(aux_abspath, expected_dstfile=abspath)  # works for .zip and .tar.
    
    return abspath


def _act_nersc_portal_path(relpath, download=False, dlfunc=None):
    """Intended to be called through wrapper such as _cluster_mask_filename().
    For files which are downloaded here: https://portal.nersc.gov/project/act
    """

    act_base_dir = os.path.join(io_utils.get_data_dir(), 'act')
    abspath = os.path.join(act_base_dir, relpath)

    if io_utils.do_download(abspath, download, dlfunc):
        url = f'https://portal.nersc.gov/project/act/{relpath}'
        io_utils.wget(abspath, url)   # calls assert os.path.exists(...) after downloading

    return abspath
    

def _cmb_filename(freq, dr, night=False, download=False, dlfunc=None):
    time = 'night' if night else 'daynight'

    if freq == 'nilc':
        if dr != 6:
            raise RuntimeError(f'ACT NILC maps are only available for dr=6')
        relpath = 'nilc/published/act-planck_dr6.02_nilc_blackbody_T.fits'
    elif dr == 5:
        relpath = f'act_planck_dr5.01_s08s18_AA_f{freq:03d}_{time}_map_srcfree.fits'
    elif dr == 6:
        relpath = f'maps/published/act-planck_dr6.02_coadd_AA_{time}_f{freq:03d}_map_srcfree.fits'
    else:
        raise RuntimeError(f'ACT {dr=} is not supported (currently we support dr=5 or dr=6)')

    return _act_path(relpath, dr, download=download, dlfunc=dlfunc)


def _ivar_filename(freq, dr, night=False, download=False, dlfunc=None):
    time = 'night' if night else 'daynight'

    if dr == 5:
        relpath = f'act_planck_dr5.01_s08s18_AA_f{freq:03d}_{time}_ivar.fits'
    elif dr == 6:
        relpath = f'maps/published/act-planck_dr6.02_coadd_AA_{time}_f{freq:03d}_ivar.fits'
    else:
        raise RuntimeError(f'ACT {dr=} is not supported (currently we support dr=5 or dr=6)')
    
    return _act_path(relpath, dr, download=download, dlfunc=dlfunc)


def _beam_filename(freq, dr, night=False, download=False, dlfunc=None):
    time = 'night' if night else 'daynight'
    if dr == 5:
        relpath = f'beams/act_planck_dr5.01_s08s18_f{freq:03d}_{time}_beam.txt'
        aux_relpath = 'act_dr5.01_auxilliary.zip'
    elif dr == 6:
        # According to https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_maps_info.html:
        #   Coadd maps are convolved with the coadd_{array}_{freq}_night_beam_tform_jitter_cmb.txt beams,
        #   where "array" is "pa5" for the f090 and f150 coadds and "pa4" for the f220 coadd.
        if not night: print('Note: daynight beam is not available for ACT DR6, using night beam instead.')
        adict = {90: 'pa5', 150: 'pa5', 220: 'pa4'}
        relpath = f'beams/main_beams/nominal/coadd_{adict[freq]}_f{freq:03d}_night_beam_tform_jitter_cmb.txt'
        aux_relpath = 'beams/act_dr6.02_main_beams.tar.gz'
    else:
        raise RuntimeError(f'ACT {dr=} is not supported (currently we support dr=5 or dr=6)')

    return _act_path(relpath, dr, download, aux_relpath=aux_relpath, dlfunc=dlfunc)


def _cluster_mask_filename(download=False, dlfunc=None):
    return _act_nersc_portal_path('dr6_nilc/ymaps_20230220/masks/cluster_mask.fits', download=download, dlfunc=dlfunc)


def _nilc_wide_mask_filename(download=False, dlfunc=None):
    return _act_nersc_portal_path('dr6_nilc/ymaps_20230220/masks/wide_mask_GAL070_apod_1.50_deg_wExtended.fits', download, dlfunc=dlfunc)


def _read_map(filename):
    """Called by read_cmb() and read_ivar()."""

    print(f'Reading {filename}\n', end='')

    # FIXME is there a way to avoid reading TQU, if we only want T?
    m = pixell.enmap.read_map(filename)

    if m.ndim == 2:
        return m

    assert m.ndim == 3

    # Use of .copy() saves memory, by keeping T data while dropping reference to TQU data.
    return m[0].copy()
