"""
The ``kszx.quijote`` module contains functions for reading Quijote simulation data products.

The Quijote simulations (Villaescusa-Navarro et al. 2020, ApJS 250, 2) are a suite of
>82,000 full N-body cosmological simulations in periodic cubic boxes of side 1 h^{-1} Gpc.

Data is auto-downloaded via Globus when ``download=True`` is passed. See
:func:`kszx.quijote.download` for bulk downloading.

References:
   - https://quijote-simulations.readthedocs.io
   - https://arxiv.org/abs/1909.05273

Units: All returned quantities use kszx physical units:
   - Positions: Mpc (not h^{-1} Mpc)
   - Masses: M_sun (not h^{-1} M_sun)
   - Velocities: dimensionless v/c (not km/s)
   - Wavenumbers: Mpc^{-1} (not h Mpc^{-1})
   - Power spectra: Mpc^3 (not (h^{-1} Mpc)^3)
"""

import os
import functools
import numpy as np

from . import io_utils
from . import globus_utils
from . import gadget_utils
from .Catalog import Catalog


####################################################################################################
# Constants


# Quijote data is split across two Globus endpoints:
#
#   EP1 (Quijote_simulations):  Pk/matter/, Linear_Pk/
#   EP2 (Quijote_simulations2): Halos/, Snapshots/ (all sim types)
#
# Remote paths on Globus (verified):
#   Halos (EP2):            /Halos/FoF/{sim_type}/{realization}/groups_{snap:03d}/
#   Pk real-space (EP1):    /Pk/matter/{sim_type}/{realization}/Pk_m_z={z}.txt
#   Pk redshift-space (EP1):/Pk/matter/{sim_type}/{realization}/Pk_m_RS{0,1,2}_z={z}.txt
#   Linear Pk (EP1):        /Linear_Pk/{sim_type}/CAMB_TABLES/CAMB_matterpow_0.dat
#     (per-realization:     /Linear_Pk/{sim_type}/{realization}/CAMB_TABLES/CAMB_matterpow_0.dat)
#   Snapshots (EP2):        /Snapshots/{sim_type}/{realization}/snapdir_{snap:03d}/
#   LH params:              GitHub (not Globus)
#
# File formats:
#   Pk (real):       2-column text (k [h/Mpc], P(k) [(Mpc/h)^3])
#   Pk (redshift):   4-column text (k, P0, P2, P4) in same units
#   Linear Pk:       2-column text (k [h/Mpc], P(k) [(Mpc/h)^3])
#   Snapshots:       compressed HDF5 (Blosc filter, requires hdf5plugin), 8 files per snapshot
#   FoF halos:       plain binary, 6-uint32 header + 84 bytes/halo (auto-detected by gadget_utils)
#
QUIJOTE_EP1 = 'f4863854-3819-11eb-b171-0ee0d5d9299f'  # Quijote_simulations
QUIJOTE_EP2 = 'e0eae0aa-5bca-11ea-9683-0e56c063f437'  # Quijote_simulations2

_c_kms = 299792.458  # speed of light in km/s


####################################################################################################
# Snapshot redshifts


_snapshot_redshifts = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.5, 4: 0.0}
_valid_redshifts = [0.0, 0.5, 1.0, 2.0, 3.0]
_redshift_to_snap = {0.0: 4, 0.5: 3, 1.0: 2, 2.0: 1, 3.0: 0}


def _snap_num(redshift):
    """Convert a redshift to the corresponding snapshot number.

    Args:
        redshift (float): One of {0, 0.5, 1, 2, 3}. Roundoff-tolerant (atol=1e-6).

    Returns:
        int: snapshot number (0-4)
    """
    z = float(redshift)
    for zval in _valid_redshifts:
        if abs(z - zval) < 1e-6:
            return _redshift_to_snap[zval]
    raise ValueError(f'Invalid redshift {redshift}. Must be one of: 0, 0.5, 1, 2, 3.')


####################################################################################################
# Simulation types and realization counts


_sim_types = {
    # Fiducial
    'fiducial': 15000,
    'fiducial_ZA': 500,
    'fiducial_LR': 1000,
    'fiducial_HR': 100,
    # Single-parameter derivatives
    'Om_p': 500, 'Om_m': 500,
    'Ob2_p': 500, 'Ob2_m': 500,
    'h_p': 500, 'h_m': 500,
    'ns_p': 500, 'ns_m': 500,
    's8_p': 500, 's8_m': 500,
    # Massive neutrinos
    'Mnu_p': 500, 'Mnu_pp': 500, 'Mnu_ppp': 500,
    # Dark energy
    'w_p': 500, 'w_m': 500,
    # DC mode (separate universe)
    'DC_p': 500, 'DC_m': 500,
    # Primordial non-Gaussianity
    'LC_p': 500, 'LC_m': 500,
    'EQ_p': 500, 'EQ_m': 500,
    'OR_CMB_p': 500, 'OR_CMB_m': 500,
    'OR_LSS_p': 500, 'OR_LSS_m': 500,
    # Modified gravity f(R)
    'fR_p': 500, 'fR_pp': 500, 'fR_ppp': 500, 'fR_pppp': 500,
    # Latin hypercubes
    'latin_hypercube': 2000,
    'latin_hypercube_HR': 2000,
    'latin_hypercube_nwLH': 2000,
    # BSQ (Big Sobol Sequence)
    'BSQ': 32768,
}


def sim_realizations(sim_type):
    """Return the number of realizations for a given simulation type.

    Args:
        sim_type (str): e.g. 'fiducial', 'Om_p', 'latin_hypercube', etc.

    Returns:
        int
    """
    if sim_type not in _sim_types:
        raise ValueError(f"Unknown sim_type '{sim_type}'. Must be one of: {sorted(_sim_types.keys())}")
    return _sim_types[sim_type]


def _check_sim_type(sim_type):
    """Validate sim_type. Raises ValueError if unknown."""
    if sim_type not in _sim_types:
        raise ValueError(f"Unknown sim_type '{sim_type}'. Must be one of: {sorted(_sim_types.keys())}")


def _check_realization(sim_type, realization):
    """Validate realization index. Raises ValueError if out of range."""
    _check_sim_type(sim_type)
    nreal = _sim_types[sim_type]
    if not (0 <= realization < nreal):
        raise ValueError(f"realization={realization} out of range for sim_type='{sim_type}' (must be 0 <= r < {nreal})")


####################################################################################################
# Cosmological parameters


# Cosmological parameters for fixed-cosmology sim types.
# Keys: Om, Ob, h, ns, sigma8, Mnu, w
_fiducial = {'Om': 0.3175, 'Ob': 0.049, 'h': 0.6711, 'ns': 0.9624, 'sigma8': 0.834, 'Mnu': 0.0, 'w': -1.0}

_cosmologies = {
    'fiducial':      _fiducial,
    'fiducial_ZA':   _fiducial,
    'fiducial_LR':   _fiducial,
    'fiducial_HR':   _fiducial,
    'Om_p':     {**_fiducial, 'Om': 0.3275},
    'Om_m':     {**_fiducial, 'Om': 0.3075},
    'Ob2_p':    {**_fiducial, 'Ob': 0.051},
    'Ob2_m':    {**_fiducial, 'Ob': 0.047},
    'h_p':      {**_fiducial, 'h': 0.6911},
    'h_m':      {**_fiducial, 'h': 0.6511},
    'ns_p':     {**_fiducial, 'ns': 0.9824},
    'ns_m':     {**_fiducial, 'ns': 0.9424},
    's8_p':     {**_fiducial, 'sigma8': 0.849},
    's8_m':     {**_fiducial, 'sigma8': 0.819},
    'Mnu_p':    {**_fiducial, 'Mnu': 0.1},
    'Mnu_pp':   {**_fiducial, 'Mnu': 0.2},
    'Mnu_ppp':  {**_fiducial, 'Mnu': 0.4},
    'w_p':      {**_fiducial, 'w': -1.05},
    'w_m':      {**_fiducial, 'w': -0.95},
    'DC_p':     _fiducial,
    'DC_m':     _fiducial,
    # PNG sim types all use fiducial cosmology
    'LC_p': _fiducial, 'LC_m': _fiducial,
    'EQ_p': _fiducial, 'EQ_m': _fiducial,
    'OR_CMB_p': _fiducial, 'OR_CMB_m': _fiducial,
    'OR_LSS_p': _fiducial, 'OR_LSS_m': _fiducial,
    # Modified gravity
    'fR_p': _fiducial, 'fR_pp': _fiducial,
    'fR_ppp': _fiducial, 'fR_pppp': _fiducial,
}

# Per-realization sim types (latin_hypercube, BSQ, etc.)
_per_realization_sim_types = {'latin_hypercube', 'latin_hypercube_HR', 'latin_hypercube_nwLH', 'BSQ'}


# Cache for LH parameter files (loaded once per lh_type).
_lh_params_cache = {}


def _get_cosmology_dict(sim_type, realization=None):
    """Return the cosmological parameters for a given simulation.

    For fixed-cosmology sim types (fiducial, Om_p, etc.), returns parameters
    from the hardcoded _cosmologies dict. For per-realization sim types
    (latin_hypercube, BSQ), reads the parameter file and returns the row
    for the given realization.

    Args:
        sim_type (str): Simulation type.
        realization (int or None): Required for latin_hypercube/BSQ, ignored otherwise.

    Returns:
        dict with keys: 'Om', 'Ob', 'h', 'ns', 'sigma8', 'Mnu', 'w'
    """
    _check_sim_type(sim_type)

    if sim_type in _cosmologies:
        return _cosmologies[sim_type]

    if sim_type in _per_realization_sim_types:
        if realization is None:
            raise ValueError(f"realization is required for sim_type='{sim_type}'")

        # Map sim_type to lh_type for read_lh_params
        if sim_type in ('latin_hypercube', 'latin_hypercube_HR'):
            lh_type = 'standard'
        elif sim_type == 'latin_hypercube_nwLH':
            lh_type = 'nwLH'
        elif sim_type == 'BSQ':
            lh_type = 'BSQ'
        else:
            raise ValueError(f"No cosmology data for sim_type='{sim_type}'")

        params = _get_lh_params_cached(lh_type)
        return {k: float(params[k][realization]) for k in params}

    raise ValueError(f"No cosmology data for sim_type='{sim_type}'")


def _get_h(sim_type, realization=None):
    """Convenience: return just h for a given sim. Used internally by all read functions."""
    return _get_cosmology_dict(sim_type, realization)['h']


def _get_lh_params_cached(lh_type):
    """Return cached LH parameter dict (loading from disk if needed)."""
    if lh_type not in _lh_params_cache:
        _lh_params_cache[lh_type] = _read_lh_params_file(lh_type)
    return _lh_params_cache[lh_type]


_LH_GITHUB_BASE = 'https://raw.githubusercontent.com/franciscovillaescusa/Quijote-simulations/master'

_lh_param_info = {
    'standard': {
        'url': f'{_LH_GITHUB_BASE}/latin_hypercube/latin_hypercube_params.txt',
        'relpath': 'latin_hypercube/latin_hypercube_params.txt',
        'col_names': ['Om', 'Ob', 'h', 'ns', 'sigma8'],
    },
    'nwLH': {
        'url': f'{_LH_GITHUB_BASE}/latin_hypercube_nwLH/latin_hypercube_params.txt',
        'relpath': 'latin_hypercube_nwLH/latin_hypercube_params.txt',
        'col_names': ['Om', 'Ob', 'h', 'ns', 'sigma8', 'Mnu', 'w'],
    },
}


def _read_lh_params_file(lh_type):
    """Read a Latin Hypercube parameter file, downloading from GitHub if needed.

    Returns dict with keys: 'Om', 'Ob', 'h', 'ns', 'sigma8' (and optionally 'Mnu', 'w').
    """
    if lh_type not in _lh_param_info:
        raise ValueError(f"Unknown lh_type '{lh_type}'. Must be one of: {sorted(_lh_param_info.keys())}")

    info = _lh_param_info[lh_type]
    quijote_base = os.path.join(io_utils.get_data_dir(), 'quijote')
    abspath = os.path.join(quijote_base, info['relpath'])

    if not os.path.exists(abspath):
        io_utils.wget(abspath, info['url'])

    data = np.loadtxt(abspath)
    return {info['col_names'][i]: data[:, i] for i in range(len(info['col_names']))}


####################################################################################################
# Path resolution and download


def _quijote_path(relpath, download=False, dlfunc=None, endpoint=None, remote_path=None):
    """Resolve a Quijote data path, optionally downloading via Globus.

    The local path is: io_utils.get_data_dir()/quijote/{relpath}

    Args:
        relpath (str): Relative path under the local quijote data directory.
        download (bool): If True, download via Globus if not on disk.
        dlfunc (str or None): Name of calling function (for error messages).
        endpoint (str or None): Globus endpoint UUID (required if download=True).
        remote_path (str or None): Path on the Globus endpoint. If None, uses '/{relpath}'.
    """
    quijote_base = os.path.join(io_utils.get_data_dir(), 'quijote')
    abspath = os.path.join(quijote_base, relpath)

    if os.path.exists(abspath):
        return abspath

    if not download:
        msg = f'Path {abspath} not found.'
        if dlfunc:
            msg += f'\nTo auto-download, call {dlfunc}() with download=True.'
        raise RuntimeError(msg)

    # Download via Globus
    if endpoint is None:
        raise RuntimeError(f'No Globus endpoint specified for downloading {relpath}')
    if remote_path is None:
        remote_path = '/' + relpath
    is_dir = not os.path.splitext(remote_path)[1]  # heuristic: no extension = directory
    globus_utils.globus_download(
        endpoint, remote_path, abspath,
        recursive=is_dir, label=f'kszx.quijote: {relpath}'
    )
    return abspath


####################################################################################################
# FoF halo catalogs


def read_halos(sim_type, realization, redshift, download=False, min_mass=None):
    r"""Read a Friends-of-Friends halo catalog, returning a kszx.Catalog.

    Args:
        sim_type (str): Simulation type, e.g. 'fiducial', 'Om_p', 'latin_hypercube'.
        realization (int): Realization index (0-indexed).
        redshift (float): One of {0, 0.5, 1, 2, 3}.
        download (bool): If True and data is not on disk, auto-download via Globus.
        min_mass (float or None): If specified, only return halos with mass >= min_mass
            (in units of M_sun).

    Returns:
        kszx.Catalog with columns:

          - x, y, z: halo positions in Mpc
          - vx, vy, vz: halo velocities (dimensionless, v/c)
          - mass: halo mass in M_sun
          - npart: number of particles per halo
    """
    _check_realization(sim_type, realization)
    snap = _snap_num(redshift)
    relpath = f'Halos/FoF/{sim_type}/{realization}/groups_{snap:03d}'
    local_dir = _quijote_path(
        relpath, download, dlfunc='kszx.quijote.read_halos',
        endpoint=QUIJOTE_EP2, remote_path=f'/{relpath}'
    )
    h = _get_h(sim_type, realization)
    raw = gadget_utils.read_fof(local_dir)
    return _fof_to_catalog(raw, h, redshift, min_mass=min_mass)


def _fof_to_catalog(raw, h, redshift, min_mass=None):
    """Convert raw FoF dict (Gadget units) to kszx.Catalog (physical units)."""
    catalog = Catalog(name='Quijote FoF halos')
    catalog.add_column('x', raw['pos'][:, 0] / (1e3 * h))     # kpc/h -> Mpc
    catalog.add_column('y', raw['pos'][:, 1] / (1e3 * h))
    catalog.add_column('z', raw['pos'][:, 2] / (1e3 * h))
    # FoF velocities on disk are v_peculiar * a; multiply by (1+z) to get v_peculiar
    catalog.add_column('vx', raw['vel'][:, 0] * (1 + redshift) / _c_kms)  # -> v/c
    catalog.add_column('vy', raw['vel'][:, 1] * (1 + redshift) / _c_kms)
    catalog.add_column('vz', raw['vel'][:, 2] * (1 + redshift) / _c_kms)
    catalog.add_column('mass', raw['mass'] * 1e10 / h)          # 10^10 M_sun/h -> M_sun
    catalog.add_column('npart', raw['npart'])
    if min_mass is not None:
        catalog.apply_boolean_mask(catalog.mass >= min_mass)
    return catalog


####################################################################################################
# Power spectra


def read_pk(sim_type, realization, redshift, space='real', species='m', download=False):
    r"""Read a pre-computed power spectrum.

    Args:
        sim_type (str): Simulation type.
        realization (int): Realization index.
        redshift (float): One of {0, 0.5, 1, 2, 3}.
        space (str): 'real' for real-space P(k), or 'redshift' for redshift-space multipoles.
        species (str): 'm' for total matter, 'cb' for CDM+baryons (neutrino sims only).
        download (bool): If True and data is not on disk, auto-download via Globus.

    Returns:
        If space='real':
            dict with keys:

              - 'k': 1-d array, wavenumber in Mpc^{-1}
              - 'pk': 1-d array, P(k) in Mpc^3

        If space='redshift':
            dict with keys:

              - 'k': 1-d array, wavenumber in Mpc^{-1}
              - 'pk0': 1-d array, monopole P_0(k) in Mpc^3
              - 'pk2': 1-d array, quadrupole P_2(k) in Mpc^3
              - 'pk4': 1-d array, hexadecapole P_4(k) in Mpc^3
    """
    _check_realization(sim_type, realization)
    snap = _snap_num(redshift)
    h = _get_h(sim_type, realization)

    # Format redshift string for filename (0 -> "0", 0.5 -> "0.5", etc.)
    zstr = f'{redshift:g}'

    # On the Globus endpoint, Pk files live at /Pk/matter/{sim_type}/{realization}/
    # Locally we store at Pk/matter/{sim_type}/{realization}/ to mirror the remote layout.
    pk_dir = f'Pk/matter/{sim_type}/{realization}'

    if space == 'real':
        filename = f'Pk_{species}_z={zstr}.txt'
        filepath = _quijote_path(
            f'{pk_dir}/{filename}',
            download, dlfunc='kszx.quijote.read_pk',
            endpoint=QUIJOTE_EP1, remote_path=f'/{pk_dir}/{filename}'
        )
        data = np.loadtxt(filepath)
        return {
            'k': data[:, 0] * h,        # h/Mpc -> Mpc^-1
            'pk': data[:, 1] / h**3,     # (Mpc/h)^3 -> Mpc^3
        }
    elif space == 'redshift':
        # Redshift-space Pk along axis 0,1,2 (files: Pk_m_RS{axis}_z=0.txt)
        # Default to axis=2 (z-axis) — 4-column format: k P0 P2 P4
        filename = f'Pk_{species}_RS2_z={zstr}.txt'
        filepath = _quijote_path(
            f'{pk_dir}/{filename}',
            download, dlfunc='kszx.quijote.read_pk',
            endpoint=QUIJOTE_EP1, remote_path=f'/{pk_dir}/{filename}'
        )
        data = np.loadtxt(filepath)
        return {
            'k': data[:, 0] * h,         # h/Mpc -> Mpc^-1
            'pk0': data[:, 1] / h**3,    # (Mpc/h)^3 -> Mpc^3
            'pk2': data[:, 2] / h**3,
            'pk4': data[:, 3] / h**3,
        }
    else:
        raise ValueError(f"space must be 'real' or 'redshift', got '{space}'")


####################################################################################################
# Linear power spectra


def read_linear_pk(sim_type, realization=None, download=False):
    r"""Read the CAMB-generated linear matter power spectrum for a given cosmology.

    For sim types with a single cosmology (e.g. 'fiducial', 'Om_p'), realization
    is ignored (all realizations share the same linear Pk). For 'latin_hypercube',
    realization is required (each has a different cosmology).

    Args:
        sim_type (str): Simulation type.
        realization (int or None): Required for latin_hypercube, ignored otherwise.
        download (bool): If True and data is not on disk, auto-download via Globus.

    Returns:
        dict with keys:

          - 'k': 1-d array, wavenumber in Mpc^{-1}
          - 'pk': 1-d array, linear P(k) in Mpc^3
    """
    _check_sim_type(sim_type)
    h = _get_h(sim_type, realization)

    if sim_type in _per_realization_sim_types:
        if realization is None:
            raise ValueError(f"realization is required for sim_type='{sim_type}'")
        relpath = f'Linear_Pk/{sim_type}/{realization}/CAMB_TABLES/CAMB_matterpow_0.dat'
    else:
        relpath = f'Linear_Pk/{sim_type}/CAMB_TABLES/CAMB_matterpow_0.dat'

    filepath = _quijote_path(
        relpath, download, dlfunc='kszx.quijote.read_linear_pk',
        endpoint=QUIJOTE_EP1, remote_path=f'/{relpath}'
    )
    data = np.loadtxt(filepath)
    return {
        'k': data[:, 0] * h,        # h/Mpc -> Mpc^-1
        'pk': data[:, 1] / h**3,    # (Mpc/h)^3 -> Mpc^3
    }


####################################################################################################
# Latin Hypercube parameters


def read_lh_params(lh_type='standard', download=False):
    r"""Read the cosmological parameters for all Latin Hypercube realizations.

    Args:
        lh_type (str): 'standard' for the 5-parameter LH (2000 sims),
            'nwLH' for the 7-parameter LH including neutrinos and w.
        download (bool): If True and data is not on disk, auto-download via Globus.

    Returns:
        dict with keys:

          - 'Om': 1-d array of Omega_m values (length 2000)
          - 'Ob': 1-d array of Omega_b values
          - 'h': 1-d array of h values
          - 'ns': 1-d array of n_s values
          - 'sigma8': 1-d array of sigma_8 values

        For lh_type='nwLH', also:

          - 'Mnu': 1-d array of M_nu values (eV)
          - 'w': 1-d array of w values
    """
    # read_lh_params uses the same internal reader as _get_cosmology_dict,
    # which downloads from GitHub if needed. The 'download' arg is accepted
    # for API consistency but is not needed (the file is small and always downloaded).
    return _read_lh_params_file(lh_type)


####################################################################################################
# Snapshots (compressed HDF5)


def read_snapshot(sim_type, realization, redshift, fields=('pos',), ptype=1, download=False):
    r"""Read particle data from a compressed HDF5 snapshot.

    Requires ``hdf5plugin`` to be installed (``conda install -c conda-forge hdf5plugin``).

    Args:
        sim_type (str): Simulation type.
        realization (int): Realization index.
        redshift (float): One of {0, 0.5, 1, 2, 3}.
        fields (tuple of str): Which fields to read:

          - 'pos' -- positions in Mpc
          - 'vel' -- velocities (dimensionless, v/c)
          - 'ids' -- particle IDs

        ptype (int): Particle type. 1 = CDM (default), 2 = neutrinos.
        download (bool): If True and data is not on disk, auto-download via Globus.

    Returns:
        dict with keys from 'fields':

          - 'pos': float32 array, shape (N,3), positions in Mpc (comoving)
          - 'vel': float32 array, shape (N,3), velocities (dimensionless, v/c)
          - 'ids': uint32 array, shape (N,), particle IDs
    """
    _check_realization(sim_type, realization)
    snap = _snap_num(redshift)
    relpath = f'Snapshots/{sim_type}/{realization}/snapdir_{snap:03d}'
    # Snapshots are on ep2 for all sim types (fiducial, derivatives, etc.)
    ep = QUIJOTE_EP2
    snapdir = _quijote_path(
        relpath, download, dlfunc='kszx.quijote.read_snapshot',
        endpoint=ep, remote_path=f'/{relpath}'
    )
    h = _get_h(sim_type, realization)

    # Try single-file first (snap_NNN.hdf5), then multi-file (snap_NNN.0.hdf5)
    single_file = os.path.join(snapdir, f'snap_{snap:03d}.hdf5')
    multi_file0 = os.path.join(snapdir, f'snap_{snap:03d}.0.hdf5')

    if os.path.exists(single_file):
        hdr, data = _read_snapshot_hdf5(single_file, ptype, fields)
        _snapshot_to_physical(hdr, data, h)
        return data

    # Multi-file snapshot
    hdr0, data0 = _read_snapshot_hdf5(multi_file0, ptype, fields)
    num_files = int(hdr0['NumFilesPerSnapshot'])

    if num_files == 1:
        _snapshot_to_physical(hdr0, data0, h)
        return data0

    chunks = [data0]
    for i in range(1, num_files):
        filepath = os.path.join(snapdir, f'snap_{snap:03d}.{i}.hdf5')
        _, datai = _read_snapshot_hdf5(filepath, ptype, fields)
        chunks.append(datai)

    merged = {}
    for key in fields:
        merged[key] = np.concatenate([c[key] for c in chunks], axis=0)
    _snapshot_to_physical(hdr0, merged, h)
    return merged


def _read_snapshot_hdf5(filepath, ptype=1, fields=('pos',)):
    """Read a single compressed HDF5 snapshot file.

    Args:
        filepath (str): Path to the HDF5 snapshot file (e.g. snap_004.hdf5).
        ptype (int): Particle type (1=CDM, 2=neutrinos).
        fields (tuple of str): 'pos', 'vel', 'ids'.

    Returns:
        (header_dict, data_dict) in raw on-disk units.
    """
    try:
        import h5py
    except ImportError:
        raise RuntimeError('h5py is not installed. Please install it: conda install h5py')

    try:
        import hdf5plugin  # noqa: F401 — registers Blosc decompression filter
    except ImportError:
        raise RuntimeError(
            'hdf5plugin is not installed. It is required for reading compressed Quijote snapshots.\n'
            'Install it with: conda install -c conda-forge hdf5plugin\n'
        )

    data = {}
    with h5py.File(filepath, 'r') as f:
        hdr = dict(f['Header'].attrs)

        group_name = f'PartType{ptype}'
        if group_name not in f:
            raise RuntimeError(f'{filepath} does not contain {group_name}')
        g = f[group_name]

        if 'pos' in fields:
            data['pos'] = g['Coordinates'][:]        # shape (N, 3), float32
        if 'vel' in fields:
            data['vel'] = g['Velocities'][:]         # shape (N, 3), float32
        if 'ids' in fields:
            data['ids'] = g['ParticleIDs'][:]        # shape (N,), uint32

    return hdr, data


def _snapshot_to_physical(hdr, data, h):
    """Convert raw HDF5 snapshot data to kszx physical units (in-place)."""
    a = float(hdr['Time'])        # scale factor
    sqrt_a = np.sqrt(a)

    if 'pos' in data:
        box = float(hdr['BoxSize'])
        if box > 10000:
            data['pos'] = data['pos'] / (1e3 * h)   # kpc/h -> Mpc
        else:
            data['pos'] = data['pos'] / h            # Mpc/h -> Mpc

    if 'vel' in data:
        # Gadget velocity convention: v_internal = v_peculiar * sqrt(a)
        data['vel'] = data['vel'] / (sqrt_a * _c_kms)  # -> v/c


####################################################################################################
# Bulk download


def download(sim_type, realizations, products=('halos',), redshifts=(0,)):
    r"""Bulk-download Quijote data via Globus.

    Submits a single Globus transfer task for all requested files, which is much
    faster than downloading one-at-a-time (Globus parallelizes across files).

    Data is stored under ``io_utils.get_data_dir()/quijote/``.

    Args:
        sim_type (str): Simulation type, e.g. 'fiducial'.
        realizations (int, list, or range): Which realizations to download.
            E.g. 1000, range(1000), or [0, 5, 10].
            If an int N is passed, interpreted as range(N).
        products (tuple of str): Which data products to download.
            Options: 'halos', 'pk', 'snapshots', 'linear_pk'.
        redshifts (tuple of float): Which redshifts to download.
            Options: 0, 0.5, 1, 2, 3.

    Example::

        # Download FoF halos and power spectra for first 1000 fiducial sims at z=0
        kszx.quijote.download('fiducial', 1000, products=('halos','pk'), redshifts=(0,))
    """
    _check_sim_type(sim_type)

    if isinstance(realizations, int):
        realizations = range(realizations)

    quijote_base = os.path.join(io_utils.get_data_dir(), 'quijote')

    # Items grouped by endpoint: {endpoint_id: [(remote_path, local_abspath, recursive), ...]}
    ep_items = {QUIJOTE_EP1: [], QUIJOTE_EP2: []}

    snap_ep = QUIJOTE_EP2

    for r in realizations:
        _check_realization(sim_type, r)
        for z in redshifts:
            snap = _snap_num(z)
            zstr = f'{z:g}'

            if 'halos' in products:
                relpath = f'Halos/FoF/{sim_type}/{r}/groups_{snap:03d}'
                ep_items[QUIJOTE_EP2].append((f'/{relpath}', os.path.join(quijote_base, relpath), True))

            if 'pk' in products:
                relpath = f'Pk/matter/{sim_type}/{r}/Pk_m_z={zstr}.txt'
                ep_items[QUIJOTE_EP1].append((f'/{relpath}', os.path.join(quijote_base, relpath), False))

            if 'snapshots' in products:
                relpath = f'Snapshots/{sim_type}/{r}/snapdir_{snap:03d}'
                ep_items[snap_ep].append((f'/{relpath}', os.path.join(quijote_base, relpath), True))

        if 'linear_pk' in products:
            if sim_type in _per_realization_sim_types:
                relpath = f'Linear_Pk/{sim_type}/{r}/CAMB_TABLES/CAMB_matterpow_0.dat'
            else:
                relpath = f'Linear_Pk/{sim_type}/CAMB_TABLES/CAMB_matterpow_0.dat'
            ep_items[QUIJOTE_EP1].append((f'/{relpath}', os.path.join(quijote_base, relpath), False))

    for ep_id, items in ep_items.items():
        if len(items) == 0:
            continue
        globus_utils.globus_download_batch(
            ep_id, items,
            label=f'kszx.quijote: {sim_type} ({len(items)} items)'
        )
