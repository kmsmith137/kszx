"""kszx.websky: read Websky v0.4 lightcone data products.

Hosted on NERSC at /global/cfs/projectdirs/sobs/v4_sims/mbs/websky/0.4/.
Files are fetched via scp from a user-configurable host alias (default
'perlmutter'); see WEBSKY_SSH_HOST and WEBSKY_REMOTE_BASE below.

Units (kszx convention):
  - positions in Mpc (Websky on disk: Mpc, no h-factor)
  - masses in M_sun (derived from Lagrangian R via Websky's Omega_m, h)
  - velocities dimensionless v/c (Websky on disk: km/s)
  - maps in RING healpix order, uK_CMB where applicable

References:
  - Stein, Alvarez, Bond 2020 (arXiv:2001.08787)
  - v0.4 README (only available on NERSC, not publicly mirrored)
"""

import functools
import os
import subprocess

import numpy as np
import healpy

from . import io_utils
from . import mlhack
from .Catalog import Catalog


####################################################################################################
# Constants


# NERSC location of the v0.4 release. Override these if your ssh alias is
# named differently or you've pointed it at a local scratch copy.
WEBSKY_REMOTE_BASE = '/global/cfs/projectdirs/sobs/v4_sims/mbs/websky/0.4'
WEBSKY_SSH_HOST    = 'perlmutter'

# Cosmological parameters used to generate Websky (from the on-disk
# cosmology.py). Copied here so we don't have to fetch a remote file.
WEBSKY_COSMO = {'Om': 0.31, 'Ob': 0.049, 'h': 0.68, 'ns': 0.965, 'sigma8': 0.81}

# Pre-computed scalar amplitude that gives sigma8 = 0.81 for the Websky
# cosmology with tau = 0.0561. To regenerate: run CAMB with a trial As, measure
# sigma8, then rescale As by (sigma8_target / sigma8_trial)**2.
_WEBSKY_AS  = 2.020958e-09
_WEBSKY_TAU = 0.0561

# 45 CIB frequencies provided by v0.4. Filenames are 'cib_NNNN.N.fits'
# (zero-padded to 4 digits before the decimal, one digit after).
CIB_FREQS_GHZ = (
    18.7, 21.6, 24.5, 27.3, 30.0, 35.9, 41.7, 44.0, 47.4, 63.9,
    67.8, 70.0, 73.7, 79.6, 90.2, 100.0, 111.0, 129.0, 143.0, 153.0,
    164.0, 189.0, 210.0, 217.0, 232.0, 256.0, 275.0, 294.0, 306.0, 314.0,
    340.0, 353.0, 375.0, 409.0, 467.0, 525.0, 545.0, 584.0, 643.0, 729.0,
    817.0, 857.0, 906.0, 994.0, 1080.0,
)

_C_KMS    = 299792.458
_T_CMB_K  = 2.7255
_HPLANCK  = 6.62607015e-34   # J*s
_KBOLTZ   = 1.380649e-23     # J/K
_CLIGHT_M = 2.99792458e8     # m/s


####################################################################################################
# Cosmology


def cosmological_params():
    r"""Return a :class:`~kszx.CosmologicalParams` for the Websky cosmology.

    The scalar amplitude is pre-calibrated to match Websky's sigma8 = 0.81
    with tau = 0.0561. Pre-computed once and hardcoded so we don't run CAMB
    at import time.
    """
    from .Cosmology import CosmologicalParams

    p = CosmologicalParams()
    p.h          = WEBSKY_COSMO['h']
    p.ns         = WEBSKY_COSMO['ns']
    p.ombh2      = WEBSKY_COSMO['Ob'] * WEBSKY_COSMO['h']**2
    p.omch2      = (WEBSKY_COSMO['Om'] - WEBSKY_COSMO['Ob']) * WEBSKY_COSMO['h']**2
    p.mnu        = 0.0
    p.tau        = _WEBSKY_TAU
    p.scalar_amp = _WEBSKY_AS
    return p


# Lazy, cached Cosmology accessible as `kszx.websky.cosmology` (no parens).
# PEP 562 module __getattr__ + functools.cache: built once, on first access.
#
# Caveats of this lazy-attribute pattern:
#   - `from kszx.websky import cosmology` triggers construction at the
#     importer's import time (the import machinery resolves the name eagerly).
#     If the importer wants lazy semantics, they must use the attribute-access
#     form `kszx.websky.cosmology` instead of `from ... import cosmology`.
#   - `dir(kszx.websky)` would normally hide attributes only served by
#     __getattr__, so we define __dir__ as well to keep tab-completion in
#     IPython / Jupyter working.
@functools.cache
def _build_cosmology():
    from .Cosmology import Cosmology
    return Cosmology(cosmological_params())


def __getattr__(name):
    if name == 'cosmology':
        return _build_cosmology()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + ['cosmology'])


####################################################################################################
# Path resolution and download


# Files in the v0.4 NERSC directory whose symlink chains are broken; we patch
# the remote path at scp time so the user gets the actual file. Local cache
# layout is unchanged (still under .../websky/0.4/).
#
# halos.pksc: intended chain is 0.4/halos.pksc -> ../0.3/halos.pksc ->
# ../0.2/halos.pksc, but the 0.3 link is broken; bypass to 0.2 directly.
_BROKEN_SYMLINK_FIXUPS = {
    'halos.pksc': '../0.2/halos.pksc',
}


def _websky_path(relpath, download=False, dlfunc=None):
    """Resolve a Websky file's local path, fetching via scp if requested."""
    abspath = os.path.join(io_utils.get_data_dir(), 'websky', '0.4', relpath)
    if not io_utils.do_download(abspath, download, dlfunc):
        return abspath
    _websky_scp(relpath, abspath)
    return abspath


def _websky_scp(relpath, abspath):
    """Fetch a single Websky file from NERSC via scp.

    The remote is addressed as f'{WEBSKY_SSH_HOST}:{WEBSKY_REMOTE_BASE}/{relpath}',
    unless `relpath` appears in `_BROKEN_SYMLINK_FIXUPS`, in which case we
    substitute the patched remote path. On NERSC many of these files are
    symlinks; scp follows them and writes a plain local file (no symlinks are
    created locally).

    To avoid leaving a half-written file on disk if scp is interrupted, we
    download to `<abspath>.tmp` and rename only after scp succeeds.

    On failure, prints the exact scp command that was tried and a hint for
    setting up ~/.ssh/config + the NERSC sshproxy script.
    """
    io_utils.mkdir_containing(abspath)

    remote_relpath = relpath
    if relpath in _BROKEN_SYMLINK_FIXUPS:
        remote_relpath = _BROKEN_SYMLINK_FIXUPS[relpath]
        print(f'kszx.websky HACK: NERSC symlink for {relpath} is broken; '
              f'fetching {remote_relpath} instead\n', end='')

    tmp_path = abspath + '.tmp'
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    src = f'{WEBSKY_SSH_HOST}:{WEBSKY_REMOTE_BASE}/{remote_relpath}'
    cmd = ['scp', src, tmp_path]
    print(f'Running: {" ".join(cmd)}')
    result = subprocess.run(cmd)
    if result.returncode != 0:
        # Clean up the partial/empty file so the next call doesn't skip the download.
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise RuntimeError(
            f"\nFailed to download {relpath} from NERSC.\n"
            f"Failed command: {' '.join(cmd)}\n\n"
            f"To make this work, add an entry to your ~/.ssh/config that maps the\n"
            f"host alias '{WEBSKY_SSH_HOST}' to a NERSC login host, e.g.:\n\n"
            f"    Host {WEBSKY_SSH_HOST}\n"
            f"        HostName perlmutter.nersc.gov\n"
            f"        User <your-nersc-username>\n\n"
            f"Tip: NERSC's sshproxy script (https://docs.nersc.gov/connect/mfa/#sshproxy)\n"
            f"     issues a short-lived (~24h) SSH key after one MFA prompt, so a bulk\n"
            f"     download won't pause for your token on every file. Run it once before\n"
            f"     kicking off `python -m kszx download_websky ...`.\n"
        )
    os.rename(tmp_path, abspath)


####################################################################################################
# Unit conversion helpers


def _kcmb_to_mjy_per_sr(freq_ghz):
    r"""K_CMB -> MJy/sr conversion factor for a delta-function bandpass.

    Returns f such that DeltaI[MJy/sr] = f * DeltaT[K_CMB] for a small CMB
    blackbody-temperature perturbation around T_CMB = 2.7255 K. Equivalently,
    f = (dB_nu/dT)|_{T=T_CMB}, with units converted from W/m^2/Hz/sr/K to MJy/sr/K.

    Sanity check: at 100/143/217/353/545/857 GHz this returns ~239/372/484/287/57/2.27,
    closely matching the Planck 2018 bandpass-integrated values (244.1, 371.74,
    483.69, 287.45, 58.04, 2.27 MJy/sr/K).
    """
    nu = freq_ghz * 1e9   # Hz
    x = _HPLANCK * nu / (_KBOLTZ * _T_CMB_K)
    prefactor = 2 * _KBOLTZ**3 * _T_CMB_K**2 / (_HPLANCK**2 * _CLIGHT_M**2)
    dBdT = prefactor * x**4 * np.exp(x) / (np.exp(x) - 1)**2
    # 1 W/m^2/Hz = 1e26 Jy = 1e20 MJy
    return dBdT * 1e20


def _ysz_kernel(freq_ghz):
    r"""Non-relativistic tSZ spectral kernel g(nu) = x*coth(x/2) - 4.

    DeltaT_CMB / T_CMB = y * g(nu), so DeltaT[uK] = y * T_CMB[uK] * g(nu).
    """
    x = _HPLANCK * (freq_ghz * 1e9) / (_KBOLTZ * _T_CMB_K)
    return x / np.tanh(x / 2) - 4


####################################################################################################
# Halo catalog


def read_halos(mmin=None, zmin=None, zmax=None, download=False):
    r"""Read the Websky halo catalog and return a :class:`~kszx.Catalog`.

    The on-disk catalog (`halos.pksc`) is ~32 GB with ~9e8 halos. To keep peak
    memory bounded, this function streams the file in chunks and applies the
    `mmin` / `zmin` / `zmax` cuts inside the streaming loop.

    Function args:

      - ``mmin`` (float or None): minimum halo mass in M_sun (default: no cut).
      - ``zmin`` (float or None): minimum true redshift (default: no cut).
      - ``zmax`` (float or None): maximum true redshift (default: no cut).
      - ``download`` (bool): if True, fetch halos.pksc from NERSC via scp.

    Returns a :class:`~kszx.Catalog` with columns:

      - ``ra_deg, dec_deg``: sky position in degrees
      - ``z``: true cosmological redshift (from comoving distance, no RSD).
        If you want the observed redshift including peculiar-velocity RSD,
        compute it from ``vr``: ``1 + zobs = (1 + z)(1 + vr)``.
      - ``vr, vtheta, vphi``: velocity components in spherical-polar coordinates,
        dimensionless (v/c)
      - ``M``: halo mass in M_sun
    """
    filepath = _websky_path('halos.pksc', download=download,
                            dlfunc='kszx.websky.read_halos')

    cosmo = _build_cosmology()
    rho_m0 = 2.775e11 * WEBSKY_COSMO['Om'] * WEBSKY_COSMO['h']**2  # M_sun/Mpc^3
    mass_prefactor = (4.0 * np.pi / 3.0) * rho_m0

    chunk_size = 10_000_000
    surviving = {k: [] for k in ('ra_deg', 'dec_deg', 'z',
                                  'vr', 'vtheta', 'vphi', 'M')}

    print(f'Reading {filepath}\n', end='')
    with open(filepath, 'rb') as f:
        header = np.fromfile(f, count=3, dtype=np.int32)
        if header.size != 3:
            raise RuntimeError(f'{filepath}: failed to read 12-byte header')
        Nhalo = int(header[0])

        # Sanity check: file size = 12 (header) + Nhalo * 40 bytes
        expected_size = 12 + Nhalo * 40
        actual_size = os.path.getsize(filepath)
        if actual_size != expected_size:
            raise RuntimeError(
                f'{filepath}: size mismatch (expected {expected_size} bytes for '
                f'Nhalo={Nhalo}, got {actual_size})')

        nread = 0
        while nread < Nhalo:
            n = min(chunk_size, Nhalo - nread)
            raw = np.fromfile(f, count=n*10, dtype=np.float32)
            if raw.size != n*10:
                raise RuntimeError(
                    f'{filepath}: short read at offset {nread} '
                    f'(expected {n*10} floats, got {raw.size})')
            raw = raw.reshape(n, 10)
            nread += n

            x, y, z = raw[:,0], raw[:,1], raw[:,2]
            vx_kms, vy_kms, vz_kms = raw[:,3], raw[:,4], raw[:,5]
            R = raw[:,6]

            M = mass_prefactor * R**3
            mask = np.ones(n, dtype=bool)
            if mmin is not None:
                mask &= (M >= mmin)
            if not mask.any():
                continue

            # Apply mass cut early to make the per-row work cheap.
            x, y, z = x[mask], y[mask], z[mask]
            vx_kms, vy_kms, vz_kms = vx_kms[mask], vy_kms[mask], vz_kms[mask]
            M = M[mask]

            chi = np.sqrt(x*x + y*y + z*z)
            z_true = cosmo.z(chi=chi)

            zmask = np.ones_like(z_true, dtype=bool)
            if zmin is not None:
                zmask &= (z_true >= zmin)
            if zmax is not None:
                zmask &= (z_true <= zmax)
            if not zmask.any():
                continue

            x, y, z = x[zmask], y[zmask], z[zmask]
            vx_kms, vy_kms, vz_kms = vx_kms[zmask], vy_kms[zmask], vz_kms[zmask]
            M, z_true = M[zmask], z_true[zmask]

            # Convert to v/c (dimensionless) before computing spherical components.
            vx = vx_kms / _C_KMS
            vy = vy_kms / _C_KMS
            vz = vz_kms / _C_KMS

            vec = np.column_stack([x, y, z])
            ra, dec = healpy.vec2ang(vec, lonlat=True)

            # epsilon for cartesian_to_spherical guards against r ~ 0;
            # 1 Mpc is plenty smaller than any chi we care about here.
            vr, vtheta, vphi = mlhack.cartesian_to_spherical(vec, vx, vy, vz, epsilon=1.0)

            surviving['ra_deg'].append(ra)
            surviving['dec_deg'].append(dec)
            surviving['z'].append(z_true)
            surviving['vr'].append(vr)
            surviving['vtheta'].append(vtheta)
            surviving['vphi'].append(vphi)
            surviving['M'].append(M)

    cat = Catalog(name='Websky halos', filename=filepath)
    for col in ('ra_deg', 'dec_deg', 'z', 'vr', 'vtheta', 'vphi', 'M'):
        if surviving[col]:
            arr = np.concatenate(surviving[col])
        else:
            arr = np.zeros(0, dtype=np.float64)
        cat.add_column(col, arr)
    return cat


####################################################################################################
# Healpix maps


def read_kappa(download=False):
    r"""Read the full-z lensing convergence map ``kap.fits``.

    Returns a 1-d HEALPix array in RING order, dimensionless (kappa).
    The full map covers z = 0..1100 (= kap_lt4.5 + kap_gt4.5); only the full
    sum is exposed here.
    """
    filepath = _websky_path('kap.fits', download=download,
                            dlfunc='kszx.websky.read_kappa')
    print(f'Reading {filepath}\n', end='')
    return healpy.read_map(filepath, dtype=np.float32)


def read_ksz(patchy=False, download=False):
    r"""Read a Websky kSZ map. Returns a HEALPix RING array in uK_CMB.

      - ``patchy=False`` (default): late-time kSZ from z<4.5 Websky (`ksz.fits`).
      - ``patchy=True``: kSZ from patchy reionization at z>5.5
        (`ksz_patchy.fits`), uncorrelated with the late-time map.

    Both files are stored in uK_CMB on disk per the v0.4 README.
    """
    relpath = 'ksz_patchy.fits' if patchy else 'ksz.fits'
    filepath = _websky_path(relpath, download=download,
                            dlfunc='kszx.websky.read_ksz')
    print(f'Reading {filepath}\n', end='')
    return healpy.read_map(filepath, dtype=np.float32)


def read_tsz(freq_ghz, download=False):
    r"""Read the Websky tSZ map (``tsz.fits``, HEALPix RING).

    On NERSC ``tsz.fits`` is a symlink to ``tsz_8192_hp.fits`` (nside=8192,
    ~800M pixels, ~3 GB at float32). scp follows the symlink and writes a
    plain local file.

      - ``freq_ghz=None``: return raw Compton-y (dimensionless).
      - ``freq_ghz=<float>``: return uK_CMB via ``y * T_CMB[uK] * g(nu)`` with
        the non-relativistic kernel ``g(nu) = x*coth(x/2) - 4``.
    """
    filepath = _websky_path('tsz.fits', download=download,
                            dlfunc='kszx.websky.read_tsz')
    print(f'Reading {filepath}\n', end='')
    y = healpy.read_map(filepath, dtype=np.float32)
    if freq_ghz is None:
        return y
    return y * np.float32(_T_CMB_K * 1.0e6 * _ysz_kernel(freq_ghz))


def read_isw(download=False):
    r"""Read the Websky late-time ISW map (``isw.fits``, HEALPix RING)."""
    filepath = _websky_path('isw.fits', download=download,
                            dlfunc='kszx.websky.read_isw')
    print(f'Reading {filepath}\n', end='')
    return healpy.read_map(filepath, dtype=np.float32)


def read_cib(freq_ghz, download=False):
    r"""Read a Websky CIB map at ``freq_ghz``, returning uK_CMB (HEALPix RING).

    The file is matched by exact frequency (rounded to 1 decimal place) against
    :data:`CIB_FREQS_GHZ`. Out-of-set frequencies raise ValueError.

    On-disk values are MJy/sr; the conversion to uK_CMB uses a delta-function
    bandpass at ``freq_ghz`` (analytic dB/dT at T_CMB=2.7255). This differs
    slightly from a real instrument bandpass (~2% at Planck frequencies).
    """
    relpath = _cib_filename(freq_ghz)
    filepath = _websky_path(relpath, download=download,
                            dlfunc='kszx.websky.read_cib')
    print(f'Reading {filepath}\n', end='')
    m = healpy.read_map(filepath, dtype=np.float32)
    # MJy/sr -> uK_CMB: divide by (Kcmb_MJy * 1e-6) = multiply by 1e6/Kcmb_MJy.
    factor = np.float32(1.0e6 / _kcmb_to_mjy_per_sr(freq_ghz))
    return m * factor


def _cib_filename(freq_ghz):
    """Resolve a freq (GHz) to the matching cib_NNNN.N.fits filename.

    Match by rounding the requested frequency to 1 decimal place and checking
    membership in CIB_FREQS_GHZ; no nearest-neighbor magic.
    """
    f = round(float(freq_ghz), 1)
    matched = [g for g in CIB_FREQS_GHZ if round(g, 1) == f]
    if not matched:
        raise ValueError(
            f'kszx.websky: freq_ghz={freq_ghz} not in CIB_FREQS_GHZ. '
            f'Available frequencies: {CIB_FREQS_GHZ}')
    g = matched[0]
    # Filename format: cib_NNNN.N.fits (4 digits before decimal, 1 after).
    return f'cib_{g:06.1f}.fits'


####################################################################################################
# CMB alms


def read_cmb_alm(lensed=True, seed=0, lmax=None, download=False):
    r"""Read Websky CMB alms in uK_CMB. Returns ``(alm_T, alm_E, alm_B)``.

      - ``lensed`` (bool): True for ``lensed_alm_seed{N}.fits``,
        False for ``unlensed_alm_seed{N}.fits``.
      - ``seed`` (int): 0 or 1.
      - ``lmax`` (int or None): if not None, truncate to this lmax.
      - ``download`` (bool): if True, fetch via scp.
    """
    if seed not in (0, 1):
        raise ValueError(f'kszx.websky.read_cmb_alm: seed must be 0 or 1, got {seed}')
    prefix = 'lensed' if lensed else 'unlensed'
    relpath = f'{prefix}_alm_seed{seed}.fits'
    filepath = _websky_path(relpath, download=download,
                            dlfunc='kszx.websky.read_cmb_alm')
    print(f'Reading {filepath}\n', end='')

    alm_T = healpy.read_alm(filepath, hdu=1)
    alm_E = healpy.read_alm(filepath, hdu=2)
    alm_B = healpy.read_alm(filepath, hdu=3)

    if lmax is not None:
        from . import healpix_utils
        alm_T = healpix_utils.degrade_alm(alm_T, lmax)
        alm_E = healpix_utils.degrade_alm(alm_E, lmax)
        alm_B = healpix_utils.degrade_alm(alm_B, lmax)

    return alm_T, alm_E, alm_B


####################################################################################################
# Bulk download


def download(*, halos=False, kappa=False, ksz=False, ksz_patchy=False,
             tsz=False, isw=False, cib=(), cmb_alm=()):
    r"""Pre-fetch Websky v0.4 data products from NERSC.

    All flags default to False; pass keywords to select what to download.

      - ``halos`` (bool): download halos.pksc (~32 GB)
      - ``kappa`` (bool): download kap.fits (full-z lensing convergence)
      - ``ksz`` (bool): download ksz.fits (late-time kSZ)
      - ``ksz_patchy`` (bool): download ksz_patchy.fits (reionization kSZ)
      - ``tsz`` (bool): download tsz.fits (nside=8192)
      - ``isw`` (bool): download isw.fits
      - ``cib`` (sequence of float): list of CIB frequencies in GHz
      - ``cmb_alm`` (sequence of str): subset of ('lensed','unlensed');
        each downloads both seeds (seed0 and seed1)
    """
    if halos:      _websky_path('halos.pksc',      download=True)
    if kappa:      _websky_path('kap.fits',        download=True)
    if ksz:        _websky_path('ksz.fits',        download=True)
    if ksz_patchy: _websky_path('ksz_patchy.fits', download=True)
    if tsz:        _websky_path('tsz.fits',        download=True)
    if isw:        _websky_path('isw.fits',        download=True)
    for f in cib:
        _websky_path(_cib_filename(f), download=True)
    for kind in cmb_alm:
        if kind not in ('lensed', 'unlensed'):
            raise ValueError(f"kszx.websky.download: cmb_alm entries must be 'lensed' or 'unlensed', got {kind!r}")
        for seed in (0, 1):
            _websky_path(f'{kind}_alm_seed{seed}.fits', download=True)
