from .Box import Box

from . import core
from . import cpp_kernels
from . import utils

import numpy as np


def fft_c2v(box, arr, points, kernel, periodic=False, threads=None):
    r"""Returns radial and transverse components of velocity reconstruction, for KSZ/ML respectively.

    Given a Fourier-space map $V(k)$, compute the vector-valued real space map
    
    $$v_j(x) = \int_k i {\hat k}_j V(k) e^{ik\cdot x}$$

    Evaluate (via interpolation) at the specified points, convert from Cartesian to
    spherical polar coordinates, and return (v_r, v_theta, v_phi).

    Context: in KSZ/ML pipelines, V(k) is the filtered galaxy field (defined as something
    U(k) rho_g(k) in Selim's notes). The radial component v_r is the reconstructed radial
    velocity (for kSZ stacking), and the transverse components (v_theta, v_phi) are the
    reconstructed transverse velocity (for ML stacking).

    Before calling fft_c2v(), caller should ensure that all Fourier-space factors have been
    included in V(k). After calling fft_c2v(), caller may want to apply additional real-space
    factors (for example, faH/bg).
    
    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.

        - ``arr``: numpy array representing a fourier-space map (shape ``box.fourier_space_shape``,
          dtype complex).

        - ``points``: numpy array of shape (n,3), where v is to be interpolated.

        - ``kernel`` (string): either ``'cic'`` or ``'cubic'`` (more options will be defined later).

        - ``periodic`` (boolean): if True, then the box has periodic boundary conditions.

        - ``threads`` (integer or None): number of parallel threads used.
          If ``threads=None``, then number of threads defaults to :func:`~kszx.utils.get_nthreads()`.

    Returns a triple (v_r, v_theta, v_phi).

    FIXME: breaks down very close to north/south pole, or very close to the origin (where "very
    close" means "within $10^{-10}$ of a pixel", so very unlikely to arise by chance).
    """

    if not isinstance(box, Box):
        raise RuntimeError("kszx.mlhack.fft_c2v(): expected 'box' arg to be kszx.Box object, got {box = }")
    if kernel is None:
        raise RuntimeError("kszx.mlhack.fft_c2v(): 'kernel' arg must be specified")
    if box.ndim != 3:
        raise RuntimeError('kszx.mlhack.fft_c2v(): currently only ndim==3 is supported')

    arr = utils.asarray(arr, 'kszx.mlhack.fft_c2v()', 'arr')
    points = utils.asarray(points, 'kszx.mlhack.fft_c2v()', 'points', dtype=float)
    kernel = kernel.lower()

    if not box.is_fourier_space_map(arr):
        raise RuntimeError(f"kszx.mlhack.fft_c2v(): expected 'arr' to be a Fourier-space map")
    if (points.ndim != 2) or (points.shape[1] != 3):
        raise RuntimeError(f"kszx.mlhack.fft_c2v(): expected points.shape=(N,{box.ndim}), got shape {points.shape}")

    npoints = len(points)
    vcart = np.zeros((3, npoints))  # real-space v_j(x), in Cartesian coordinates evaluated at 'points'
    
    for axis in range(3):
        t = core.multiply_kfunc(box, arr, lambda k: 1./k, dc=0.)
        t *= 1j * box.get_k_component(axis, zero_nyquist=True)
        core.zero_nyquist_modes(box, t)

        k = box.get_k_component(axis)
        t = core.fft_c2r(box, t, spin=0, threads=threads)
        vcart[axis,:] = core.interpolate_points(box, t, points, kernel)
        del t

    vx, vy, vz = vcart
    return cartesian_to_spherical(points, vx, vy, vz, 1.0e-10 * box.pixsize)

    
def cartesian_to_spherical(points, vx, vy, vz, epsilon):
    """
    points: shape (n,3)
    vx, vy, vz: shape (n,)

    Returns (v_r, v_theta, v_phi).
    """

    points = utils.asarray(points, 'kszx.mlhack.cartesian_to_spherical()', 'points')
    vx = utils.asarray(vx, 'kszx.mlhack.cartesian_to_spherical()', 'vx')
    vy = utils.asarray(vy, 'kszx.mlhack.cartesian_to_spherical()', 'vy')
    vz = utils.asarray(vz, 'kszx.mlhack.cartesian_to_spherical()', 'vz')

    if (points.ndim != 2) or (points.shape[1] != 3):
        raise RuntimeError(f"kszx.mlhack.fft_c2v(): expected points.shape=(N,{box.ndim}), got shape {points.shape}")

    npoints = len(points)
    if any((t.shape != (npoints,)) for t in (vx,vy,vz)):
        raise RuntimeError(f"kszx.mlhack.fft_c2v(): expected u.shape=({npoints,}), got shape {t.shape}")
    
    x, y, z = np.transpose(points)

    r = np.sqrt(x*x + y*y + z*z)
    rinv = 1.0 / np.maximum(r, epsilon)

    rho = np.sqrt(x*x + y*y)
    rhoinv = 1.0 / np.maximum(rho, epsilon)

    xhat = x * rhoinv
    yhat = y * rhoinv
    cos_theta = z * rinv
    sin_theta = rho * rinv

    v_r = rinv * (x*vx + y*vy + z*vz)
    v_theta = (xhat * cos_theta * vx) + (yhat * cos_theta * vy) - (sin_theta * vz)
    v_phi = -(yhat * vx) + (xhat * vy)

    return (v_r, v_theta, v_phi)
