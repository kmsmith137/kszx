import os
import healpy
import scipy.fft
import pixell.enmap
import numpy as np

from ..Box import Box
from .. import cpp_kernels
from .. import utils
from . import numba_utils    



####################################################################################################


def _to_float(x, errmsg):
    try:
        return float(x)
    except:
        raise RuntimeError(errmsg)


def _sqrt_pk(box, pk, regulate):
    """Helper for simulate_gaussian_field()."""

    if callable(pk):
        if box.npix.shape == (3,) and np.all(box.npix == box.npix[0]):
            k = numba_utils.get_k_3D_box(dim = box.npix[0],L = box.pixsize*box.npix[0])
        else:
            k = box.get_k(regulate=regulate)
        pk = pk(k)
        
        if pk.shape != box.fourier_space_shape:
            raise RuntimeError('kszx.simulate_gaussian_field(): function pk() returned unexpected shape')
        if pk.dtype != float:
            raise RuntimeError('kszx.simulate_gaussian_field(): function pk() returned dtype={pk.dtype} (expected float)')
        if np.min(pk) < 0:
            raise RuntimeError('kszx.simulate_gaussian_field(): function pk() returned negative values')

        del k
        return numba_utils._sqrt(pk)   # returns sqrt(P(k))

    pk = _to_float(pk, 'kszx.simulate_gaussian_field(): expected pk argument to be either callable, or a real scalar')

    if pk < 0:
        raise RuntimeError('kszx.simulate_gaussian_field(): expected scalar pk argument to be non-negative')
    
    return np.sqrt(pk)

    
def simulate_white_noise(box, *, fourier):
    r"""Simulate white noise, in either real space or Fourier space, normalized to $P(k)=1$.

    Intended as a helper for ``simulate_gaussian_field()``, but may be useful on its own.

    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.

        - ``fourier`` (boolean): determines whether output is real-space or Fourier-space.
    
    Return value: 

      - A numpy array representing a real-space (``fourier=False``) or Fourier-space (``fourier=True``) map.

        The real-space and Fourier-space array shapes are given by ``box.real_space_shape``
        and ``box.fourier_space_shape``, and are related as follows:

        $$(\mbox{real-space shape}) &= (n_0, n_1, \cdots, n_{d-1}) \\
        (\mbox{Fourier-space shape}) &= (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$

    Note: our normalization conventions for the simulated field are (in Fourier and real space):
    
    $$\langle f(k) f(k')^* \rangle = V_{\rm box} \delta_{kk'}$$
    $$\langle f(x) f(x') \rangle = V_{\rm pix}^{-1} \delta_{xx'}$$
    """

    if not fourier:
        rms = 1.0 / np.sqrt(box.pixel_volume)
        return numba_utils.random.normal(size=box.real_space_shape, scale=rms)
        
    # Simulate white noise in Fourier space.
    nd = box.ndim
    rms = np.sqrt(0.5 * box.box_volume)
    ret = np.empty(box.fourier_space_shape, dtype=np.complex128)        
    ret.real = numba_utils.random_normal(size=box.fourier_space_shape, scale=rms)
    ret.imag = numba_utils.random_normal(size=box.fourier_space_shape, scale=rms)

    # The rest of this function imposes the reality condition f(-k) = f(k)^*.
    
    # t = modes where k_{nd-1} is self-conjugate
    n = box.npix[nd-1]
    s1 = (slice(None),) * (nd-1)
    s2 = slice(0,1) if (n % 2) else slice(0, (n//2)+1, (n//2))
    tview = ret[s1+(s2,)] 
    tcopy = np.conj(tview)   # copy and complex conjugate

    # Apply parity operation k -> (-k) to 'tcopy'.
    for axis in range(nd-1):
        n = box.npix[axis]
        s1 = (slice(None),) * axis
        s2fwd = (slice(1,n),)
        s2rev = (slice(n-1,0,-1),)
        s3 = (slice(None),) * (nd-axis-1)
        u = np.copy(tcopy[s1+s2rev+s3])
        tcopy[s1+s2fwd+s3] = u

    # Replace f(k) by (f(k) - f(-k)^*) / sqrt(2)
    tview += tcopy
    tview *= np.sqrt(0.5)   # preserve variance
    return ret


def simulate_gaussian_field(box, pk, pk0=None):
    r"""Simulates a Gaussian field (in Fourier space) with specified power spectrum P(k).

    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.

        - ``pk`` (function or scalar): The power spectrum, represented as a function $k \rightarrow P(k)$.
          If the power spectrum is constant in $k$, then a scalar can be used instead of a function.

        - ``pk0`` (scalar or None): This optional argument is intended to regulate cases
          where $\lim_{k\rightarrow 0} P(k) = \infty$. If ``pk0`` is specified, then ``pk()`` is
          not evaluated at k=0, and the value of ``pk0`` is used instead of ``Pk(0)``.
    
    Return value: 

         - A numpy array representing a Fourier-space map. (Array shape is given by
           ``box.fourier_space_shape``, and dtype is complex, see note below.)

    Notes:

       - The normalization of the simulated field is:

         $$\langle f(k) f(k')^* \rangle = V_{\rm box} P(k) \delta_{kk'}$$
    
       - The function ``pk()`` must be vectorized: its argument 'k' will be a 3-dimensional arary,
         and the return value should be a real-valued array with the same shape.
    
       - k-values passed to ``pk()`` will be in "physical" units, i.e. the factor ``(2*pi / box.boxsize)``
         is included.

       - The returned array is a Fourier-space map.
    
         Reminder: real-space and Fourier-space array shapes are given by ``box.real_space_shape``
         and ``box.fourier_space_shape``, and are related as follows:

         $$(\mbox{real-space shape}) &= (n_0, n_1, \cdots, n_{d-1}) \\
         (\mbox{Fourier-space shape}) &= (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$
    """

    assert isinstance(box, Box)

    sqrt_pk = _sqrt_pk(box, pk, regulate = (pk0 is not None))
    ret = simulate_white_noise(box, fourier=True)

    dc = ret[(0,)*box.ndim]   # must precede multiplying by sqrt_pk
    
    numba_utils.multiply_inplace(ret,sqrt_pk)

    if pk0 is not None:
        pk0 = _to_float(pk0, 'kszx.simulate_gaussian_field(): expected pk0 argument to be a real scalar')
        if pk0 < 0:
            raise RuntimeError('kszx.simulate_gaussian_field(): expected pk0 argument to be non-negative')
        ret[(0,)*box.ndim] = np.sqrt(pk0) * dc
    
    return ret


def apply_kernel_compensation(box, arr, kernel, exponent=-0.5):
    r"""Modifies Fourier-space map 'arr' in-place, to debias interpolation/gridding.

    Context: gridding kernels (see :func:`~kszx.grid_points()`) multiplicatively bias 
    power spectrum estimation,

    $$<P(k)>_{\rm estimated} = C(k) \, P(k)_{true}$$

    Here, $C(k)$ is a "compensation factor" satisfying $0 \le C(k) \le 1$ which depends 
    on both the magnitude and orientation of $k$.

    There is a similar bias which pertains to interpolation kernels, rather than gridding
    kernels (see :func:`~kszx.interpolate_points()`). Suppose we start with a Fourier-space
    map $f(k)$, then Fourier transform and interpolate at random locations. One would
    expect that an interpolated value $f_{\rm interp}$ has variance

    $$\langle f_{\rm interp}^2 \rangle = \int \frac{d^3k}{(2\pi)^3} \, f(k)^2$$

    However, the interpolation kernel produces a bias: the actual variance is

    $$\langle f_{\rm interp}^2 \rangle = \int \frac{d^3k}{(2\pi)^3} \, C(k) f(k)^2$$

    The function ``apply_kernel_compensation`` multiplies Fourier-space map ``arr``
    in-place by $C(k)^p$, where $p$ is the ``exponent`` argument.  Here are two 
    common applications:

      1. Before calling :func:`~kszx.estimate_power_spectrum()` on one or more Fourier-space 
         maps, you should call ``apply_kernel_compensation()`` on each map, to multiply by
         $C(k)^{-1/2}$. This will mitigate the power spectrum estimation bias noted above.

      2. Before calling :func:`~kszx.interpolate_points()` on a map, you should call
         ``apply_kernel_compensation()`` on the map, to multiply by $C(k)^{-1/2}$. 
         This will mitigate the interpolation bias noted above. (This assumes that 
         you start with the map in Fourier space, and FFT before interpolating.)
    
    Function args:

        - ``box`` (kszx.Box): defines pixel size, bounding box size, and location of observer.
          See :class:`~kszx.Box` for more info.

        - ``arr``: numpy array representing a Fourier-space map. The array shape should be given by
          ``box.fourier_space_shape`` and the dtype should be ``complex``, see note below.

        - ``kernel`` (string): either ``'cic'`` or ``'cubic'`` (more options will be defined later).

        - ``exponent`` (float): array will be multiplied by ``C(k)**exponent``. (The default value
          is ``exponent = -0.5``, since this value arises in both applications above.)
    
    Return value: None (the ``arr`` argument is modified in-place, by multiplying by ``C(k)**exponent``).

    Reminder: real-space and Fourier-space array shapes are given by ``box.real_space_shape``
    and ``box.fourier_space_shape``, and are related as follows:

    $$(\mbox{real-space shape}) &= (n_0, n_1, \cdots, n_{d-1}) \\
    (\mbox{Fourier-space shape}) &= (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$
    """

    # See tex notes. The variable 's' is sin(k*L/2)
    if kernel == 'cic':
        f = numba_utils._cic_ker
    elif kernel == 'cubic':
        f = numba_utils._cubic_ker
    else:
        raise RuntimeError(f'kszx.gridding_pk_multiplier(): {kernel=} is not supported')

    #assert isinstance(box, Box)
    assert box.is_fourier_space_map(arr)  # check shape and type of input array
    numba_utils._apply_kernel_compensation(arr, f, exponent)


    

