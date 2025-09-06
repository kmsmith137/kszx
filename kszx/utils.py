"""The ``kszx.utils`` module contains miscelleanous utilities, that didn't really fit in elsewhere."""

import os
import numpy as np
import scipy.special
import scipy.integrate
import scipy.interpolate
import multiprocessing

from . import cpp_kernels


####################################################################################################


_curr_nthreads = os.cpu_count()


def get_nthreads():
    """Returns default number of threads, for functions which are multithreaded (e.g. scipy.fft).

    Currently equal to ``os.cpu_count()``, except for worker tasks in a multiprocessing pool.
    In that case, we divide the number of threads by the number of workers in the pool.

    Note: for this to work, you should construct the pool with :func:`kszx.utils.Pool()`, 
    not ``multiprocessing.Pool()``!
    """
    
    return _curr_nthreads


def set_nthreads(nthreads, reseed_numpy_rng=False):
    """Sets the default number of threads. Not intended to be called directly -- use with caution!

    (The intended use case for this function is in :func:`kszx.utils.Pool()`, when initializing
    worker threads. In particular, the ``reseed_numpy_rng`` arg isn't really related to threads,
    but is included for convenience in kszx.utils.Pool().)
    """

    assert nthreads > 0
    assert nthreads <= os.cpu_count()
    
    global _curr_nthreads
    _curr_nthreads = nthreads

    cpp_kernels.omp_set_num_threads(nthreads)
    
    # FIXME more code needed here, to propagate new 'nthreads' to all libraries (e.g. blas, numba).
    
    if reseed_numpy_rng:
        # I think this is the best way to reseed numpy's global RNG.
        # Note that the global RNG is an MT19937, with 624 bytes of state.
        ss = np.random.SeedSequence(pool_size=8)
        np.random.seed(ss.generate_state(624))

    
def Pool(processes=None, reseed_numpy_rng=True):
    """Wrapper around multiprocessing.Pool(), which reseeds the numpy RNG and calls set_nthreads() in each worker.

    This function addresses some issues with multiprocessing.Pool:

       - By default, workers inherit a copy of the parent's global RNG state, so functions
         like ``np.random.normal()`` produce the same random numbers in each worker. This can
         be a huge problem -- e.g. a Pool which makes Monte Carlo sims will generate the same
         realization multiple times. We address this by re-seeding numpy's default RNG state
         in each worker.

       - Functions which use a thread pool (e.g. :func:`kszx.core.fft_r2c()`) should use fewer 
         threads when called from workers, in order to avoid oversubscription issues.

       - (Minor.) If the number of workers (``processes`` arg) is unspecified, then I'd prefer
         to use the number of CPU cores, not the value of ``os.cpu_count()`` (which is usually twice
         the number of cores, since hyperthreading is usually enabled).

    Usage is the same as ``multiprocessing.Pool()``::

      def square(i):
          return i*i

      # NOTE: I recommend calling apply_async() in a loop, not calling map_async()!
      # This is because map_async() "swallows" exceptions until the very end, whereas
      # apply_async() aborts computation as soon as an exception is encountered.
      
      with kszx.utils.Pool() as pool:
          f = [ pool.apply_async(square, (i,)) for i in range(100) ]
          f = [ x.get() for x in f ]
    
      print('Here is a list of the first 100 squares:', f)

    Function arguments:

       - ``processes`` (int): number of worker subprocesses. If None, then we use the number
         of CPU cores (which is usually half the value of os.cpu_count()).

       - ``reseed_numpy_rng`` (boolean): if True (the default), then numpy's global RNG will 
         be re-seeded in each worker thread. You should always do this, unless you're confident
         that the workers won't use the global RNG.
    """

    if processes is None:
        # FIXME assumes that hyperthreading is enabled, so that the number of threads
        # is twice the number of cores. Is there an easy way to check this? (Could do
        # it by reading and parsing /proc/cpuinfo, but that's a pain.)
        processes = max(get_nthreads() // 2, 1)
        
    assert processes > 0
    assert processes == int(processes)
    worker_nthreads = max(get_nthreads() // int(processes), 1)
    
    print(f'Creating multiprocessing pool with {processes} worker processes, and {worker_nthreads} threads/worker.')

    if reseed_numpy_rng:
        print('Global numpy RNG will be re-seeded in each worker process.')
    else:
        print('Warning: global numpy RNG is shared between worker process! (reseed_numpy_rng=False)')
    
    initargs = (worker_nthreads, reseed_numpy_rng)   # arguments to set_nthreads() called in each worker
    return multiprocessing.Pool(processes, initializer=set_nthreads, initargs=initargs)


####################################################################################################


def ra_dec_to_xyz(ra_deg, dec_deg, r=None):
    """Converts spherical polar coordinates (ra_deg, dec_deg) to cartesian coords (x,y,z).
    
    Function arguments:
    
       - ``ra_deg`` (array): RA in degrees (not radians!)
       - ``dec_deg`` (array): DEC in degrees (not radians!)
       - ``r`` (optional array): radial coordinates (units of length)

    Return value:

       - ``xyz`` (array): new array with a length-3 axis appended.
         (E.g. if ra_deg and dec_deg have shape (m,n), then xyz has shape (m,n,3)).

    The coordinate systems are related by::

        x = r * cos(dec) * cos(ra)
        y = r * cos(dec) * sin(ra)
        z = r * sin(dec)

    Note that the radial coordinate ``r`` must be specified as a comoving distance. If you
    have redshifts instead, you should call ``Cosmology.chi()``::

        cosmo = kszx.Cosmology('planck18+bao')
        xyz = kszx.utils.ra_dec_to_xyz(ra_deg, dec_deg, cosmo.chi(z=z))
    """
    
    ra_deg = np.asarray(ra_deg)
    dec_deg = np.asarray(dec_deg)
    assert ra_deg.shape == dec_deg.shape

    # FIXME memory-inefficent (uses too many temp arrays)
    
    ra = ra_deg * (np.pi/180.)
    dec = dec_deg * (np.pi/180.)
    cos_dec = np.cos(dec)
    
    ret = np.empty(ra_deg.shape + (3,), dtype=float)
    ret[...,0] = cos_dec * np.cos(ra)
    ret[...,1] = cos_dec * np.sin(ra)
    ret[...,2] = np.sin(dec)

    if r is not None:
        r = np.asarray(r)
        assert r.shape == ra_deg.shape
        ret *= np.reshape(r, r.shape + (1,))
    
    return ret


def xyz_to_ra_dec(xyz, return_r=False):
    """Converts cartesian coords (x,y,z) to spherical polar coordinates (ra_deg, dec_deg).

    Function arguments:
    
       - ``xyz`` (array): array whose last axis has length 3.
       - ``return_r`` (boolean): indicates whether radial coords are returned, see below.

    Return value:

       - ``ra_deg`` (array): RA in degrees (not radians!)
       - ``dec_deg`` (array): DEC in degrees (not radians!)
       - ``r`` (array): radial coordinates (in same units as ``xyz``)
       - If ``return_r`` is True, then the return value is a triple ``(ra_deg, dec_deg, r)``.
         If ``return_r`` is False, then the return value is a pair ``(ra_deg, dec_deg)``.

    The returned arrays have one lower dimension than ``xyz``. (E.g. if ``xyz`` has shape (m,n,3)
    then ``ra_deg``, ``dec_deg``, and ``r`` all have shape (m,n).)

    The coordinate systems are related by::

        x = r * cos(dec) * cos(ra)
        y = r * cos(dec) * sin(ra)
        z = r * sin(dec)
    """

    xyz = np.asarray(xyz)
    assert xyz.shape[-1] == 3

    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]

    ra_deg = np.arctan2(y, x)
    ra_deg *= (180./np.pi)
    
    dec_deg = np.arctan2(z, np.sqrt(x*x + y*y))
    dec_deg *= (180./np.pi)

    if return_r:
        r = np.sqrt(x*x + y*y + z*z)
        return ra_deg, dec_deg, r
    else:
        return ra_deg, dec_deg
    

####################################################################################################


def W_tophat(x):
    r"""Returns Fourier transform of a 3-d tophat W(x), where x=kR. Vectorized.
    
    W(x) is given by any of the equivalent forms:

    .. math::

       W(x) &= 3 (\sin(x) - x \cos(x)) / x^3 \\
            &= 3 j_1(x) / x \\
            &= j_0(x) + j_2(x)
    """

    # Timing showed that this implementation was fastest.
    mask = (x != 0)   # to regulate division by zero
    ret = 3 * scipy.special.spherical_jn(1,x) / np.where(mask,x,1.0)
    return np.where(mask, ret, 1.0)


####################################################################################################


def asarray(x, caller, arg, dtype=None, allow_none=False):
    if allow_none and (x is None):
        return x
    
    try:
        assert x is not None
        return np.asarray(x, dtype)
    except:
        pass

    s = f' with dtype {dtype}' if (dtype is not None) else ''
    raise RuntimeError(f"{caller}: couldn't convert {arg} to array{s} (value={x})")
    

def one_hot(shape, indices, dtype=float, value=1):
    """Returns 'one-hot' array (all entries are zero except one)."""
    
    ret = np.zeros(shape, dtype=dtype)
    ret[indices] = value
    return ret

    
def scattered_add(lhs, ix, rhs, normalize_sum = None):
    """Returns sum of RHS values (before applying 'normalize_sum').
    FIXME: Is there a numpy function that does this? If not, write a C++ kernel.
    """
    
    assert lhs.ndim == 1
    assert ix.ndim == 1

    n = len(ix)
    rhs = np.asarray(rhs) if (rhs is not None) else None

    # Split lhs into (scalar, vector) components (the vector can be None).
    if rhs is None:
        rs, rv = 1.0, None
    elif rhs.ndim == 0:
        rs, rv = float(rhs), None
    elif rhs.shape == (n,):
        rs, rv = 1.0, rhs
    else:
        raise RuntimeError("kszx.utils.scattered_add(): 'rhs' and 'ix' have inconsistent shapes")

    rsum = (rs*np.sum(rv)) if (rv is not None) else (rs*n)
    
    if normalize_sum:
        assert rsum > 0.0
        rs *= (normalize_sum / rsum)
        
    # FIXME slow non-vectorized loops, write C++ helper function at some point.
    # (I don't think there is a numpy function that helps.)

    if rv is None:
        for i in range(n):
            lhs[ix[i]] += rs
    elif rs == 1.0:
        for i in range(n):
            lhs[ix[i]] += rv[i]
    else:
        for i in range(n):
            lhs[ix[i]] += rs * rv[i]

    return rsum
            

####################################################################################################


def quad(f, xmin, xmax, epsabs=0.0, epsrel=1.0e-4, points=None):
    """Wrapper for scipy.integrate.quad()."""
    
    assert xmin < xmax

    if points is not None:
        # Remove points outside range (not sure if scipy does this automatically)
        points = [ x for x in points if (xmin < x < xmax) ]                                       
    
    return scipy.integrate.quad(f, xmin, xmax, epsabs=epsabs, epsrel=epsrel)[0]


def dblquad(f, xmin, xmax, ymin, ymax, *, epsabs=0.0, epsrel=1.0e-4):
    """
    Wrapper for scipy.integrate.dblquad().
      - The 'f' argument is f(x,y) [not f(y,x) as in scipy.integrate.dblquad]
      - The 'ymin' and 'ymax' arguments can either be floats or functions of x
    """

    ff = lambda x,y: f(y,x)  # swap
    return scipy.integrate.dblquad(f, xmin, xmax, ymin, ymax, epsabs=epsabs, epsrel=epsrel)[0]


def spline1d(xvec, yvec):
    """Wrapper for scipy.interpolate.InterpolatedUnivariateSpline()."""
    
    assert xvec.shape == yvec.shape
    assert xvec.ndim == yvec.ndim == 1
    return scipy.interpolate.InterpolatedUnivariateSpline(xvec, yvec)


def spline2d(xvec, yvec, zmat):
    """
    Wrapper for scipy.interpolate.RectBivariateSpline().
    
    Note that RectBivariateSpline.__call__(x,y) takes a 'grid' argument:

       grid=False  means "broadcast x,y to get a set of points"
       grid=True   means "take the Cartesian product of x,y to get a grid"

    Warning: grid=True is the default! (I usually want grid=False).
    """

    assert xvec.ndim == yvec.ndim == 1
    assert zmat.shape == (len(xvec), len(yvec))
    return scipy.interpolate.RectBivariateSpline(xvec, yvec, zmat)



def logspace(xmin, xmax, n=None, dlog=None):
    """
    Returns a 1-d array of values, uniformly log-spaced over the range (xmin, xmax).
    
    The spacing can be controlled by setting either the 'n' argument (number of points)
    or 'dlog' (largest allowed spacing in log(x)).
    """

    assert 0 < xmin < xmax
    assert (n is None) or (n >= 2)

    if (n is None) and (dlog is not None):
        n = int((np.log(xmax) - np.log(xmin)) / dlog) + 2
    elif (n is None) or (dlog is not None):
        raise RuntimeError("logspace: exactly one of 'n', 'dlog' should be None")

    ret = np.exp(np.linspace(np.log(xmin), np.log(xmax), n))
    ret[0] = xmin   # get rid of roundoff error
    ret[-1] = xmax  # get rid of roundoff error
    
    return ret


def log_uniform(xmin, xmax, size=None):
    assert 0 < xmin < xmax
    return np.exp(np.random.uniform(np.log(xmin), np.log(xmax), size=size))


def covscale(m):
    """Converts covariance matrix to correlation matrix (with 1's on the diagonal)."""
    
    m = np.array(m)
    assert m.ndim == 2
    assert m.shape[0] == m.shape[1]
    assert np.all(m.diagonal() > 0.0)

    x = 1.0 / np.sqrt(m.diagonal())
    return m * x.reshape((1,-1)) * x.reshape((-1,1))


def is_perfect_square(n):
    if n < 0:
        return False
    m = int(np.round(np.sqrt(n)))
    return n == m**2


def is_sorted(x):
    assert x.ndim == 1
    assert len(x) >= 2
    return np.all(x[:-1] < x[1:])
    

def trapezoid_weights(xmin, xmax, n):
    """Returns xvec, wvec."""

    assert xmin < xmax
    assert n >= 2
    
    xvec = np.linspace(xmin, xmax, n)
    wvec = np.full(n, (xmax-xmin)/(n-1))
    wvec[0] *= 0.5
    wvec[-1] *= 0.5

    return xvec, wvec


def array_slice(arr, axis, *args):
    assert 0 <= axis < arr.ndim
    t1 = (slice(None),) * axis
    t2 = (slice(*args),)
    return a[t1+t2]
        
        
def range_checked_index_operator(source_array, index_array, default_value=None, error_message=None):
    """Returns source_array[index_array], but substitutes 'default_value' if index is out of range."""
    
    assert source_array.ndim == 1
    assert len(source_array) > 0

    n = len(source_array)
    valid = np.logical_and(index_array >= 0, index_array < n)

    if default_value is not None:
        index_array = np.where(valid, index_array, 0)
        return np.where(valid, source_array[index_array], default_value)
    
    if np.all(valid):
        return source_array[index_array]

    if error_message is None:
        error_message = 'index out of range, and no default_value specfied'

    raise RuntimeError(error_message)


def contains_duplicates(l):
    return len(l) != len(set(l))


def boxcar_sum(a, m, normalize=False):
    """
    Given a 1-d array 'a' of length n, return a length-n array containing boxcar-summed values of a.
    Each boxcar consists of (2*m+1) array elements, or fewer near the edges of the array.

    E.g. if m=2, then

      ret[0] = a[0] + a[1] + a[2]
      ret[1] = a[0] + a[1] + a[2] + a[3]
      ret[2] = a[0] + a[1] + a[2] + a[3] + a[4]
      ret[3] = a[1] + a[2] + a[3] + a[4] + a[5]
        ...
      ret[n-4] = a[n-6] + a[n-5] + a[n-4] + a[n-3] + a[n-2]
      ret[n-3] = a[n-5] + a[n-4] + a[n-3] + a[n-2] + a[n-1]
      ret[n-2] = a[n-4] + a[n-3] + a[n-2] + a[n-1]
      ret[n-1] = a[n-3] + a[n-2] + a[n-1]
    
    If 'normalize' is True, then "boxcar means" are computed instead of "boxcar sums".
    """

    # conversion to float helps with float32 dynamic-range issues
    a = np.asarray(a, dtype=float)
    assert a.ndim == 1
    assert m >= 0

    if m == 0:
        return np.copy(a)
    
    n = len(a)
    assert n > 0
        
    if n <= m+1:
        t = np.mean(a) if normalize else np.sum(a)
        return np.full((n,), t, dtype=a.dtype)

    c = np.cumsum(a)
    ret = np.zeros(n, dtype=a.dtype)
    ret[:-m] = c[m:]
    ret[-m:] = c[-1]
    ret[(m+1):] -= c[:(-m-1)]
    
    if normalize:
        den = np.full((n,), 2*m+1)
        den[:m] -= np.arange(m,0,-1)
        den[-m:] -= np.arange(1,m+1)
        ret = ret / den

    return ret


def subtract_binned_means(data, x, nbins):
    """Averages a 1-d 'data' array in x-bins, and returns a copy of 'data' with binned means subtracted.
    If nbins==0, then returns a copy of 'data'."""

    assert nbins >= 0
    assert data.shape == x.shape

    if nbins == 0:
        return np.copy(data)
        
    # Compute bin edges
    bins = np.linspace(np.min(x), np.max(x), nbins + 1)
    
    # Digitize x values into bins
    bin_indices = np.digitize(x, bins) - 1  # subtract 1 to make bin_indices 0-based

    # Clip bin indices to ensure they're within range
    bin_indices = np.clip(bin_indices, 0, nbins - 1)

    # Compute sum and count per bin
    sum_per_bin = np.bincount(bin_indices, weights=data, minlength=nbins)
    count_per_bin = np.bincount(bin_indices, minlength=nbins)
    
    # Avoid division by zero
    count_per_bin = np.maximum(count_per_bin, 1)
    mean_per_bin = sum_per_bin / count_per_bin

    # Subtract binned means
    return data - mean_per_bin[bin_indices]


def random_rotation_matrix(N):
    """Returns a random N-by-N rotation matrix."""

    # Precise meaning of "random": drawn from Haar measure on O(N).
    # Algorithm: apply Gram-Schmidt process to a Gaussian random matrix.

    ret = np.random.normal(size=(N,N))
    
    for i in range(N):
        for j in range(i):
            c = np.dot(ret[i,:], ret[j,:])
            ret[i,:] -= c * ret[j,:]
        ret[i,:] /= np.sqrt(np.dot(ret[i,:],ret[i,:]))

    return ret


def contract_axis(arr, weights, axis):
    """'Contract a D-dimensional array with a 1-d weights array, returning a (D-1) dimensional array.

    Function arguments:

      - ``arr``: D-dimensional array of shape ``(n[0], n[1], ..., n[D-1])``.

      - ``weights`` 1-d array of length ``n[axis]``.

      - ``axis``: integer 0 <= axis < D.

    Returns a (D-1) dimensional array of shape ``(n[0], ..., n[axis-1], n[axis+1], ..., n[D-1])``,
    defined by "contracting" the weights array along the specified axis of 'arr'.
    """
    
    arr = np.asarray(arr)
    weights = np.asarray(weights)
    
    assert 0 <= axis < arr.ndim
    assert weights.shape == (arr.shape[axis],)

    # Transpose 'arr' so that 'axis' is last.
    perm = np.zeros(arr.ndim, dtype=int)
    perm[:axis] = np.arange(axis)
    perm[axis:-1] = np.arange(axis+1, arr.ndim)
    perm[-1] = axis
    arr = np.transpose(arr, perm)

    # Now we can just call np.dot().
    return np.dot(arr, weights)


