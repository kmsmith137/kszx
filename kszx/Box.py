import io    # StringIO
import functools
import numpy as np


class Box:
    def __init__(self, npix, pixsize, cpos=None):
        r"""The Box class represents a pixelization of an N-dimensional box.

        The Box class requires the caller to manually choose the box size and observer
        location. See :class:`~kszx.BoundingBox` for a subclass which automatically 
        chooses these parameters to match a galaxy survey.

        Constructor args:

            - npix (tuple or 1-d array): 
               Grid dimensions, e.g. (1024,1024,1024)
               For a non-cubic box, use unequal dimensions, e.g. (512,512,1024).

            - pixsize (float): 
               Pixel side length (caller-specified units, usually Mpc).
               Note that pixsize is a scalar, i.e. pixels are always cubic.

            - cpos (tuple, 1-d array, or None):
               Location of box center, in coordinate system with observer at origin.

               Defaults to cpos=(npix*pixsize)/2, i.e. observer at lower-left corner.
               For a box with observer at the center, use cpos=(0,0,0).

               The observer location matters in some situations (e.g. gridding a galaxy 
               catalog, where galaxy locations are specified in observer coordinates),
               but not others (e.g. taking an FFT).

        Maps and array shapes:

             Note that we define a Box class, but not a Map class.

             Instead, a real-space map is represented by a pair (box, arr), where 'arr' is a 
             numpy array with dtype=float. Similarly, a Fourier-space map is represented by a 
             pair (box, arr), where 'arr' is a numpy array with dtype=complex.

             The real-space and Fourier-space array shapes are related as follows:

        $$(\mbox{real-space shape}) = (n_0, n_1, \cdots, n_{d-1})$$
        $$(\mbox{Fourier-space shape})= (n_0, n_1, \cdots, \lfloor n_{d-1}/2 \rfloor + 1)$$

        Members:
        
           - ndim (integer): Number of dimensions (usually 3).
           - pixsize (float): Pixel side length in user-defined coordinates.
           - boxsize (length-ndim array): Box side lengths in user-defined coordinates (equal to npix * pixsize).
           - npix (length-ndim array): Real-space map shape, represented as numpy array.
           - real_space_shape (tuple): same as 'npix', but represented as tuple instead of 1-d array
           - nk (length-ndim array): Fourier-space map shape, represented as numpy array.
           - fourier_space_shape (tuple) same as 'nk', but represented as tuple instead of 1-d array
           - cpos (length-ndim array): location of box center, in observer coordinates
           - lpos (length-ndim array): location of lower left box corner, in observer coordinates
           - rpos (length-ndim array): location of upper right box corner, in observer coordinates
           - kfund (length-ndim array): Lowest frequency on each axis (equal to 2pi/boxsize)
           - knyq (float): Nyquist frequency (equal to pi/pixsize)
           - box_volume (float): Box volume (equal to prod(boxsize))
           - pixel_volume (float): Pixel volume (equal to pixsize^N)

        Fourier conventions:

        $$f(k) = V_{pix} \sum_x f(x) e^{-ik\cdot x}$$

        $$f(x) = V_{box}^{-1} \sum_k f(k) e^{ik\cdot x}$$

        $$\langle f(k) f(k')^* \rangle = V_{box} P(k) \delta_{k,k'}$$
        """
        
        npix = np.asarray(npix)
        assert npix.ndim == 1
        assert npix.dtype.kind in ['i','u']    # integer-valued
        assert np.all(npix > 0)

        ndim = len(npix)
        assert 1 <= ndim <= 3

        pixsize = float(pixsize)
        assert(pixsize > 0)
        
        if cpos is not None:
            cpos = np.array(cpos, dtype=float)
            assert cpos.shape == npix.shape

        # Real-space shape
        self.ndim = ndim
        self.npix = np.array(npix, dtype=int)
        self.real_space_shape = tuple(self.npix)

        # Fourier-space shape
        self.nk = np.copy(self.npix)
        self.nk[-1] = (self.nk[-1] // 2) + 1
        self.fourier_space_shape = tuple(self.nk)
 
        # Lengths / volumes / wavenumbers
        self.pixsize = pixsize
        self.boxsize = self.npix * self.pixsize
        self.kfund = 2*np.pi / self.boxsize
        self.knyq = np.pi / self.pixsize
        self.box_volume = np.prod(self.boxsize)
        self.pixel_volume = (self.pixsize)**ndim

        # Box location in observer coordinates.
        t = 0.5 * (self.npix - 1) * self.pixsize
        self.cpos = cpos if (cpos is not None) else (self.boxsize/2.)
        self.lpos = self.cpos - t
        self.rpos = self.cpos + t

        # Used internally in self.get_r(): which real-space pixel has the smallest r-coordinate?
        self._ix_smallest_r = -self.lpos / self.pixsize
        self._ix_smallest_r = np.array(np.round(self._ix_smallest_r), dtype=int)
        self._ix_smallest_r = np.maximum(self._ix_smallest_r, 0)
        self._ix_smallest_r = np.minimum(self._ix_smallest_r, self.npix-1)
        self._ix_smallest_r = tuple(self._ix_smallest_r)

        
    def is_real_space_map(self, arr):
        """Returns True if array 'arr' has the right shape/dtype for real-space map."""
        return isinstance(arr, np.ndarray) and np.all(arr.shape == self.real_space_shape) and (arr.dtype == float)

    
    def is_fourier_space_map(self, arr):
        """Returns True if array 'arr' has the right shape/dtype for Fourier-space map."""
        return isinstance(arr, np.ndarray) and np.all(arr.shape == self.fourier_space_shape) and (arr.dtype == complex)


    def get_k_component(self, axis, zero_nyquist=True, one_dimensional=False):
        """Returns array (k[0], k[1], ...) containing signed k-values along specified axis.

        Args:
            axis (integer): satisfies 0 <= axis < ndim.
            zero_nyquist (boolean): if True (recommended) zero will be returned at k_nyquist.
            one_dimensonal (booolean):
               - if True, the returned array will have shape (N,), where N = self.nk[axis].
               - if False, the returned array will have shape (1,...,1, N, 1,...1).

        Returns: real-valued array with shape described above.

        Notes:
            - k-values include the usual factor (2pi / boxsize).
            - Roughly half the k-values will be positive, and roughly half will be
              negative (except for the last axis, where negative values are not stored).
        """

        nd = self.ndim
        assert 0 <= axis < nd

        n = self.npix[axis]
        nk = self.nk[axis]
        ret = np.arange(nk, dtype=float)

        if axis != (self.ndim-1):
            m = n//2 + 1
            ret[m:] = np.arange(m-n, 0, dtype=float)

        if zero_nyquist and (n % 2) == 0:
            ret[n//2] = 0

        ret *= self.kfund[axis]
        
        if not one_dimensional:
            shape = (1,)*axis + (nk,) + (1,)*(nd-axis-1)
            ret = ret.reshape(shape)

        return ret


    def get_k(self, exponent=1, regulate=False):
        """Returns an N-dimensional array containing |k|^exponent values.

        Args:
            exponent (float): return |k|^exponent. If k<0, then regulate=True is implied.
            regulate (boolean): if True, then replace k=0 by k=(2pi/boxsize).

        Returns: real-valued array with shape (self.fourier_space_shape).
        """

        k2 = [ self.get_k_component(axis, zero_nyquist=False)**2 for axis in range(self.ndim) ]
        ret = functools.reduce(np.add, k2)
            
        if regulate or (exponent < 0):
            ret[(0,)*self.ndim] = np.min(self.kfund)**2

        if exponent != 2:
            ret **= (0.5*exponent)

        return ret


    def get_r_component(self, axis, one_dimensional=False):
        """Returns array (r[0], r[1], ...) containing signed observer coordinate along specified axis.

        Args:
            axis (integer): satisfies 0 <= axis < ndim.
            one_dimensonal (boolean):
               - if True, the returned array will have shape (N,), where N = self.npix[axis].
               - if False, the returned array will have shape (1,...,1, N, 1,...1).

        Returns: real-valued array with shape described above.
        
        Note: returned r-components are in "observer" coordinates, i.e. the observer is at
        the origin, and the box corners are given by self.lpos, self.rpos.
        """        
        
        assert 0 <= axis < self.ndim

        n = self.npix[axis]
        ret = np.arange(n,dtype=float) - 0.5*(n-1)
        ret = (ret * self.pixsize) + self.cpos[axis]

        if not one_dimensional:
            shape = (1,)*axis + (n,) + (1,)*(self.ndim-axis-1)
            ret = ret.reshape(shape)

        return ret


    def get_r(self, exponent=1, regulate=False, eps=1.0e-6):
        """Returns an N-dimensional array containing |r|^exponent values.

        Args:
            exponent (float): return |r|^exponent. If k<0, then regulate=True is implied.
            regulate (boolean): if True, then replace r = max(r,eps*pixsize).
            eps (float): only used if regulation is performed.

        Returns: real-valued array with shape (self.real_space_shape).
        
        Note: returned r-values are in "observer" coordinates, i.e. the observer is at
        the origin, and the box corners are given by self.lpos, self.rpos.
        """
        
        assert 0 < eps < 0.5

        r2 = [ self.get_r_component(axis)**2 for axis in range(self.ndim) ]
        ret = functools.reduce(np.add, r2)
        
        if regulate or (exponent < 0):
            i = self._ix_smallest_r
            ret[i] = max(ret[i], (eps * self.pixsize)**2)

        if exponent != 2:
            ret **= (0.5 * exponent)

        return ret


    def _print_box_members(self, f, end=','):
        """Helper function called by __str__(). (This factorization is convenient for the BoundingBox subclass.)"""        
        print(f'    npix = {self.npix},', file=f)
        print(f'    pixsize = {self.pixsize},', file=f)
        print(f'    boxsize = {self.boxsize},', file=f)
        print(f'    kfund = {self.kfund},', file=f)
        print(f'    knyq = {self.knyq},', file=f)
        print(f'    lpos = {self.lpos},', file=f)
        print(f'    cpos = {self.cpos},', file=f)
        print(f'    rpos = {self.rpos}{end}', file=f)
        
    def __str__(self):
        with io.StringIO() as f:
            print(f'Box(', file=f)
            self._print_box_members(f)
            print(f')', file=f)
            return f.getvalue()
    
    def __repr__(self):
        return str(self)
