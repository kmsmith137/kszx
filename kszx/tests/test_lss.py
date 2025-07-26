from .. import Box
from .. import core
from .. import utils
from . import helpers

import itertools
import numpy as np


####################################################################################################


class RandomPoly:
    def __init__(self, degree, lpos, rpos):
        assert lpos.shape == rpos.shape
        assert lpos.ndim == 1
        assert degree >= 1

        self.ndim = len(lpos)
        self.degree = degree
        self.lpos = lpos
        self.rpos = rpos
        self.cpos = (rpos + lpos) / 2.
        self.coeffs = np.zeros((self.ndim, degree+1))

        for axis in range(self.ndim):
            dx = (rpos[axis] - lpos[axis]) / 2.
            for i in range(degree+1):
                self.coeffs[axis,i] = np.random.uniform(-1.0, 1.0) / dx**i


    def _eval_axis(self, x, axis):
        t = x - self.cpos[axis]
        tpow = np.ones_like(t)
        ret = np.zeros_like(t)

        for i in range(self.degree+1):
            ret += self.coeffs[axis,i] * tpow
            tpow *= t

        return ret
    
            
    def eval_points(self, points):
        assert points.ndim == 2
        assert points.shape[-1] == self.ndim
        
        npoints = points.shape[0]
        ret = np.ones(npoints)

        for axis in range(self.ndim):
            ret *= self._eval_axis(points[:,axis], axis)

        return ret
        
        
    def eval_grid(self, shape, ishift):
        assert len(shape) == len(ishift) == self.ndim
        ret = np.ones(shape)

        for axis in range(self.ndim):
            x = np.linspace(self.lpos[axis], self.rpos[axis], shape[axis])
            x = np.roll(x, ishift[axis])
            y = self._eval_axis(x, axis)
            s = np.concatenate((np.ones(axis,dtype=int), (-1,), np.ones(self.ndim-axis-1,dtype=int)))
            ret *= np.reshape(y, s)

        return ret


####################################################################################################


def test_interpolation():
    print('test_interpolation(): start')

    for _ in range(100):
        # Currently, interpolate_points() supports CIC and cubic.
        kernel, degree = ('cic',1) if (np.random.uniform() < 0.5) else ('cubic',3)
        periodic = (np.random.uniform() < 0.5)
        
        # Currently, interpolate_points() only supports ndim=3.
        box = helpers.random_box(ndim=3, nmin=degree+1)
        ndim = box.ndim  # placeholder for future expansion
        
        ishift = np.random.randint(-1000,1000,size=ndim) if periodic else np.zeros(ndim,dtype=int)
        poly_lpos = box.lpos + ishift * box.pixsize
        poly_rpos = box.rpos + ishift * box.pixsize
        
        poly = RandomPoly(degree=degree, lpos=poly_lpos, rpos=poly_rpos)
        
        npoints = np.random.randint(100, 200)
        pad = (degree - 1 + 1.0e-7) * (box.pixsize/2.)
        points = np.random.uniform(poly_lpos + pad, poly_rpos - pad, size=(npoints,ndim))
        exact_vals = poly.eval_points(points)

        grid = poly.eval_grid(box.npix, ishift)
        interpolated_vals = core.interpolate_points(box, grid, points, kernel=kernel, periodic=periodic)

        epsilon = helpers.compare_arrays(exact_vals, interpolated_vals)
        # print(f'{epsilon=}')
        assert epsilon < 1.0e-12
        
    print('test_interpolation(): pass')


def test_interpolation_gridding_consistency():
    print('test_interpolation_gridding_consistency(): start')
    
    for _ in range(100):
        # Currently, interpolate_points() supports CIC and cubic.
        kernel, degree = ('cic',1) if (np.random.uniform() < 0.5) else ('cubic',3)
        periodic = (np.random.uniform() < 0.5)
        
        # Currently, interpolate_points() only supports ndim=3.
        box = helpers.random_box(ndim=3, nmin=degree+1)
        ndim = box.ndim  # placeholder for future expansion

        npoints = np.random.randint(100, 200)
        pad = (-1000 * box.pixsize) if periodic else ((degree - 1 + 1.0e-7) * (box.pixsize/2.))
        points = np.random.uniform(box.lpos+pad, box.rpos-pad, size=(npoints,ndim))

        g = np.random.normal(size=box.npix)  # random grid
        wscal = np.random.uniform(1.0, 2.0)  # random wscal

        # w = random weights, either 0-d or 1-d (passed to core.grid_points() below)
        # w1 = 1-d representation of 'w' (passed to np.dot() below)
        if np.random.uniform() < 0.5:
            w = w1 = np.random.normal(size=npoints)
        else:
            w = np.random.uniform(1.0, 2.0)
            w1 = np.full(npoints, w)
        
        Ag = wscal * core.interpolate_points(box, g, points, kernel=kernel, periodic=periodic)
        Aw = core.grid_points(box, points, weights=w, kernel=kernel, periodic=periodic, wscal=wscal)

        dot1 = np.dot(w1,Ag)
        dot2 = helpers.map_dot_product(box,Aw,g)
        
        den = np.dot(w1,w1) * np.dot(Ag,Ag)
        den += helpers.map_dot_product(box,g,g) * helpers.map_dot_product(box,Aw,Aw)

        epsilon = np.abs(dot1-dot2) / den**(0.5)
        assert epsilon < 1.0e-12
        
    print('test_interpolation_gridding_consistency(): pass')


####################################################################################################


# FIXME rename this
def test_simulate_gaussian():
    print('test_simulate_gaussian(): start')
    
    for _ in range(100):
        box = helpers.random_box()
        arr = core.simulate_white_noise(box, fourier=True)
        arr2 = core.fft_c2r(box, arr)
        arr2 = core.fft_r2c(box, arr2)
        eps = np.max(np.abs(arr-arr2)) *  np.sqrt(box.pixel_volume)
        assert eps < 1.0e-10
        
    print('test_simulate_gaussian(): pass')


# FIXME turn into proper unit test
def monte_carlo_simulate_gaussian(npix, pixsize):
    box = Box(npix, pixsize)
    p_real = np.zeros(box.fourier_space_shape)
    p_fourier = np.zeros(box.fourier_space_shape)
        
    for nmc in itertools.count():
        x = core.simulate_white_noise(box, fourier=False)
        x = core.fft_r2c(box, x)
        p_real += np.abs(x)**2

        y = core.simulate_white_noise(box, fourier=True)
        p_fourier += np.abs(y)**2

        if utils.is_perfect_square(nmc):
            r = p_fourier/p_real
            print(f'{nmc=} min_ratio={np.min(r)} max_ratio={np.max(r)}')
            
        
####################################################################################################


def _test_estimate_power_spectrum(box, kbin_edges, nmaps=None, use_dc=False):
    """Compares estimate_power_spectrum() to a reference implementation.
    
    Note that nmaps=1 is slightly different from nmaps=None. 
    (The pk arrays have shape (1,1,nkbins) versus (nkbins,) in the two cases.)

    In principle, there is a small chance that this test can fail, if estimate_power_spectrum() 
    and the reference implementation put a k-mode in different bins, due to roundoff error. 
    """

    M = nmaps if (nmaps is not None) else 1
    maps = np.array([ core.simulate_white_noise(box, fourier=True) for _ in range(M) ])
    
    map_arg = maps if (nmaps is not None) else maps[0]
    pk, bc = core.estimate_power_spectrum(box, map_arg, kbin_edges, use_dc=use_dc, allow_empty_bins=True, return_counts=True)
    
    nbins = len(kbin_edges) - 1
    ref_pk = np.zeros((M,M,nbins))
    ref_bc = np.zeros(nbins, dtype=int)

    for ix in helpers.generate_indices(box.fourier_space_shape):
        i = np.array(ix, dtype=int)
            
        if (not use_dc) and np.all(i==0):
            continue

        k = np.array(i, dtype=float)
        k = np.minimum(k, box.npix - k)
        k *= (2*np.pi) / box.boxsize
        k = np.dot(k,k)**0.5
        b = np.searchsorted(kbin_edges, k, side='right')-1
        
        if (b < 0) or (b >= nbins):
            continue

        t = 1 if ((ix[-1]==0) or (2*ix[-1] == box.npix[-1])) else 2
        z = maps[(slice(None),) + ix]
        assert z.shape == (M,)
        
        ref_pk[:,:,b] += t * np.outer(z,z.conj()).real
        ref_bc[b] += t

    for b in range(nbins):
        if ref_bc[b] > 0:
            ref_pk[:,:,b] /= (ref_bc[b] * box.box_volume)

    if nmaps is None:
        ref_pk = ref_pk[0,0,:]
    
    tol = 1.0e-10 * (bc + ref_bc)**0.5
    
    assert np.all(bc == ref_bc)
    assert np.all(np.abs(pk - ref_pk) <= tol)


def test_estimate_power_spectrum():
    """Compares estimate_power_spectrum() to a reference implementation.
    
    In principle, there is a small chance that this test can fail, if estimate_power_spectrum() 
    and the reference implementation put a k-mode in different bins, due to roundoff error. 
    """

    print('test_estimate_power_spectrum(): start')
    
    for iouter in range(100):
        box = helpers.random_box()
        kbin_edges = helpers.random_kbin_edges(box)
        use_dc = (np.random.uniform() < 0.5)

        # Note: C++ kernel calls different functions for M=1, M=2, M=3, M=4, and M>4.
        nmaps = np.random.randint(1,8) if (np.random.uniform() < 0.95) else None
        # print(f'{nmaps=}')

        _test_estimate_power_spectrum(box, kbin_edges, nmaps=nmaps, use_dc=use_dc)
        
    print('test_estimate_power_spectrum(): pass')


####################################################################################################


def _test_kbin_average(box, kbin_edges, use_dc=False):
    """Compares kbin_average() to a reference implementation.
    
    In principle, there is a small chance that this test can fail, if kbin_average() 
    and the reference implementation put a k-mode in different bins, due to roundoff error. 
    """

    # Random function f(k) = sum_i coeffs[i] * cos(rvals[i]*k)
    ncoeffs = 5
    rmax = 2.0 / np.min(box.kfund)
    rvals = np.random.uniform(low=0, high=rmax, size=ncoeffs)
    coeffs = np.random.normal(size=ncoeffs)

    def f(k):
        ret = np.zeros_like(k)
        for i in range(ncoeffs):
            ret += coeffs[i] * np.cos(rvals[i]*k)
        return ret

    fmean, bc = core.kbin_average(box, f, kbin_edges, use_dc=use_dc, allow_empty_bins=True, return_counts=True)

    nbins = len(kbin_edges) - 1
    ref_fk = f(box.get_k())
    ref_fmean = np.zeros(nbins)
    ref_bc = np.zeros(nbins, dtype=int)

    for ix in helpers.generate_indices(box.fourier_space_shape):
        i = np.array(ix, dtype=int)
            
        if (not use_dc) and np.all(i==0):
            continue

        k = np.array(i, dtype=float)
        k = np.minimum(k, box.npix - k)
        k *= (2*np.pi) / box.boxsize
        k = np.dot(k,k)**0.5
        b = np.searchsorted(kbin_edges, k, side='right')-1
        
        if (b < 0) or (b >= nbins):
            continue

        t = 1 if ((ix[-1]==0) or (2*ix[-1] == box.npix[-1])) else 2
        ref_fmean[b] += t * ref_fk[ix]
        ref_bc[b] += t

    for b in range(nbins):
        if ref_bc[b] > 0:
            ref_fmean[b] /= ref_bc[b]

    eps = np.max(np.abs(fmean - ref_fmean))
    # print(f'{eps=}')
    assert np.all(bc == ref_bc)
    assert eps < 1.0e-12
    

def test_kbin_average():
    """Compares kbin_average() to a reference implementation.
    
    In principle, there is a small chance that this test can fail, if kbin_average() 
    and the reference implementation put a k-mode in different bins, due to roundoff error. 
    """

    print('test_kbin_average(): start')
    
    for iouter in range(100):
        box = helpers.random_box()
        kbin_edges = helpers.random_kbin_edges(box)
        use_dc = (np.random.uniform() < 0.5)
        _test_kbin_average(box, kbin_edges, use_dc=use_dc)
        
    print('test_kbin_average(): pass')
