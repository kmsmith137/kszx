from .. import Box

import numpy as np

    
def compare_arrays(arr1, arr2):
    assert arr1.shape == arr2.shape
    assert arr1.dtype == arr2.dtype

    t = arr1-arr2
    num = np.vdot(t,t)
    den = np.vdot(arr1,arr1) + np.vdot(arr2,arr2)
    
    return np.sqrt(num/den) if (den > 0.0) else 0.0


def generate_indices(shape):
    if len(shape) == 1:
        for i in range(shape[0]):
            yield (i,)
    else:
        for t in generate_indices(shape[:-1]):
            for i in range(shape[-1]):
                yield t + (i,)


def random_shape(ndim=None, nmin=1):
    if ndim is None:
        ndim = np.random.randint(1,4)
    
    ret = np.zeros(ndim, dtype=int)
    nmax = int(10000 ** (1./ndim))
    assert nmax >= (nmin+1)
    
    for d in range(ndim):
        if np.random.uniform() < 0.1:
            ret[d] = nmin
        elif np.random.uniform() < 0.1:
            ret[d] = nmin+1
        else:
            ret[d] = np.random.randint(nmin, nmax+1)

    return ret


def random_box(ndim=None, nmin=2, avoid_small_r=False):
    npix = random_shape(ndim, nmin)
    pixsize = np.random.uniform(1.0, 10.0)

    while True:
        t = np.random.uniform(0, pixsize)
        cpos = np.random.uniform(-t*npix, t*npix, size=len(npix))
        box = Box(npix, pixsize, cpos)
        
        if avoid_small_r:
            # The 'avoid_small_r' arg is for tests which are sensitive to the direction
            # of {\hat r}, e.g. spin-l FFTs.
            rmin2 = sum(np.min(box.get_r_component(axis))**2 for axis in range(box.ndim))
            if rmin2 < 0.01 * pixsize**2:
                continue
            
        return box


def random_kbin_edges(box, nbins=None):
    if nbins is None:
        nbins = np.random.randint(2, 11)
    
    kmax = 1.1 * np.sqrt(box.ndim) * box.knyq
    kbin_edges = np.random.uniform(0.0, kmax, nbins+1)
    kbin_edges = np.sort(kbin_edges)
    kbin_edges += np.linspace(0.0, 1.0e-10 * kmax, nbins+1)
    kbin_edges[0] = kbin_edges[0] if (np.random.uniform() < 0.5) else 0.0
    
    return kbin_edges
