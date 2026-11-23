from . import helpers
from . import test_fft

from .. import mlhack
from .. import core

import numpy as np


def test_cartesian_to_spherical():
    print(f'test_cartesian_to_spherical(): start')

    for iouter in range(100):
        n = np.random.randint(10,20)
        points = np.random.normal(size=(n,3))
        
        v = np.random.normal(size=(3,n))
        vv = np.sum(v**2, axis=0)**0.5
        v /= vv[None,:]
        vv = np.sum(v**2, axis=0)

        w = mlhack.cartesian_to_spherical(points, v[0,:], v[1,:], v[2,:], epsilon=1.0e-10)
        w = np.array(w)  # shape (3,n)
        ww = np.sum(w**2, axis=0)
        assert np.all(np.abs(ww-1.0) < 1.0e-5)
    
    print(f'test_cartesian_to_spherical(): end')

    
def test_fft_c2v():
    print(f'test_fft_c2v(): start')
    
    for iouter in range(100):
        box = helpers.random_box(ndim=3, nmin=10)
        kernel, degree = ('cic',1) if (np.random.uniform() < 0.5) else ('cubic',3)
        
        npoints = np.random.randint(100, 200)

        # Hmmm, I thought that the mlhack.fft_c2v() radial component v_r
        # would agree precisely with core.interpolate_points(..., fft=True, spin=1)
        # but this is not quite true since multiplying by (1/r) does not commute
        # with interpolation. So, instead of using uniform-random points, we use
        # randomly selected grid points.
        
        # pad = (degree - 1 + 1.0e-7) * (box.pixsize/2.)
        # points = np.random.uniform(box.lpos + pad, box.rpos - pad, size=(npoints,3))

        ix = np.random.randint(degree, box.npix[0]-degree, size=npoints)
        iy = np.random.randint(degree, box.npix[1]-degree, size=npoints)
        iz = np.random.randint(degree, box.npix[2]-degree, size=npoints)
        x = box.lpos[0] + (ix * box.pixsize)
        y = box.lpos[1] + (iy * box.pixsize)
        z = box.lpos[2] + (iz * box.pixsize)
        points = np.copy(np.transpose([x,y,z]))
        
        f = core.simulate_white_noise(box, fourier=True)
        vr, v_theta, v_phi = mlhack.fft_c2v(box, f, points, kernel, periodic=False)
        vr2 = core.interpolate_points(box, f, points, kernel, fft=True, spin=1, periodic=False)

        epsilon = np.max(np.abs(vr-vr2))
        assert epsilon < 1.0e-10

    print(f'test_fft_c2v(): end')
    
