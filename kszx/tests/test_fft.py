from .. import core
from .. import cpp_kernels
from . import helpers

import numpy as np
import scipy.special


####################################################################################################


def xli_tp(l, i, theta, phi):
    if i == 0:
        return np.sqrt(4*np.pi/(2*l+1)) * scipy.special.sph_harm_y(l, 0, theta, phi).real
    elif i % 2:
        return np.sqrt(8*np.pi/(2*l+1)) * scipy.special.sph_harm_y(l, (i+1)//2, theta, phi).real
    else:
        return np.sqrt(8*np.pi/(2*l+1)) * scipy.special.sph_harm_y(l, (i+1)//2, theta, phi).imag


def xli_xyz(l, i, x, y, z):
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    return xli_tp(l, i, theta, phi)


def xli_rs_box(l, i, box):
    xyz = [ box.get_r_component(axis) for axis in range(3) ]
    return xli_xyz(l, i, xyz[0], xyz[1], xyz[2])


def xli_fs_box(l, i, box):
    xyz = [ box.get_k_component(axis) for axis in range(3) ]
    xli = xli_xyz(l, i, xyz[0], xyz[1], xyz[2])
    xli[0,0,0] = 0.   # zero DC mode

    for axis in range(3):
        npix = box.real_space_shape[axis]
        nk = box.fourier_space_shape[axis]
        
        if (npix % 2) == 0:
            # zero Nyquist frequency (there must be a better way to do this)
            m = np.ones(nk, dtype=bool)
            m[npix//2] = False
            m = np.reshape(m, (1,)*axis + (nk,) + (1,)*(2-axis))
            xli *= m

    return xli


####################################################################################################


def test_xli():
    """Tests the identity sum_i X_{li}(v) X_{li}^*(w) = P_l(v.w)."""

    print('test_xli(): start')
    
    for _ in range(100):
        l = np.random.randint(10)
        v, w = np.random.normal(size=(2,3))
        
        accum = 0.0
        for i in range(2*l+1):
            xli_v = xli_xyz(l, i, v[0], v[1], v[2])
            xli_w = xli_xyz(l, i, w[0], w[1], w[2])
            accum += xli_v * xli_w
        
        mu = np.dot(v,w) / (np.dot(v,v) * np.dot(w,w))**0.5
        pl = scipy.special.legendre_p(l, mu)
        eps = np.abs(accum - pl)
        eps = float(eps[0])  # shape (1,) -> scalar
        
        # print(f'{eps=} {l=} {accum=} {pl=}')
        assert eps < 1.0e-13
    
    print('test_xli(): pass')


def test_multiply_xli_real_space():
    """Tests cpp_kernels.test_multiply_xli_real_space()."""
    
    print('test_multiply_xli_real_space(): start')
    
    for _ in range(100):
        box = helpers.random_box(ndim=3)
        l = np.random.randint(9)
        i = np.random.randint(2*l+1)
        
        src = np.random.normal(size = box.real_space_shape)
        dst1 = xli_rs_box(l,i,box) * src
        dst2 = np.zeros(box.real_space_shape)
        cpp_kernels.multiply_xli_real_space(dst2, src, l, i, box.lpos[0], box.lpos[1], box.lpos[2], box.pixsize)

        eps = np.max(np.abs(dst1-dst2))
        print(f'{eps=} {l=} {i=} {box.npix=}')
        assert float(eps) < 1.0e-13
        
    print('test_multiply_xli_real_space(): pass')


def test_multiply_xli_fourier_space():
    """Tests cpp_kernels.test_multiply_xli_fourier_space()."""
    
    print('test_multiply_xli_fourier_space(): start')
    
    for _ in range(100):
        box = helpers.random_box(ndim=3)
        l = np.random.randint(9)
        i = np.random.randint(2*l+1)

        fs = box.fourier_space_shape
        src = np.random.normal(size=fs) + 1j*np.random.normal(size=fs)
        dst1 = xli_fs_box(l,i,box) * src
        dst2 = np.zeros(fs, dtype=complex)
        cpp_kernels.multiply_xli_fourier_space(dst2, src, l, i, box.npix[2])

        eps = np.max(np.abs(dst1-dst2))
        print(f'{eps=} {l=} {i=} {box.npix=}')
        assert float(eps) < 1.0e-13
        
    print('test_multiply_xli_fourier_space(): pass')
