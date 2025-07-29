from .. import core
from .. import cpp_kernels
from .. import Box
from .. import utils
from . import helpers

import numpy as np
import scipy.special


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
    
    if l > 0:
        core.zero_nyquist_modes(box, xli, zero_dc=True)
    
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
        box = helpers.random_box(ndim=3, nmin=3, avoid_small_r=True)
        l = np.random.randint(9)
        i = np.random.randint(2*l+1)
        coeff = np.random.uniform()
        accum = (np.random.uniform() < 0.5)
        
        src = np.random.normal(size = box.real_space_shape)
        dst1 = np.random.normal(size = box.real_space_shape)
        dst2 = (coeff * xli_rs_box(l,i,box) * src) + (dst1 if accum else 0)
        cpp_kernels.multiply_xli_real_space(dst1, src, l, i, box.lpos[0], box.lpos[1], box.lpos[2], box.pixsize, coeff, accum)

        eps = np.max(np.abs(dst1-dst2))
        # print(f'{eps=} {l=} {i=} {box.npix=}')
        assert float(eps) < 1.0e-13
        
    print('test_multiply_xli_real_space(): pass')


def test_multiply_xli_fourier_space():
    """Tests cpp_kernels.test_multiply_xli_fourier_space()."""
    
    print('test_multiply_xli_fourier_space(): start')
    
    for _ in range(100):
        box = helpers.random_box(ndim=3, nmin=3)
        l = np.random.randint(9)
        i = np.random.randint(2*l+1)
        coeff = np.random.uniform() * (1j if (l % 2) else 1+0j)
        accum = (np.random.uniform() < 0.5)

        fs = box.fourier_space_shape
        src = np.random.normal(size=fs) + 1j*np.random.normal(size=fs)
        dst1 = np.random.normal(size=fs) + 1j*np.random.normal(size=fs)
        dst2 = (coeff * xli_fs_box(l,i,box) * src) + (dst1 if accum else 0)
        cpp_kernels.multiply_xli_fourier_space(dst1, src, l, i, box.npix[2], coeff, accum)
                    
        eps = np.max(np.abs(dst1-dst2))
        # print(f'{eps=} {l=} {i=} {box.npix=}')
        assert float(eps) < 1.0e-13
        
    print('test_multiply_xli_fourier_space(): pass')


####################################################################################################


def test_fft_inverses():
    """Tests that fft_r2c() and fft_c2r() are inverses (spin-0 only)."""

    print('test_fft_inverses(): start')
    
    for iouter in range(100):
        box = helpers.random_box()
        x = core.simulate_white_noise(box, fourier=False)
        Fx = core.fft_r2c(box, x)
        FFx = core.fft_c2r(box, Fx)
        eps = np.max(np.abs(x-FFx)) * np.sqrt(box.pixel_volume)
        assert eps < 1.0e-13
        
    print('test_fft_inverses(): pass')


def test_fft_transposes():
    """Tests that fft_r2c() and fft_c2r() are transposes (for arbitrary spin)."""
    
    print('test_fft_transposes(): start')

    for iouter in range(100):
        spin = np.random.randint(1,9) if (np.random.uniform() < 0.5) else 0
        ndim, nmin = (3, 3) if (spin > 0) else (None, 2)
        box = helpers.random_box(ndim=ndim, nmin=nmin, avoid_small_r=True)
        # print(f'{spin=}')
        # print(box)

        x = core.simulate_white_noise(box, fourier=False)
        y = core.simulate_white_noise(box, fourier=True)
        Fx = core.fft_r2c(box, x, spin=spin)
        Fy = core.fft_c2r(box, y, spin=spin)

        mdot = lambda v,w: helpers.map_dot_product(box,v,w)
        dot1 = mdot(y,Fx)
        dot2 = mdot(Fy,x)
        den = mdot(y,y)*mdot(Fx,Fx) + mdot(Fx,Fx)*mdot(y,y)
        epsilon = np.abs(dot1-dot2) / den**0.5
        # print(f'{dot1=} {dot2=}')
        # print(f'{epsilon=}')
        assert epsilon < 1.0e-12
        
    print('test_fft_transposes(): pass')
    

####################################################################################################


def old_fft_r2c_spin1(box, arr, zero_nyquist=True):
    """Old implementation of spin-1 FFT. As a unit test, we compare to the new spin-l implementation."""
    
    ret = np.zeros(box.fourier_space_shape, dtype=complex)
    
    for axis in range(box.ndim):
        t = core.multiply_rfunc(box, arr, lambda r: 1./r, regulate=True)
        t *= box.get_r_component(axis)
        t = core.fft_r2c(box, t, spin=0)
        t *= -1j * box.get_k_component(axis, zero_nyquist=True)  # note minus sign here
        ret += t
        del t

    core.multiply_kfunc(box, ret, lambda k: 1./k, dc=0, in_place=True)

    # The old spin-1 code didn't do this, but the new spin-l code does.
    if zero_nyquist:
        core.zero_nyquist_modes(box, ret)
    
    return ret


def old_fft_c2r_spin1(box, arr, zero_nyquist=True):
    """Old implementation of spin-1 FFT. As a unit test, we compare to the new spin-l implementation."""

    ret = np.zeros(box.real_space_shape, dtype=float)

    for axis in range(box.ndim):
        t = core.multiply_kfunc(box, arr, lambda k: 1./k, dc=0.)
        t *= 1j * box.get_k_component(axis, zero_nyquist=True)

        # The old spin-1 code didn't do this, but the new spin-l code does.
        if zero_nyquist:
            core.zero_nyquist_modes(box, t)
            
        t = core.fft_c2r(box, t, spin=0)
        t *= box.get_r_component(axis)
        ret += t
        del t

    core.multiply_rfunc(box, ret, lambda r: 1./r, regulate=True, in_place=True)
    return ret


def reference_fft_r2c_spin2(box, arr):
    """I got super paranoid and implemented a spin-2 r2c FFT by hand."""

    ret = np.zeros(box.fourier_space_shape, dtype=complex)
    
    for i in range(box.ndim):
        for j in range(box.ndim):
            t = core.multiply_rfunc(box, arr, lambda r: 1./r**2, regulate=True)
            t *= box.get_r_component(i)
            t *= box.get_r_component(j)
            t = core.fft_r2c(box, t, spin=0)
            t *= box.get_k_component(i, zero_nyquist=True)
            t *= box.get_k_component(j, zero_nyquist=True)
            ret += (1.5) * t
            del t

    core.multiply_kfunc(box, ret, lambda k: 1./k**2, dc=0, in_place=True)
    ret -= 0.5 * core.fft_r2c(box, arr, spin=0)
    
    core.zero_nyquist_modes(box, ret, zero_dc=True)
    return ret
    

def test_spin_12_ffts():
    """Compares spin-l FFTs to reference implementations of spin-1 and spin-2."""
    
    print('test_spin_12_ffts(): start')
     
    for _ in range(100):
        box = helpers.random_box(ndim=3, nmin=3, avoid_small_r=True)

        # Part 1: spin-1 r2c
        
        src = core.simulate_white_noise(box, fourier=False)
        dst1 = core.fft_r2c(box, src, spin=1)
        dst2 = old_fft_r2c_spin1(box, src)
        
        num = np.max(np.abs(dst1-dst2))
        den = np.max(np.abs(dst1)) + np.max(np.abs(dst2))
        eps = num/den
        assert eps < 1.0e-12

        # Part 2: spin-1 c2r
        
        src = core.simulate_white_noise(box, fourier=True)
        dst1 = core.fft_c2r(box, src, spin=1)
        dst2 = old_fft_c2r_spin1(box, src)
        
        num = np.max(np.abs(dst1-dst2))
        den = np.max(np.abs(dst1)) + np.max(np.abs(dst2))
        eps = num/den
        assert eps < 1.0e-12

        # Part 3: spin-2 r2c
        
        src = core.simulate_white_noise(box, fourier=False)
        dst1 = core.fft_r2c(box, src, spin=2)
        dst2 = reference_fft_r2c_spin2(box, src)
        
        num = np.max(np.abs(dst1-dst2))
        den = np.max(np.abs(dst1)) + np.max(np.abs(dst2))
        eps = num/den
        assert eps < 1.0e-12
        
    print('test_spin_12_ffts(): pass')
