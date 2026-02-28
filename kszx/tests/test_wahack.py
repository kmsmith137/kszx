from . import helpers
from .. import wahack
from .. import core
from .. import cpp_kernels
from .. import utils

import numpy as np
import scipy.special


def test_flatten_real():
    print('test_flatten_real(): start')

    for _ in range(100):
        box = helpers.random_box(ndim=3)
        n = np.prod(box.npix)

        # Test that flatten -> unflatten are inverses
        arr = core.simulate_white_noise(box, fourier=False)
        arr2 = wahack.flatten(box, arr)
        arr2 = wahack.unflatten(box, arr2, fourier=False)
        eps = helpers.compare_arrays(arr, arr2)
        assert eps < 1.0e-10
        
        # unflatten -> flatten are inverses
        arr = np.random.normal(size=n)
        arr2 = wahack.unflatten(box, arr, fourier=False)
        arr3 = wahack.flatten(box, arr2)
        eps = helpers.compare_arrays(arr, arr3)
        assert eps < 1.0e-10

        # Test that under flatten/unflatten, map_dot_product(..., normalize=True)
        # corresponds to ordinary np.dot().
        
        dot1 = np.dot(arr, arr)
        dot2 = core.map_dot_product(box, arr2, arr2)
        eps = np.abs(dot1-dot2) / (np.abs(dot1) + np.abs(dot2))
        assert eps < 1.0e-10


    print('test_flatten_real(): pass')


def test_flatten_fourier():
    print('test_flatten_fourier(): start')

    for _ in range(100):
        box = helpers.random_box(ndim=3)
        n = np.prod(box.npix)

        # flatten -> unflatten are inverses
        arr = core.simulate_white_noise(box, fourier=True)
        arr2 = wahack.flatten(box, arr)
        assert arr2.shape == (n,)
        assert arr2.dtype == float
        arr2 = wahack.unflatten(box, arr2, fourier=True)
        eps = helpers.compare_arrays(arr, arr2)
        assert eps < 1.0e-10

        # unflatten -> flatten are inverses
        arr = np.random.normal(size=n)
        arr2 = wahack.unflatten(box, arr, fourier=True)
        arr3 = wahack.flatten(box, arr2)
        eps = helpers.compare_arrays(arr, arr3)
        assert eps < 1.0e-10

        # Test that under flatten/unflatten, map_dot_product(..., normalize=True)
        # corresponds to ordinary np.dot().
        
        dot1 = np.dot(arr, arr)
        dot2 = core.map_dot_product(box, arr2, arr2)
        eps = np.abs(dot1-dot2) / (np.abs(dot1) + np.abs(dot2))
        assert eps < 1.0e-10

    print('test_flatten_fourier(): pass')


def multiply_ylm_real_space(box, arr, l, m):
    """Multiply real-space map 'arr' by Y_{lm}(\\hat x), returning (re, im).

    Here arr, re, im are real-valued real-space maps, and
    arr * Y_{lm}(\\hat x) = re + i*im.
    """

    lp = (box.lpos[0], box.lpos[1], box.lpos[2], box.pixsize)
    re = np.empty(box.real_space_shape, dtype=float)
    im = np.empty(box.real_space_shape, dtype=float)

    if m == 0:
        # Y_{l0} = y_{l0}, which is real.
        coeff = np.sqrt((2*l+1) / (4*np.pi))
        cpp_kernels.multiply_xli_real_space(re, arr, l, 0, *lp, coeff, False)
        im = np.zeros(box.real_space_shape, dtype=float)
    elif m > 0:
        # Y_{lm} = y_{l,2m-1} + i*y_{l,2m}
        coeff = np.sqrt((2*l+1) / (8*np.pi))
        cpp_kernels.multiply_xli_real_space(re, arr, l, 2*m-1, *lp, coeff, False)
        cpp_kernels.multiply_xli_real_space(im, arr, l, 2*m, *lp, coeff, False)
    else:  # m < 0
        # Y_{l,-am} = (-1)^am * (y_{l,2am-1} - i*y_{l,2am})
        am = abs(m)
        sign = (-1)**am
        coeff = sign * np.sqrt((2*l+1) / (8*np.pi))
        cpp_kernels.multiply_xli_real_space(re, arr, l, 2*am-1, *lp, coeff, False)
        cpp_kernels.multiply_xli_real_space(im, arr, l, 2*am, *lp, -coeff, False)

    return re, im


def translate_cyclic(arr, s):
    """Cyclically translate an N-dimensional array by integer shift vector s.

    Returns an array 'out' satisfying:
        out[i_0, ..., i_{N-1}] = arr[(i_0 + s_0) % n_0, ..., (i_{N-1} + s_{N-1}) % n_{N-1}]
    """

    arr = np.asarray(arr)
    s = np.asarray(s)

    if arr.ndim == 0:
        raise ValueError('translate_cyclic: arr must have ndim >= 1')
    if s.shape != (arr.ndim,):
        raise ValueError(f'translate_cyclic: expected s.shape=({arr.ndim},), got s.shape={s.shape}')
    if not np.issubdtype(s.dtype, np.integer):
        raise TypeError(f'translate_cyclic: s must be integer-valued, got dtype={s.dtype}')

    ret = arr
    for axis in range(arr.ndim):
        ret = np.roll(ret, -int(s[axis]), axis=axis)

    return ret


def Q_slow(box, f1, f2, l1, l2, l3, s):
    """
    Computes Q_{L1,L2,L3}(s) using the "real-space" definition.

    f1, f2 are real-space maps (f_i = W_i * F_i in notation from the paper).

    s is a length-3 integer-valued displacement vector. Displacements are "cyclic"
    relative to the box dimensions. Throw an exception if s=0.

    Returns a complex scalar. (Note that Q is secretly real, but checking that
    the imaginary part is zero is a useful code test.)
    """

    s = np.asarray(s)
    if np.all(s == 0):
        raise ValueError('Q_slow: s=0 is not allowed (hat s is undefined)')

    # Y_{l3,m3}(hat s). Direction of s doesn't depend on pixsize (it cancels).
    s_float = s.astype(float)
    r_s = np.sqrt(np.sum(s_float**2))
    theta_s = np.arccos(s_float[2] / r_s)
    phi_s = np.arctan2(s_float[1], s_float[0])

    C = (4*np.pi)**1.5 * np.sqrt((2*l1+1) * (2*l2+1) * (2*l3+1))

    result = 0.0 + 0.0j

    for m1 in range(-l1, l1+1):
        for m2 in range(-l2, l2+1):
            m3 = -m1 - m2
            if abs(m3) > l3:
                continue

            w3j = utils.wigner_3j(l1, l2, l3, m1, m2, m3)
            if w3j == 0:
                continue

            # f1(x1) * Y_{l1,m1}(hat x1)
            g1_re, g1_im = multiply_ylm_real_space(box, f1, l1, m1)

            # f2(x1+s) * Y_{l2,m2}(hat(x1+s)):
            # multiply f2 by Y_{l2,m2}, then translate so pixel x1 gets value at x1+s.
            g2_re, g2_im = multiply_ylm_real_space(box, f2, l2, m2)
            g2_re = translate_cyclic(g2_re, s)
            g2_im = translate_cyclic(g2_im, s)

            # Integral: V_pix * sum_{x1} g1(x1) * g2(x1+s)
            # where g1, g2 are complex: g = g_re + i*g_im
            integral = box.pixel_volume * (
                np.sum(g1_re * g2_re - g1_im * g2_im)
                + 1j * np.sum(g1_re * g2_im + g1_im * g2_re)
            )

            # Y_{l3,m3}(hat s)
            ylm_s = scipy.special.sph_harm_y(l3, m3, theta_s, phi_s)

            result += w3j * ylm_s * integral

    return C * result


if __name__ == '__main__':
    test_flatten_real()
    test_flatten_fourier()

