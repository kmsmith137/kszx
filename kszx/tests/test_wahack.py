from . import helpers
from .. import wahack
from .. import core
from .. import cpp_kernels

import numpy as np


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


if __name__ == '__main__':
    test_flatten_real()
    test_flatten_fourier()

