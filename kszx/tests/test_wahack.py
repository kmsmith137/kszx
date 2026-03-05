from . import helpers
from .. import Box
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


if __name__ == '__main__':
    test_flatten_real()
    test_flatten_fourier()
    test_Q_vs_Q_slow()

