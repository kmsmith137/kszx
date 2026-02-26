from . import helpers
from .. import wahack
from .. import core

import numpy as np


def test_flatten_real():
    """Test that flatten() and unflatten() are inverses (real-space maps)."""

    print('test_flatten_real(): start')

    for _ in range(100):
        box = helpers.random_box(ndim=3)
        n = np.prod(box.npix)

        # flatten -> unflatten
        arr = core.simulate_white_noise(box, fourier=False)
        arr2 = wahack.flatten(box, arr)
        arr2 = wahack.unflatten(box, arr2, fourier=False)
        eps = helpers.compare_arrays(arr, arr2)
        assert eps < 1.0e-10

        # unflatten -> flatten
        arr = np.random.normal(size=n)
        arr2 = wahack.unflatten(box, arr, fourier=False)
        arr2 = wahack.flatten(box, arr2)
        eps = helpers.compare_arrays(arr, arr2)
        assert eps < 1.0e-10
    
    print('test_flatten_real(): pass')


def test_flatten_fourier():
    """Test that flatten() and unflatten() are inverses (Fourier-space maps)."""

    print('test_flatten_fourier(): start')

    for _ in range(100):
        box = helpers.random_box(ndim=3)
        n = np.prod(box.npix)

        # flatten -> unflatten
        arr = core.simulate_white_noise(box, fourier=True)
        arr2 = wahack.flatten(box, arr)
        assert arr2.shape == (n,)
        assert arr2.dtype == float
        arr2 = wahack.unflatten(box, arr2, fourier=True)
        eps = helpers.compare_arrays(arr, arr2)
        assert eps < 1.0e-10

        # unflatten -> flatten
        arr = np.random.normal(size=n)
        arr2 = wahack.unflatten(box, arr, fourier=True)
        arr2 = wahack.flatten(box, arr2)
        eps = helpers.compare_arrays(arr, arr2)
        assert eps < 1.0e-10

    print('test_flatten_fourier(): pass')


if __name__ == '__main__':
    test_flatten_real()
    test_flatten_fourier()

