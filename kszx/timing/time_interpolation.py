import time
import numpy as np

from .. import core
from ..Box import Box


def time_interpolation(box_nside=1024, npoints=10**8, kernel=None, periodic=False):
    if kernel is None:
        for k in [ 'cic', 'cubic' ]:
            time_interpolation(box_nside, npoints, kernel=k, periodic=periodic)
        return
    
    print('time_interpolation: start')

    npix = (box_nside, box_nside, box_nside)
    box = Box(npix, pixsize=1.0)
    arr = np.ones(npix)
    points = np.random.uniform(size=(npoints,3), low = 2.1, high = box_nside-3.1)
    
    t0 = time.time()
    ret = core.interpolate_points(box, arr, points, kernel, periodic=periodic)
    dt = time.time() - t0
    print(f'time_interpolation({box_nside=}, {npoints=}, {kernel=}, {periodic=}): {dt} seconds, {1.0e9 * (dt/npoints)} ns/point')

    
