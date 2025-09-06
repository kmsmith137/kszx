from .. import Box
from .. import core
from .. import cpp_kernels

import time
import numpy as np


def time_multiply_xli_real_space(box_nside=1024, niter=10, l=5, i=1):
    print('time_multiply_xli_real_space: start')    
    npix = (box_nside, box_nside, box_nside)
    dst = np.ones(npix)
    src = np.ones(npix)
    lpos = 1.0

    t0 = time.time()
    for _ in range(niter):
        cpp_kernels.multiply_xli_real_space(dst, src, l, i, 1.0, 1.0, 1.0, 1.0, 1.0, False)
        
    dt = time.time() - t0
    nbytes = 16 * niter * box_nside**3
    print(f'time_multiply_xli_real_space({box_nside=}, {niter=}, {l=}, {i=}): {dt} seconds, {1.0e-9 * (nbytes/dt)} GB/sec')


def time_multiply_xli_fourier_space(box_nside=1024, niter=10, l=5, i=1):
    print('time_multiply_xli_fourier_space: start')    
    npix = (box_nside, box_nside, box_nside//2 + 1)
    dst = np.ones(npix, dtype=complex)
    src = np.ones(npix, dtype=complex)
    coeff = 1j if (l % 2) else 1+0j

    t0 = time.time()
    for _ in range(niter):
        cpp_kernels.multiply_xli_fourier_space(dst, src, l, i, box_nside, coeff, False)
        
    dt = time.time() - t0
    nbytes = 16 * niter * box_nside**3
    print(f'time_multiply_xli_fourier_space({box_nside=}, {niter=}, {l=}, {i=}): {dt} seconds, {1.0e-9 * (nbytes/dt)} GB/sec')
