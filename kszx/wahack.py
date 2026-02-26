from . import Box

import numpy as np


####################################################################################################
#
# Helpers for flattening/unflattening arrays with conjugacy constraint u[i]* = u[(-i)%n].
#
# The structure is recursive: to flatten a D-dimensional array, we split the last axis into
# "boundary" indices (where -k ≡ k, i.e. k=0 and k=n/2 if n is even) and "interior" indices.
# Interior modes have independent real/imaginary parts. Boundary slices satisfy the same
# conjugacy constraint in (D-1) dimensions, so we recurse. The base case (1D) decomposes
# into self-conjugate scalars (real-valued) and conjugate pairs.
#
# Helpers write directly into a pre-allocated destination array to avoid intermediate
# allocations and np.concatenate() calls. Each helper returns the updated offset.


def _flatten_conj_1d(u, out, offset):
    """Write n real values from conjugate-symmetric length-n array u into out[offset:]."""
    n = len(u)
    nh = n // 2
    end = nh if (n % 2 == 0) else (n + 1) // 2
    ni = end - 1
    out[offset] = u[0].real
    offset += 1
    if n % 2 == 0:
        out[offset] = u[nh].real
        offset += 1
    out[offset:offset+ni] = u[1:end].real
    out[offset+ni:offset+2*ni] = u[1:end].imag
    return offset + 2 * ni


def _unflatten_conj_1d(inp, offset, dest):
    """Read from inp[offset:], write conjugate-symmetric values into dest[:]."""
    n = len(dest)
    nh = n // 2
    end = nh if (n % 2 == 0) else (n + 1) // 2
    ni = end - 1
    dest[0] = inp[offset]
    offset += 1
    if n % 2 == 0:
        dest[nh] = inp[offset]
        offset += 1
    dest[1:end].real = inp[offset:offset+ni]
    dest[1:end].imag = inp[offset+ni:offset+2*ni]
    offset += 2 * ni
    if end > 1:
        dest[n-1:n-end:-1] = np.conj(dest[1:end])
    return offset


def _flatten_conj_2d(t, out, offset):
    """Write n0*n1 real values from conjugate-symmetric (n0,n1) array t into out[offset:]."""
    n0, n1 = t.shape
    nh = n1 // 2
    end = nh if (n1 % 2 == 0) else (n1 + 1) // 2
    ni = n0 * (end - 1)
    # Boundary columns satisfy 1D conjugacy on axis 0
    offset = _flatten_conj_1d(t[:, 0], out, offset)
    if n1 % 2 == 0:
        offset = _flatten_conj_1d(t[:, nh], out, offset)
    # Interior columns: all n0 complex values are independent
    out[offset:offset+ni] = t[:, 1:end].real.reshape(-1)
    out[offset+ni:offset+2*ni] = t[:, 1:end].imag.reshape(-1)
    return offset + 2 * ni


def _unflatten_conj_2d(inp, offset, dest):
    """Read from inp[offset:], write conjugate-symmetric values into dest[:,:]."""
    n0, n1 = dest.shape
    nh = n1 // 2
    end = nh if (n1 % 2 == 0) else (n1 + 1) // 2
    ni = end - 1
    isize = n0 * ni
    offset = _unflatten_conj_1d(inp, offset, dest[:, 0])
    if n1 % 2 == 0:
        offset = _unflatten_conj_1d(inp, offset, dest[:, nh])
    dest[:, 1:end].real = inp[offset:offset+isize].reshape(n0, ni)
    dest[:, 1:end].imag = inp[offset+isize:offset+2*isize].reshape(n0, ni)
    offset += 2 * isize
    # Fill conjugate half: t[i, n1-j] = conj(t[(-i)%n0, j])
    if end > 1:
        neg_i = (-np.arange(n0)) % n0
        dest[:, n1-1:n1-end:-1] = np.conj(dest[neg_i, 1:end])
    return offset


####################################################################################################


def flatten(box, arr):
    """
    Flattens 3-d real-space or Fourier-space map to a real (not complex) vector.
    The length of the vector is N = (box.npix[0] * box.npix[1] * box.npix[2]).

    For a real-space map, this is a trivial operation. For a Fourier-space map,
    the array is assumed to satisfy f(k)^* = f(-k). This ensures that the total
    number of real independent degrees of freedom is equal to N, even though the
    array size is slightly larger.
    """

    assert isinstance(box, Box)
    assert box.ndim == 3
    arr = np.asarray(arr)

    if box.is_real_space_map(arr):
        # Real-space map: trivial flattening operation
        return arr.reshape(-1)

    elif box.is_fourier_space_map(arr):
        n0, n1, n2 = box.npix
        n = n0 * n1 * n2
        out = np.empty(n)
        offset = 0

        # Boundary slices: k2=0 (always), and k2=n2/2 (if n2 is even).
        # Each boundary slice has shape (n0, n1) and satisfies the 2D conjugacy
        # constraint t[i,j]^* = t[(-i)%n0, (-j)%n1].
        offset = _flatten_conj_2d(arr[:,:,0], out, offset)
        if n2 % 2 == 0:
            offset = _flatten_conj_2d(arr[:,:,-1], out, offset)

        # Interior modes (k2 not on the boundary) have independent real
        # and imaginary parts, contributing 2 real DOFs per mode.
        interior = arr[:,:,1:-1] if (n2 % 2 == 0) else arr[:,:,1:]
        isize = interior.size
        out[offset:offset+isize] = interior.real.reshape(-1)
        out[offset+isize:offset+2*isize] = interior.imag.reshape(-1)
        offset += 2 * isize

        assert offset == n
        return out

    else:
        raise RuntimeError('bad shape/dtype')


def unflatten(box, arr, *, fourier=True):
    """Inverse of flatten()."""

    assert isinstance(box, Box)
    assert box.ndim == 3

    arr = np.asarray(arr)
    n = np.prod(box.npix)

    if (arr.dtype != float) or (arr.shape != (n,)):
        raise RuntimeError('unflatten(): got {arr.shape=} and {arr.dtype=}, expected dtype=float and shape={(n,)}')

    if not fourier:
        # Real-space map: trivial unflattening operation
        return arr.reshape(box.real_space_shape)

    else:
        n0, n1, n2 = box.npix
        nk2 = n2 // 2 + 1
        ret = np.zeros(box.fourier_space_shape, dtype=complex)
        offset = 0

        # Boundary slices
        offset = _unflatten_conj_2d(arr, offset, ret[:,:,0])
        if n2 % 2 == 0:
            offset = _unflatten_conj_2d(arr, offset, ret[:,:,-1])

        # Interior modes
        n_boundary = 2 if (n2 % 2 == 0) else 1
        n_interior = nk2 - n_boundary
        isize = n0 * n1 * n_interior

        interior = ret[:,:,1:-1] if (n2 % 2 == 0) else ret[:,:,1:]
        interior.real = arr[offset:offset+isize].reshape(n0, n1, n_interior)
        interior.imag = arr[offset+isize:offset+2*isize].reshape(n0, n1, n_interior)
        offset += 2 * isize

        assert offset == n
        return ret
