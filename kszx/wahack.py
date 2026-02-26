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


def _flatten_conj_1d(u):
    """Flatten length-n complex array with u[i]* = u[(-i)%n] to n real values."""
    n = len(u)
    nh = n // 2
    end = nh if (n % 2 == 0) else (n + 1) // 2
    parts = []
    # Self-conjugate at i=0 (real-valued)
    parts.append(u[0:1].real)
    # Self-conjugate at i=n/2 (real-valued, only if n even)
    if n % 2 == 0:
        parts.append(u[nh:nh+1].real)
    # Paired modes i=1,...,end-1: real and imaginary parts are independent
    parts.append(u[1:end].real)
    parts.append(u[1:end].imag)
    return np.concatenate(parts)


def _unflatten_conj_1d(v, n):
    """Inverse of _flatten_conj_1d. Returns complex array of length n with conjugacy."""
    u = np.zeros(n, dtype=complex)
    nh = n // 2
    end = nh if (n % 2 == 0) else (n + 1) // 2
    n_interior = end - 1

    offset = 0
    u[0] = v[0]
    offset += 1
    if n % 2 == 0:
        u[nh] = v[offset]
        offset += 1
    u[1:end] = v[offset:offset+n_interior] + 1j * v[offset+n_interior:offset+2*n_interior]
    offset += 2 * n_interior
    # Fill conjugate half: u[n-i] = conj(u[i])
    if end > 1:
        u[n-1:n-end:-1] = np.conj(u[1:end])
    assert offset == n
    return u


def _flatten_conj_2d(t):
    """Flatten complex (n0,n1) array with t[i,j]* = t[(-i)%n0,(-j)%n1] to n0*n1 real values."""
    n0, n1 = t.shape
    nh = n1 // 2
    end = nh if (n1 % 2 == 0) else (n1 + 1) // 2
    parts = []
    # Boundary columns (where -j ≡ j mod n1) satisfy 1D conjugacy on axis 0
    parts.append(_flatten_conj_1d(t[:, 0]))
    if n1 % 2 == 0:
        parts.append(_flatten_conj_1d(t[:, nh]))
    # Interior columns: all n0 complex values are independent
    interior = t[:, 1:end]
    parts.append(interior.real.reshape(-1))
    parts.append(interior.imag.reshape(-1))
    return np.concatenate(parts)


def _unflatten_conj_2d(v, n0, n1):
    """Inverse of _flatten_conj_2d. Returns complex (n0,n1) array with conjugacy."""
    t = np.zeros((n0, n1), dtype=complex)
    nh = n1 // 2
    end = nh if (n1 % 2 == 0) else (n1 + 1) // 2
    n_interior = end - 1

    offset = 0
    t[:, 0] = _unflatten_conj_1d(v[offset:offset+n0], n0)
    offset += n0
    if n1 % 2 == 0:
        t[:, nh] = _unflatten_conj_1d(v[offset:offset+n0], n0)
        offset += n0
    isize = n0 * n_interior
    t[:, 1:end] = v[offset:offset+isize].reshape(n0, n_interior) + 1j * v[offset+isize:offset+2*isize].reshape(n0, n_interior)
    offset += 2 * isize
    # Fill conjugate half: t[i, n1-j] = conj(t[(-i)%n0, j])
    if end > 1:
        neg_i = (-np.arange(n0)) % n0
        t[:, n1-1:n1-end:-1] = np.conj(t[neg_i, 1:end])
    assert offset == n0 * n1
    return t


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
        parts = []

        # Boundary slices: k2=0 (always), and k2=n2/2 (if n2 is even).
        # Each boundary slice has shape (n0, n1) and satisfies the 2D conjugacy
        # constraint t[i,j]^* = t[(-i)%n0, (-j)%n1].
        parts.append(_flatten_conj_2d(arr[:,:,0]))
        if n2 % 2 == 0:
            parts.append(_flatten_conj_2d(arr[:,:,-1]))

        # Interior modes (k2 not on the boundary) have independent real
        # and imaginary parts, contributing 2 real DOFs per mode.
        interior = arr[:,:,1:-1] if (n2 % 2 == 0) else arr[:,:,1:]
        parts.append(interior.real.reshape(-1))
        parts.append(interior.imag.reshape(-1))

        return np.concatenate(parts)

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
        bsize = n0 * n1

        # Boundary slice at k2=0
        ret[:,:,0] = _unflatten_conj_2d(arr[offset:offset+bsize], n0, n1)
        offset += bsize

        # Boundary slice at k2=n2/2 (only if n2 is even)
        if n2 % 2 == 0:
            ret[:,:,-1] = _unflatten_conj_2d(arr[offset:offset+bsize], n0, n1)
            offset += bsize

        # Interior modes: reconstruct complex from real and imaginary parts.
        n_boundary = 2 if (n2 % 2 == 0) else 1
        n_interior = nk2 - n_boundary
        isize = n0 * n1 * n_interior

        real_part = arr[offset:offset+isize].reshape(n0, n1, n_interior)
        imag_part = arr[offset+isize:offset+2*isize].reshape(n0, n1, n_interior)
        offset += 2 * isize

        if n2 % 2 == 0:
            ret[:,:,1:-1] = real_part + 1j * imag_part
        else:
            ret[:,:,1:] = real_part + 1j * imag_part

        assert offset == n
        return ret
