from . import Box

import numpy as np


####################################################################################################
#
# Helpers for flattening/unflattening arrays with conjugacy constraint arr[i]* = arr[(-i) % n].
#
# The structure is recursive: split the last axis into "boundary" indices (where -k ≡ k,
# i.e. k=0 and k=n/2 if n is even) and "interior" indices. Interior modes have independent
# real/imaginary parts. Boundary slices satisfy the same conjugacy constraint in (D-1)
# dimensions, so we recurse. The base case is a 0-d array (self-conjugate scalar = real).
#
# Both helpers write directly into a pre-allocated destination array and return the
# updated offset. The n_last parameter is the logical size of the last axis (which may
# differ from the array shape for rfft convention).


def _flatten_conj(arr, n_last, out, offset):
    """Flatten conjugate-symmetric complex array to real values.

    arr: complex array, last axis has conjugacy with logical size n_last.
    out: pre-allocated real output array.
    offset: current write position in out.
    Returns: updated offset.
    """

    nh = n_last // 2
    end = nh if (n_last % 2 == 0) else (n_last + 1) // 2

    # Boundary slices: recurse (or write scalar for 0-d base case)
    for idx in [0] + ([nh] if (n_last % 2 == 0) else []):
        sub = arr[..., idx]
        if sub.ndim == 0:
            out[offset] = sub.real
            offset += 1
        else:
            offset = _flatten_conj(sub, sub.shape[-1], out, offset)

    # Interior modes: independent real and imaginary parts
    interior = arr[..., 1:end]
    isize = interior.size
    out[offset:offset+isize] = interior.real.reshape(-1)
    out[offset+isize:offset+2*isize] = interior.imag.reshape(-1)
    return offset + 2 * isize


def _unflatten_conj(inp, n_last, dest, offset):
    """Unflatten real values into a conjugate-symmetric complex array.

    inp: real input array.
    n_last: logical size of last axis of dest.
    dest: pre-allocated complex destination array (written in-place).
    offset: current read position in inp.
    Returns: updated offset.
    """

    nh = n_last // 2
    end = nh if (n_last % 2 == 0) else (n_last + 1) // 2

    # Boundary slices: recurse (or read scalar for 0-d base case)
    for idx in [0] + ([nh] if (n_last % 2 == 0) else []):
        sub = dest[..., idx]
        if sub.ndim == 0:
            dest[..., idx] = inp[offset]
            offset += 1
        else:
            offset = _unflatten_conj(inp, sub.shape[-1], sub, offset)

    # Interior modes: read real and imaginary parts
    interior = dest[..., 1:end]
    isize = interior.size
    interior.real = inp[offset:offset+isize].reshape(interior.shape)
    interior.imag = inp[offset+isize:offset+2*isize].reshape(interior.shape)
    offset += 2 * isize

    # Fill conjugate half (full DFT only, not rfft)
    if end > 1 and dest.shape[-1] == n_last:
        negated = dest[..., 1:end]
        for ax in range(dest.ndim - 1):
            neg = (-np.arange(dest.shape[ax])) % dest.shape[ax]
            slices = [slice(None)] * negated.ndim
            slices[ax] = neg
            negated = negated[tuple(slices)]
        dest[..., n_last-1:n_last-end:-1] = np.conj(negated)

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
        ret = arr.reshape(-1, copy=True)
        ret *= (box.pixsize)**1.5
        return ret

    elif box.is_fourier_space_map(arr):
        n = np.prod(box.npix)
        out = np.empty(n)
        offset = _flatten_conj(arr, box.npix[-1], out, 0)
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
        ret = np.reshape(arr, box.real_space_shape, copy=True)
        ret *= (box.pixsize)**(-1.5)
        return ret

    else:
        ret = np.zeros(box.fourier_space_shape, dtype=complex)
        offset = _unflatten_conj(arr, box.npix[-1], ret, 0)
        assert offset == n
        return ret
