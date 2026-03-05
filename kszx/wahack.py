from . import Box
from . import core
from . import cpp_kernels
from . import utils

import numpy as np


class Coeffs:
    def __init__(self, l1E, l2E, l1S, l2S):
        r"""
        Precomputes coefficients needed to compute estimator response.

        The coefficient is a function of nine l-values (l1E,l2E,l3E,l1S,l2S,l3S,L1,L2,L3)
        and is defined by:
        
           (4\pi)^2 (2L3+1) 
             sqrt[ (2*l3E+1) (2*l3S+1) (2L1+1) (2L2+1) ]
             C_{l1E,l2E,l3E} C_{l1S,l2S,l3S} C_{l1E,l1S,L1} C_{l2E,l2S,L2}
             ninej[ l1E l2E l3E \\ l1S l2S l3S \\ L1 L2 L3 ]
        
        It arises in the first boxed equation in the "main calculation" section
        ("winbox1" in the tex).

        The constructor initializes:

          - self.map5: dict (l3E,l3S,L1,L2,L3) -> (floating-point coeff above)
              Only nonzero coeffs are tabulated!
        
          - self.l3E_vals: sorted list of all l3E-values arising in self.map5.
          - self.l3S_vals: sorted list of all l3ES-values arising in self.map5.
          - self.L1_vals: sorted list of all L1-values arising in self.map5.
          - self.L2_vals: sorted list of all L2-values arising in self.map5.
        """

        self.map5 = {}

        for l3E in range(abs(l1E - l2E), l1E + l2E + 1, 2):
            for l3S in range(abs(l1S - l2S), l1S + l2S + 1, 2):
                for L1 in range(abs(l1E - l1S), l1E + l1S + 1, 2):
                    for L2 in range(abs(l2E - l2S), l2E + l2S + 1, 2):
                        L3_min = max(abs(L1 - L2), abs(l3E - l3S))
                        L3_max = min(L1 + L2, l3E + l3S)

                        for L3 in range(L3_min, L3_max + 1):
                            coeff = (4 * np.pi)**2 * (2*L3 + 1)
                            coeff *= np.sqrt((2*l3E + 1) * (2*l3S + 1) * (2*L1 + 1) * (2*L2 + 1))
                            coeff *= utils.wigner_3j(l1E, l2E, l3E, 0, 0, 0)
                            coeff *= utils.wigner_3j(l1S, l2S, l3S, 0, 0, 0)
                            coeff *= utils.wigner_3j(l1E, l1S, L1, 0, 0, 0)
                            coeff *= utils.wigner_3j(l2E, l2S, L2, 0, 0, 0)
                            coeff *= utils.wigner_9j(l1E, l2E, l3E, l1S, l2S, l3S, L1, L2, L3)

                            if coeff != 0:
                                self.map5[(l3E, l3S, L1, L2, L3)] = coeff

        self.l3E_vals = sorted(set(k[0] for k in self.map5))
        self.l3S_vals = sorted(set(k[1] for k in self.map5))
        self.L1_vals = sorted(set(k[2] for k in self.map5))
        self.L2_vals = sorted(set(k[3] for k in self.map5))
        self.L3_vals = sorted(set(k[4] for k in self.map5))


class Vlm:
    r"""
    Computes Fourier-space maps V_{lm}(k), given real-space map f(x) and set of l-values.

    Constructor arguments
    ---------------------

      - box: instance of class Box
      - f: real-space map (W*F in notation from the paper)
      - ls: set of nonnegative integers

    Members
    -------

      - self.box: the Box instance
      - self.ls: sorted list of l-values
      - self.pli: dictionary (l,i) -> Fourier-space map, where 0 <= i <= 2l, see below.

    Description
    -----------

    Recall that V_{lm}(k) is defined by
      V_{lm}(k) = int_x e^{-ik.x} f(x) Y_{lm}(\hat x)

    and satisfies:
      V_{l,-m}(k) = (-1)^m V_{lm}(-k)^*

    In implementation, it's convenient to work in a real basis.
    For 0 <= i <= 2l, define real spherical harmonics y_{li} by:
       Y_{l0}      for i=0
       Re(Y_{lm})  for i=2m-1
       Im(Y_{lm})  for i=2m

    Define v_{li}(k) by:
      v_{li}(k) = int_x e^{-ik.x} f(x) y_{li}(\hat x)

    Then v_{li}(k) is "real", in the sense that:
      v_{li}(-k) = v_{li}(k)^*

    This is convenient because v_{li} can be represented as an "ordinary" Fourier-space map.

    The V_{lm}(k) maps are given in terms of v_{li}(k) as follows:
    
      V_{l0}(k) = v_{l0}(k)                                    for m = 0
      V_{lm}(k) = v_{l,2m-1}(k) + i v_{l,2m}(k)                for m > 0
      V_{l,-m}(k) = (-1)^m [ v_{l,2m-1}(k) - i v_{l,2m}(k) ]   for m > 0
    """

    def __init__(self, box, f, ls):
        self.box = box
        self.ls = sorted(set(ls))

        tmp = np.empty(box.real_space_shape, dtype=float)
        self.vli = {}

        for l in self.ls:
            for i in range(2*l + 1):
                # multiply_xli_real_space multiplies by X_{li}, but we need Z_{li} (unnormalized).
                # X_{l0} = sqrt(4pi/(2l+1)) * Z_{l0}, X_{li} = sqrt(8pi/(2l+1)) * Z_{li} for i>0.
                # We want FFT(f * Z_{li}) = (1/c_{li}) * FFT(f * X_{li}).
                coeff = np.sqrt((2*l + 1) / (4*np.pi)) if (i == 0) else np.sqrt((2*l + 1) / (8*np.pi))
                cpp_kernels.multiply_xli_real_space(tmp, f, l, i, box.lpos[0], box.lpos[1], box.lpos[2], box.pixsize, coeff, False)
                self.vli[(l,i)] = core.fft_r2c(box, tmp)

    def vlm_components(self, l, m):
        """Returns (alpha, beta) such that V_{lm}(k) = alpha + i*beta.

        alpha, beta are "real" (conjugacy-symmetric) Fourier-space maps.
        beta is None when m=0 (meaning zero).
        """

        if m == 0:
            alpha = self.vli[(l, 0)]
            beta = None
        elif m > 0:
            alpha = self.vli[(l, 2*m-1)]
            beta = self.vli[(l, 2*m)]
        else:
            am = abs(m)
            sign = (-1)**am
            alpha = sign * self.vli[(l, 2*am-1)]
            beta = -sign * self.vli[(l, 2*am)]

        return alpha, beta


class Plm:
    def __init__(self, box, pk, ls):
        r"""
        Computes real-space maps P_{lm}(s), given Fourier-space map P(k) and set of l-values.

        Constructor arguments
        ---------------------

          - box: instance of class Box
          - pk: self-conjgate Fourier space map, either P(k) or -iP(k), see below.
          - ls: set of nonnegative integers

        Members
        -------

          - self.box: the Box instance
          - self.pk: self-conjugate Fourier-space map, see below.
          - self.ls: sorted list of l-values
          - self.vli: dictionary (l,i) -> Fourier-space map, where 0 <= i <= 2l, see below.

        Description
        -----------

        Recall the definition:
           \tilde P_{lm}(s) = \int_k e^{ik.s} P(k) Y_{\ell m}^*(s)

        We assume that P(k) satisfies:
           P(-k)^* = (-1)^l P(k)

        which implies that P_{lm}(s) satisfies:
           P_{lm}^*(s) = (-1)^m P_{l,-m}(s)
        
        The 'pk' constructor arg is P(k) if l is even, or (-iP(k)) if l is odd.
        This ensures that 'pk' is always self-conjugate (i.e. pk[-k]^* = pk[k]).
        This is convenient, since most kszx functions (e.g. core.fft_c2r() operate
        on self-conjugate Fourier-space maps.

        For this to make sense, l-values in 'ls' must either be all-even, or all-odd.
        We throw an exception otherwise.

        As in 'class Vlm' it's convenient to work in a real basis.
        FIXME -- incomplete
        """

        # Calls cpp_kernels.multiply_xli_real_space() and core.fft_c2r().
        pass

    
    def plm_components(self, l, m):
        """Returns pair (Re P_{lm}(s), Im P_{lm}(s))."""
        pass
    
    
def Qlllm(vlm1, vlm2, l1, l2, l3, m3):
    r"""
    Compute Q^{L1,L2}_{L3,M3}(x), and return it as a pair (r,s) of real-valued real-space maps.

    Recall the definition:
    
      Q^{L1,L2}_{L3,M3}(k)
         = sum_{M1,M2} threej(L1,L2,L3,M1,M2,M3) V^{L1,M1}_1(-k) V^{L2,M2}_2(k).
         = sum_{M1,M2} threej(L1,L2,L3,M1,M2,M3) (-1)^M1 V^{L1,-M1}_1(k)^* V^{L2,M2}_2(k).

      Q^{L1,L2}_{L3,M3}(x) = int_k Q^{L1,L2}_{L3,M3}(k) e^{ik.x}
    
    We write Q^{L1,L2}_{L3,M3}(x) = r(x) + i*s(x), where r,s are real-valued real-space maps,
    and return a pair (r,s).

    The 'vlm1' and 'vlm2' arguments are instances of 'class Vlm'.

    Note that Q^{L1,L2}_{L3,M3} satisfies:

       Q^{L1,L2}_{L3,M3}(-k)^* = (-1)^(L1+L2+L3+M3) Q^{L1,L2}_{L3,-M3}(k)
       Q^{L1,L2}_{L3,M3}(x)^* = (-1)^(L1+L2+L3+M3) Q^{L1,L2}_{L3,-M3}(x)
    """

    box = vlm1.box
    r = np.zeros(box.fourier_space_shape, dtype=complex)
    s = np.zeros(box.fourier_space_shape, dtype=complex)

    for m1 in range(-l1, l1+1):
        m2 = -m1 - m3
        if abs(m2) > l2:
            continue

        w3j = utils.wigner_3j(l1, l2, l3, m1, m2, m3)
        if w3j == 0:
            continue

        # Using second definition: (-1)^M1 * V1^{l1,-m1}(k)^* * V2^{l2,m2}(k)
        # V1^{l1,-m1}(k) = a1 + i*b1, so V1^{l1,-m1}(k)^* = a1^* - i*b1^*
        # V2^{l2,m2}(k) = a2 + i*b2
        a1, b1 = vlm1.vlm_components(l1, -m1)
        a2, b2 = vlm2.vlm_components(l2, m2)

        # (a1^* - i*b1^*)(a2 + i*b2) = (a1^*a2 + b1^*b2) + i(a1^*b2 - b1^*a2)
        coeff = w3j * (-1)**m1
        r += coeff * np.conj(a1) * a2
        if (b1 is not None) and (b2 is not None):
            r += coeff * np.conj(b1) * b2
        if b2 is not None:
            s += coeff * np.conj(a1) * b2
        if b1 is not None:
            s -= coeff * np.conj(b1) * a2

    r = core.fft_c2r(box, r)
    s = core.fft_c2r(box, s)
    return r, s
    

def hP_mean(box, U, P, vlm1, vlm2, l1E, l2E, l1S, l2S):
    r"""
    Computes <\hat P> for a single rank-one estimator and rank-one signal.

    Implements the first "boxed" equation in the "main calculation" section,
    given as a sum over (l3E, l3S, L1, L2, L3, m3E, m3S, M3).

    Note that the Fourier-space weighting U(k) and signal power P(k) must satsify:

       U(-k)^* = (-1)^{l1E + l2E} U(k)
       P(-k)^* = (-1)^{l1S + l2S} P(k)

    The 'U' argument is U(k) if (l1E+l2E) is even, or (-iU(k)) if (l1E+l2E) is
    odd. This convention ensures that the 'U' argument is always a self-conjugate
    map (U(-k)^* = U(k)). This is convenient, since most kszx functions (e.g.
    core.fft_c2r() assume that Fourier-space maps are self-conjugate).

    The 'P' argument works the sampe way, with sign determined by (l1S+l2S).

    Current implementation is a brute-force sum, organized for code clarity
    not speed. We call 
    
    Arguments:
    
      - U: Self-conjugate Fourier-space map, either U(k) or -iU(k)
    
      - P: Self-conjugate Fourier-space map, either P(k) or -iP(k)
    
      - vlm1, vlm2: instance of class Vlm, representing 
    """

    pass


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
#
# The scale factors sc and ic normalize self-conjugate and interior (paired) modes
# respectively, so that np.dot(v,v) equals map_dot_product(box, f, f).


def _flatten_conj(arr, n_last, out, offset, sc, ic):
    """Flatten conjugate-symmetric complex array to real values.

    arr: complex array, last axis has conjugacy with logical size n_last.
    out: pre-allocated real output array.
    offset: current write position in out.
    sc: scale factor for self-conjugate modes (0-d base case).
    ic: scale factor for interior (paired) modes.
    Returns: updated offset.
    """

    nh = n_last // 2
    end = nh if (n_last % 2 == 0) else (n_last + 1) // 2

    # Boundary slices: recurse (or write scalar for 0-d base case)
    for idx in [0] + ([nh] if (n_last % 2 == 0) else []):
        sub = arr[..., idx]
        if sub.ndim == 0:
            out[offset] = sub.real * sc
            offset += 1
        else:
            offset = _flatten_conj(sub, sub.shape[-1], out, offset, sc, ic)

    # Interior modes: independent real and imaginary parts
    interior = arr[..., 1:end]
    isize = interior.size
    out[offset:offset+isize] = interior.real.reshape(-1)
    out[offset+isize:offset+2*isize] = interior.imag.reshape(-1)
    out[offset:offset+2*isize] *= ic
    return offset + 2 * isize


def _unflatten_conj(inp, n_last, dest, offset, sc, ic):
    """Unflatten real values into a conjugate-symmetric complex array.

    inp: real input array.
    n_last: logical size of last axis of dest.
    dest: pre-allocated complex destination array (written in-place).
    offset: current read position in inp.
    sc: scale factor for self-conjugate modes (divided out).
    ic: scale factor for interior modes (divided out).
    Returns: updated offset.
    """

    nh = n_last // 2
    end = nh if (n_last % 2 == 0) else (n_last + 1) // 2

    # Boundary slices: recurse (or read scalar for 0-d base case)
    for idx in [0] + ([nh] if (n_last % 2 == 0) else []):
        sub = dest[..., idx]
        if sub.ndim == 0:
            dest[..., idx] = inp[offset] / sc
            offset += 1
        else:
            offset = _unflatten_conj(inp, sub.shape[-1], sub, offset, sc, ic)

    # Interior modes: read real and imaginary parts
    interior = dest[..., 1:end]
    isize = interior.size
    interior.real = inp[offset:offset+isize].reshape(interior.shape)
    interior.imag = inp[offset+isize:offset+2*isize].reshape(interior.shape)
    interior *= (1.0 / ic)
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

    The normalization is chosen so that np.dot(flatten(box,f), flatten(box,g)) gives
    the same result as map_dot_product(box, f, g).
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
        sc = 1.0 / np.sqrt(box.box_volume)
        ic = np.sqrt(2.0 / box.box_volume)
        offset = _flatten_conj(arr, box.npix[-1], out, 0, sc, ic)
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
        sc = 1.0 / np.sqrt(box.box_volume)
        ic = np.sqrt(2.0 / box.box_volume)
        offset = _unflatten_conj(arr, box.npix[-1], ret, 0, sc, ic)
        assert offset == n
        return ret
