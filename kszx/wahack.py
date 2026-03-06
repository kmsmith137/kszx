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

    In implementation, it's convenient to use real spherical harmonics X_{li}(\hat x).
    (See the bottom of the "FFTs" page of the sphnix docs for their definition.)
    
    Define v_{li}(k) by:
      v_{li}(k) = int_x e^{-ik.x} f(x) X_{li}(\hat x)

    Then v_{li}(k) is "self-conjugate", in the sense that:
      v_{li}(-k) = v_{li}(k)^*

    This is convenient because many functions in kszx operate on self-conjugate
    Fourier-space maps (e.g. FFTs).

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
                cpp_kernels.multiply_xli_real_space(tmp, f, l, i, box.lpos[0], box.lpos[1], box.lpos[2], box.pixsize, 1.0, False)
                self.vli[(l,i)] = core.fft_r2c(box, tmp)

    def vlm_components(self, l, m):
        """Returns self-conjugate maps (alpha, beta) such that V_{lm}(k) = alpha(k) + i*beta(k).

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

        (Similar to 'class Vlm' but with Fourier and real space exchanged.)

        Constructor arguments
        ---------------------

          - box: instance of class Box
          - pk: self-conjugate Fourier space map, either P(k) or -iP(k), see below.
          - ls: set of nonnegative integers

        Members
        -------

          - self.box: the Box instance
          - self.pk: self-conjugate Fourier-space map, see below.
          - self.ls: sorted list of l-values
          - self.pli: dictionary (l,i) -> real-space map, where 0 <= i <= 2l, see below.

        Description
        -----------

        Recall the definition:
           \tilde P_{lm}(s) = \int_k e^{ik.s} P(k) Y_{\ell m}(\hat k)^*

        We assume that P(k) satisfies:
           P(-k)^* = (-1)^l P(k)

        which implies that P_{lm}(s) satisfies:
           P_{lm}^*(s) = (-1)^m P_{l,-m}(s)
        
        The 'pk' constructor arg is P(k) if l is even, or (-iP(k)) if l is odd.
        This ensures that 'pk' is always self-conjugate (i.e. pk[-k]^* = pk[k]).
        This is convenient, since most kszx functions (e.g. core.fft_c2r()) operate
        on self-conjugate Fourier-space maps.

        For this to make sense, l-values in 'ls' must either be all-even, or all-odd.
        We throw an exception otherwise.

        In implementation, it's convenient to use real spherical harmonics X_{li}(\hat k).
        (See the bottom of the "FFTs" page of the sphinx docs for their definition.)

        Define p_{li}(s) by:
           p_{li}(s) = int_k e^{ik.s} pk(k) X_{li}(\hat k)

        Then p_{li}(s) is real-valued:
           p_{li}(s)^* = p_{li}(s)

        The \tP_{lm}(s) maps are given in terms of p_{li}(s) as follows:
           \tP_{l0}(s) = p_{l0}(s)                                    for m = 0
           \tP_{lm}(s) = p_{l,2m-1}(s) - i p_{l,2m}(s)                for m > 0
           \tP_{l,-m}(s) = (-1)^m [ p_{l,2m-1}(s) + i p_{l,2m}(s) ]   for m > 0
        """

        self.box = box
        self.pk = pk
        self.ls = sorted(set(ls))

        parities = set(l % 2 for l in self.ls)
        if len(parities) > 1:
            raise RuntimeError('Plm: l-values must be all-even or all-odd')

        tmp = np.empty(box.fourier_space_shape, dtype=complex)
        self.pli = {}

        for l in self.ls:
            for i in range(2*l + 1):
                cpp_kernels.multiply_xli_fourier_space(tmp, pk, l, i, box.npix[2], 1.0, False)
                self.pli[(l,i)] = core.fft_c2r(box, tmp)

    def plm_components(self, l, m):
        """Returns real-valued real-space maps (Re \tP_{lm}(s), Im \tP_{lm}(s)).

        Im is None when m=0 (meaning zero).
        """

        if m == 0:
            return self.pli[(l, 0)], None
        elif m > 0:
            return self.pli[(l, 2*m-1)], -self.pli[(l, 2*m)]
        else:
            am = abs(m)
            sign = (-1)**am
            return sign * self.pli[(l, 2*am-1)], sign * self.pli[(l, 2*am)]
    
    
def _to_complex(re, im):
    """Convert (Re, Im) pair to complex array. Im=None means zero."""
    if im is None:
        return re + 0j
    return re + 1j * im


def _negate_map(arr):
    """Given real-space map f(s) on a periodic grid, return f(-s)."""
    for ax in range(arr.ndim):
        idx = (-np.arange(arr.shape[ax])) % arr.shape[ax]
        arr = np.take(arr, idx, axis=ax)
    return arr


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
    

class TestPipeline:
    r"""
    Computes <\hat P> for a single rank-one estimator and rank-one signal.

    Implements the first "boxed" equation in the "main calculation" section,
    of the paper, given as a sum over (l3E, l3S, L1, L2, L3, m3E, m3S, M3).

    Note that the Fourier-space weighting U(k) and signal power P(k) must satsify:

       U(-k)^* = (-1)^{l1E + l2E} U(k)
       P(-k)^* = (-1)^{l1S + l2S} P(k)

    The 'uk' argument is U(k) if (l1E+l2E) is even, or (-iU(k)) if (l1E+l2E) is
    odd. This convention ensures that the 'U' argument is always a self-conjugate
    map (U(-k)^* = U(k)). This is convenient, since most kszx functions (e.g. FFTs
    assume that Fourier-space maps are self-conjugate).

    The 'pk' argument works the sampe way, with sign determined by (l1S+l2S).

    The 'f1' and 'f2' arguments are the real-space maps f_i(x) = W_i(x) F_i(x),
    in notation from the paper.

    Current implementation is a brute-force sum, organized for code clarity
    not speed.

    The constructor computes <\hat P> and stores it in self.hP_mean.
    """

    def __init__(self, box, uk, f1, f2, pk, l1E, l2E, l1S, l2S):
        self.box = box
        self.uk = uk
        self.f1 = f1
        self.f2 = f2
        self.pk = pk
        self.l1E = l1E
        self.l2E = l2E
        self.l1S = l1S
        self.l2S = l2S

        self.coeffs = Coeffs(l1E, l2E, l1S, l2S)

        if not self.coeffs.map5:
            self.hP_mean = 0.0
            return

        self.vlm1 = Vlm(box, f1, self.coeffs.L1_vals)
        self.vlm2 = Vlm(box, f2, self.coeffs.L2_vals)
        self.plm_u = Plm(box, uk, self.coeffs.l3E_vals)
        self.plm_p = Plm(box, pk, self.coeffs.l3S_vals)

        L_triples = sorted(set((k[2], k[3], k[4]) for k in self.coeffs.map5))

        ret = 0.0

        for L1, L2, L3 in L_triples:
            for M3 in range(-L3, L3+1):
                q = _to_complex(*Qlllm(self.vlm1, self.vlm2, L1, L2, L3, M3))

                for l3E in self.coeffs.l3E_vals:
                    for l3S in self.coeffs.l3S_vals:
                        c = self.coeffs.map5.get((l3E, l3S, L1, L2, L3), 0)
                        if c == 0:
                            continue

                        for m3E in range(-l3E, l3E+1):
                            m3S = -m3E - M3
                            if abs(m3S) > l3S:
                                continue

                            w3j = utils.wigner_3j(l3E, l3S, L3, m3E, m3S, M3)
                            if w3j == 0:
                                continue

                            u = _to_complex(*self.plm_u.plm_components(l3E, m3E))
                            p = _negate_map(_to_complex(*self.plm_p.plm_components(l3S, m3S)))

                            integral = box.pixel_volume * np.sum(u * q * p)
                            ret += c * w3j * integral

        self.hP_mean = np.real(ret)

        
    def eval_hP(self, delta1, delta2):
        r"""Evaluate $\hat P$ on two window-weighted real-space data maps, returning a real scalar.

        The arguments delta1, delta2 are real-space maps $d_i(x) = W_i(x) \delta_i(x)$.

        Reminder: \hat P is defined by

          $\hat P = \int_k U(k) \int_{x_1 x_2} e^{-ik(x_1-x_2)}
                    d_1(x_1) d_2(x_2) P_{l1E}(\hat k \cdot \hat x_1) P_{l2E}(\hat k \cdot \hat x_2)$

        This equals sigma * map_dot_product(box, uk * alpha1, alpha2), where
        alpha_i = fft_r2c(box, delta_i, spin=liE), and the sign factor sigma
        accounts for three phase mismatches:

          1. fft_r2c includes epsilon_l^* but the estimator has no epsilon factor,
             contributing epsilon_{l1E} * epsilon_{l2E}.
          2. The x2-integral needs e^{+ik.x2}, i.e. B(-k) = epsilon_{l2E} * alpha2(k)^*
             (via self-conjugacy), but the dot product already conjugates alpha2.
          3. self.uk is the self-conjugate version of U(k), related by
             U(k) = epsilon_{l1E+l2E} * uk(k).

        Combined: sigma = epsilon_{l1E} * epsilon_{l2E} * epsilon_{l1E+l2E}
                        = +1 if both spins even, -1 otherwise.
        """

        alpha1 = core.fft_r2c(self.box, delta1, spin=self.l1E)
        alpha2 = core.fft_r2c(self.box, delta2, spin=self.l2E)
        dot = core.map_dot_product(self.box, self.uk * alpha1, alpha2)
        sign = 1 if (self.l1E % 2 == 0 and self.l2E % 2 == 0) else -1
        return sign * dot


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
