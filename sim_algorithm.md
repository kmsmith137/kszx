# Algorithm for simulating rank-1 2PCF fields

## Problem

Simulate real-space Gaussian fields $\delta_1(\mathbf{x}), \delta_2(\mathbf{x})$ with the rank-1 2PCF
from Eq. (hspin_sig) of the paper, with $F_i = 1$:

$$\langle \delta_1(\mathbf{x}_1) \delta_2(\mathbf{x}_2) \rangle
= \int_{\mathbf{k}'} e^{i\mathbf{k}' \cdot (\mathbf{x}_1 - \mathbf{x}_2)} \,
P(\mathbf{k}') \,
\mathcal{L}_{\ell_1^S}(\hat{k}' \cdot \hat{x}_1) \,
\mathcal{L}_{\ell_2^S}(\hat{k}' \cdot \hat{x}_2)$$

where $\int_{\mathbf{k}'} = \int d^3k' / (2\pi)^3$.

## Inputs

- `box`: a `kszx.Box` instance
- `l1S`, `l2S`: non-negative integers (signal spins)
- `pk`: a self-conjugate Fourier-space map (real-valued, non-negative),
  following the wahack convention below

## Wahack convention for `pk`

The function $P(\mathbf{k})$ in the 2PCF satisfies $P(-\mathbf{k})^* = (-1)^{\ell_1^S + \ell_2^S} P(\mathbf{k})$.

Define $p = (\ell_1^S + \ell_2^S) \bmod 2$. Then:

- If $p = 0$ (even): `pk` $= P(\mathbf{k})$
- If $p = 1$ (odd): `pk` $= -i P(\mathbf{k})$, i.e. $P(\mathbf{k}) = i \cdot$ `pk`

This ensures `pk` is always self-conjugate ($\text{pk}(-\mathbf{k})^* = \text{pk}(\mathbf{k})$).

Equivalently: $P(\mathbf{k}) = i^p \cdot \text{pk}(\mathbf{k})$.

## Algorithm

```python
import numpy as np
import kszx

def simulate_rank1(box, l1S, l2S, pk):
    """Simulate delta_1, delta_2 with rank-1 2PCF (Eq. hspin_sig, F_i=1).

    Args:
        box: kszx.Box
        l1S, l2S: signal spins (non-negative integers)
        pk: self-conjugate Fourier-space map (real, non-negative),
            wahack convention (see above)

    Returns:
        (delta1, delta2): real-space maps
    """
    # Step 1: simulate Gaussian field with power spectrum pk
    noise = kszx.simulate_white_noise(box, fourier=True)
    g = np.sqrt(pk) * noise    # g(k) with <g(k) g(k')*> = V_box pk(k) delta_{kk'}

    # Step 2: apply spin-l FFTs
    delta1 = kszx.fft_c2r(box, g, spin=l1S)
    delta2 = kszx.fft_c2r(box, g, spin=l2S)

    # Step 3: sign correction (see derivation below)
    if l1S % 2 == 0 and l2S % 2 == 1:
        delta2 = -delta2

    return delta1, delta2
```

## Sign correction factor

Define $\sigma = i^p / (\epsilon_{\ell_1^S} \epsilon_{\ell_2^S}^*)$, where
$\epsilon_\ell = i$ for odd $\ell$ and $\epsilon_\ell = 1$ for even $\ell$
(the kszx spin-FFT phase convention).

All four parity cases:

| $\ell_1^S$ | $\ell_2^S$ | $\epsilon_{\ell_1}\epsilon_{\ell_2}^*$ | $i^p$ | $\sigma$ |
|:---:|:---:|:---:|:---:|:---:|
| even | even | $1$ | $1$ | $+1$ |
| odd  | odd  | $1$ | $1$ | $+1$ |
| odd  | even | $i$ | $i$ | $+1$ |
| even | odd  | $-i$ | $i$ | $-1$ |

So $\sigma = -1$ when $\ell_1^S$ is even and $\ell_2^S$ is odd, and $\sigma = +1$ otherwise.

## Derivation

### Setup

The kszx spin-$l$ c2r FFT computes (see `docs/source/fft.rst`):
$$\delta(x) = V_{\text{box}}^{-1} \sum_k \epsilon_l \, P_l(\hat{k} \cdot \hat{x}) \, g(k) \, e^{ik \cdot x}$$

We generate both fields from the same self-conjugate Gaussian field $g(k)$:
$$\delta_1(x_1) = V_{\text{box}}^{-1} \sum_k \epsilon_{\ell_1^S} P_{\ell_1^S}(\hat{k}\cdot\hat{x}_1) \, g(k) \, e^{ik\cdot x_1}$$
$$\delta_2(x_2) = \sigma \cdot V_{\text{box}}^{-1} \sum_{k'} \epsilon_{\ell_2^S} P_{\ell_2^S}(\hat{k}'\cdot\hat{x}_2) \, g(k') \, e^{ik'\cdot x_2}$$

### Key correlator identity

For a self-conjugate Gaussian field $g$ with $\langle g(k) g(k')^* \rangle = V_{\text{box}} \, \text{pk}(k) \, \delta_{kk'}$,
the un-conjugated correlator is:
$$\langle g(k) \, g(k') \rangle = V_{\text{box}} \, \text{pk}(k) \, \delta_{k,-k'}$$

**Proof:** For a generic mode $k \neq -k$, write $g(k) = a + ib$ with $\text{Var}(a) = \text{Var}(b) = V_\text{box}\,\text{pk}(k)/2$.
Self-conjugacy gives $g(-k) = g(k)^* = a - ib$. Then:
- $\langle g(k)^2 \rangle = \langle a^2 - b^2 \rangle = 0$
- $\langle g(k)\,g(-k) \rangle = \langle a^2 + b^2 \rangle = V_\text{box}\,\text{pk}(k)$

For self-conjugate modes ($k = -k$, i.e. $k=0$ or Nyquist), $g(k)$ is real and
$\langle g(k)^2 \rangle = V_\text{box}\,\text{pk}(k)$, consistent with $\delta_{k,-k'} = \delta_{kk'}$.

### Computing the 2PCF

$$\langle \delta_1(x_1)\,\delta_2(x_2) \rangle
= V_\text{box}^{-2}\, \sigma\, \epsilon_{\ell_1^S} \epsilon_{\ell_2^S}
  \sum_{k,k'} P_{\ell_1^S}(\hat{k}\cdot\hat{x}_1)\,P_{\ell_2^S}(\hat{k}'\cdot\hat{x}_2)\,
  \langle g(k)\,g(k') \rangle \, e^{ik\cdot x_1 + ik'\cdot x_2}$$

Substituting $\langle g(k)\,g(k') \rangle = V_\text{box}\,\text{pk}(k)\,\delta_{k,-k'}$
and setting $k' = -k$:

$$= V_\text{box}^{-1}\, \sigma\, \epsilon_{\ell_1^S} \epsilon_{\ell_2^S}
  \sum_k \text{pk}(k) \, P_{\ell_1^S}(\hat{k}\cdot\hat{x}_1) \,
  P_{\ell_2^S}(-\hat{k}\cdot\hat{x}_2) \, e^{ik\cdot(x_1-x_2)}$$

Using $P_\ell(-\hat{k}\cdot\hat{x}) = (-1)^\ell P_\ell(\hat{k}\cdot\hat{x})$
and $\epsilon_\ell (-1)^\ell = \epsilon_\ell^*$:

$$= V_\text{box}^{-1}\, \sigma\, \epsilon_{\ell_1^S} \epsilon_{\ell_2^S}^* \sum_k
  \text{pk}(k) \, P_{\ell_1^S}(\hat{k}\cdot\hat{x}_1) \, P_{\ell_2^S}(\hat{k}\cdot\hat{x}_2)
  \, e^{ik\cdot(x_1-x_2)}$$

### Matching the target

The target 2PCF (discrete version of Eq. hspin_sig) is:
$$V_\text{box}^{-1} \sum_k P(k) \, P_{\ell_1^S}(\hat{k}\cdot\hat{x}_1) \, P_{\ell_2^S}(\hat{k}\cdot\hat{x}_2)
  \, e^{ik\cdot(x_1-x_2)}$$

with $P(k) = i^p \, \text{pk}(k)$. Matching coefficients:
$$\sigma \, \epsilon_{\ell_1^S} \epsilon_{\ell_2^S}^* = i^p$$

$$\sigma = \frac{i^p}{\epsilon_{\ell_1^S} \epsilon_{\ell_2^S}^*}$$

This is always $\pm 1$ (see table above), with $\sigma = -1$ only when
$\ell_1^S$ is even and $\ell_2^S$ is odd.

### Correctness check: reality of the 2PCF

The simulation produces real-valued $\delta_1, \delta_2$ (guaranteed by self-conjugacy of $g$
and the $\epsilon_l$ phase in the c2r transform). Therefore $\langle \delta_1 \delta_2 \rangle$ is
automatically real. One can verify the target 2PCF is also real by checking that the integrand
at $\mathbf{k}$ and $-\mathbf{k}$ are complex conjugates, using the conjugacy condition on $P$.

### Correctness check: epsilon identities

The identity $\epsilon_\ell (-1)^\ell = \epsilon_\ell^*$ follows from:
- Even $\ell$: $1 \cdot 1 = 1 = 1^*$
- Odd $\ell$: $i \cdot (-1) = -i = i^*$

The identity $\epsilon_\ell^2 = (-1)^\ell$ follows similarly.
