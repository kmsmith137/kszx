r"""Radial profiles of halos: the Battaglia (2016) gas optical depth and a truncated NFW.

Conventions:

  - Halo masses are $M_{200c}$ in Msol, passed as ``logm`` $= \log_{10}(M_{200c}/M_\odot)$.
  - Radii are comoving Mpc and wavenumbers comoving Mpc$^{-1}$ (no h-units, as in the rest of kszx).
  - The only cosmology needed is a :class:`~kszx.Cosmology`: its ``H(z=...)`` and the parameters
    ``ombh2``, ``omch2``, ``h``. There is no pyccl dependency.
  - In profile methods, ``cosmo`` is positional and everything else is a keyword (as in
    :class:`~kszx.Cosmology`). ``z`` is a scalar; ``r`` (or ``R``) and ``logm`` broadcast.

Every profile is a :class:`HaloProfile`: a spherical ``density(r)`` truncated at a comoving radius
``r_t``, with its line-of-sight projection ``projected(R)`` and its 3-d Fourier transform ``fourier(k)``.
A halo at comoving distance $\chi$ subtends ``projected(chi * theta)`` at angle $\theta$, and its
harmonic-space profile is ``fourier((l+1/2)/chi) / chi**2``.
"""

import numpy as np
from scipy.special import sici


_C_KMS = 299792.458                                   # speed of light [km/s]
_G_MPC_KMS = 4.30091e-9                               # G [Mpc (km/s)^2 / Msol]
_SIGMA_T = 6.6524587158e-29 / 3.08567758149e22**2     # Thomson cross section [Mpc^2]
_M_P = 1.67262192369e-27 / 1.988409870698051e30       # proton mass [Msol]


####################################################################################################


def rho_crit(cosmo, *, z):
    r"""Returns the critical density $\rho_c(z) = 3H(z)^2 / 8\pi G$, in physical Msol/Mpc$^3$.

    Function args:

       - ``cosmo`` (:class:`~kszx.Cosmology`).
       - ``z`` (float or array): redshift.
    """
    H = _C_KMS * np.asarray(cosmo.H(z=z))    # km/s/Mpc  (cosmo.H() is in Mpc^{-1})
    return 3.0 * H**2 / (8.0 * np.pi * _G_MPC_KMS)


def r200c(cosmo, *, logm, z):
    r"""Returns the comoving radius $R_{200c}$ [Mpc] of halos with $\log_{10}(M_{200c}/M_\odot)$ = ``logm``.

    $M_{200c} = (4\pi/3)\, 200 \rho_c(z) R_{\rm phys}^3$ and $R_{200c} = (1+z) R_{\rm phys}$.
    ``logm`` and ``z`` broadcast.
    """
    M = 10.0 ** np.asarray(logm, dtype=float)
    z = np.asarray(z, dtype=float)
    return np.cbrt(3.0 * M / (800.0 * np.pi * rho_crit(cosmo, z=z))) * (1.0 + z)


def c200c_duffy08(cosmo, *, logm, z):
    r"""Returns the Duffy et al. (2008) concentration $c_{200c}$ (full sample, NFW fit).

    $c_{200c} = 5.71\, (M_{200c} / 2 \times 10^{12}\, h^{-1} M_\odot)^{-0.084}\, (1+z)^{-0.47}$.
    ``logm`` and ``z`` broadcast.
    """
    M = 10.0 ** np.asarray(logm, dtype=float)
    return 5.71 * (M * cosmo.h / 2.0e12) ** -0.084 * (1.0 + np.asarray(z, dtype=float)) ** -0.47


####################################################################################################


class HaloProfile:
    r"""Base class for a spherical halo profile, truncated at comoving radius ``r_t``.

    Subclasses define :meth:`r_t` and :meth:`density`. This class then supplies the line-of-sight
    projection :meth:`projected` and the Fourier transform :meth:`fourier`, both by quadrature,
    so a new profile only has to write down $\rho(r)$. A subclass with a closed-form transform
    can override :meth:`fourier` (see :class:`NFW`).
    """

    def r_t(self, cosmo, *, logm, z):
        """Comoving truncation radius [Mpc]. Must be defined by subclass."""
        raise NotImplementedError(f'{type(self).__name__}.r_t() is not defined')

    def density(self, cosmo, *, r, logm, z):
        """Profile at comoving radius ``r`` [Mpc], zero for ``r > r_t``. Must be defined by subclass."""
        raise NotImplementedError(f'{type(self).__name__}.density() is not defined')

    def projected(self, cosmo, *, R, logm, z, n_l=128):
        r"""Line-of-sight projection $2 \int_0^L \rho(\sqrt{R^2 + l^2})\, dl$, $L = \sqrt{r_t^2 - R^2}$.

        Function args:

           - ``R`` (float or array): comoving projected radius [Mpc].
           - ``logm`` (float or array): $\log_{10}(M_{200c}/M_\odot)$, broadcast against ``R``.
           - ``z`` (float): redshift.
           - ``n_l`` (int): number of Gauss-Legendre nodes.

        Gauss-Legendre in $\ln l$ on $[10^{-9} L, L]$, so the core is resolved at any $R$.
        ``R`` is clipped below at $10^{-6} r_t$, and the result is zero for $R \ge r_t$.
        """
        lm = np.asarray(logm, dtype=float)
        R, rt, lm = np.broadcast_arrays(np.asarray(R, dtype=float), self.r_t(cosmo, logm=lm, z=z), lm)
        R = np.maximum(R, 1.0e-6 * rt)
        s, w = np.polynomial.legendre.leggauss(n_l)
        a = 0.5 * np.log(1.0e9)
        l = np.sqrt(np.maximum(rt**2 - R**2, 0.0))[..., None] * np.exp(a * (s - 1.0))
        rho = self.density(cosmo, r=np.hypot(R[..., None], l), logm=lm[..., None], z=z)
        return 2.0 * a * (rho * l * w).sum(axis=-1)

    def fourier(self, cosmo, *, k, logm, z):
        r"""Fourier transform $u(k) = 4\pi \int_0^{r_t} r^2 \rho(r)\, \sin(kr)/(kr)\, dr$.

        Function args:

           - ``k`` (float or array): comoving wavenumber [Mpc$^{-1}$].
           - ``logm`` (float or array): $\log_{10}(M_{200c}/M_\odot)$.
           - ``z`` (float): redshift.

        Returns an array of shape ``logm.shape + k.shape``. Units are those of ``density()`` times Mpc$^3$.

        Quadrature: 16-point Gauss-Legendre on at least 64 equal segments of $[0, r_t]$, each
        shorter than 1/3 of a period of $\sin(kr)$ at the largest $k$.
        """
        kshape, mshape = np.shape(k), np.shape(logm)
        k = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
        lm = np.atleast_1d(np.asarray(logm, dtype=float)).ravel()
        x, w = np.polynomial.legendre.leggauss(16)
        out = np.empty((lm.size, k.size))

        for i, m in enumerate(lm):
            rt = float(self.r_t(cosmo, logm=m, z=z))
            ns = max(64, int(np.ceil(k.max() * rt / 2.0)))
            h = rt / ns
            r = (h * np.arange(ns)[:, None] + 0.5 * h * (x + 1.0)).ravel()
            f = np.tile(0.5 * h * w, ns) * r**2 * self.density(cosmo, r=r, logm=m, z=z)
            out[i] = 4.0 * np.pi * (np.sinc(np.outer(k, r) / np.pi) @ f)

        return out.reshape(mshape + kshape)


####################################################################################################


class NFW(HaloProfile):
    def __init__(self, truncate=1.0):
        r"""NFW profile with the Duffy et al. (2008) $c_{200c}$, truncated at ``truncate`` $\times R_{200c}$.

        Normalized to unit volume integral, so ``fourier(k=0) = 1``:

        $$\rho(r) = \frac{1}{4\pi r_s^3\, m(C)} \frac{1}{y (1+y)^2}, \qquad y = r/r_s, \quad
        r_s = R_{200c}/c_{200c}, \quad C = r_t/r_s, \quad m(C) = \ln(1+C) - C/(1+C).$$

        (This is the profile of satellite galaxies in an HOD, or of the dark matter in a halo.)

        Constructor args:

           - ``truncate`` (float): truncation radius in units of $R_{200c}$ (default 1).
        """
        self.truncate = float(truncate)

    def _rs_C(self, cosmo, logm, z):
        c = c200c_duffy08(cosmo, logm=logm, z=z)
        return r200c(cosmo, logm=logm, z=z) / c, self.truncate * c

    def r_t(self, cosmo, *, logm, z):
        """Comoving truncation radius ``truncate * R_200c`` [Mpc]."""
        return self.truncate * r200c(cosmo, logm=logm, z=z)

    def density(self, cosmo, *, r, logm, z):
        """Normalized NFW density at comoving radius ``r`` [Mpc$^{-3}$]; ``r`` and ``logm`` broadcast."""
        rs, C = self._rs_C(cosmo, logm, z)
        y = np.asarray(r, dtype=float) / rs
        m = np.log1p(C) - C / (1.0 + C)
        out = 1.0 / (4.0 * np.pi * rs**3 * m * y * (1.0 + y)**2)
        return np.where(y <= C * (1.0 + 1.0e-9), out, 0.0)

    def fourier(self, cosmo, *, k, logm, z):
        r"""Fourier transform of the truncated NFW profile, in closed form (dimensionless, 1 at $k=0$).

        $$u(k) = \frac{1}{m(C)} \Big[ \sin q\, \big(\mathrm{Si}((1+C)q) - \mathrm{Si}(q)\big)
        - \frac{\sin Cq}{(1+C)q} + \cos q\, \big(\mathrm{Ci}((1+C)q) - \mathrm{Ci}(q)\big) \Big],
        \qquad q = k r_s.$$

        Returns an array of shape ``logm.shape + k.shape``.
        """
        rs, C = self._rs_C(cosmo, logm, z)
        rs, C = np.asarray(rs)[..., None], np.asarray(C)[..., None]
        q = np.maximum(rs * np.ravel(np.asarray(k, dtype=float)), 1.0e-12)
        si1, ci1 = sici((1.0 + C) * q)
        si0, ci0 = sici(q)
        u = np.sin(q) * (si1 - si0) - np.sin(C * q) / ((1.0 + C) * q) + np.cos(q) * (ci1 - ci0)
        u /= np.log1p(C) - C / (1.0 + C)
        return u.reshape(np.shape(logm) + np.shape(k))


####################################################################################################


class Battaglia(HaloProfile):
    # (A, alpha_M, alpha_z) for each of rho_0, alpha, beta (Battaglia 2016, arXiv:1607.02442, Table 2, Delta = 200).
    fits = {
        'AGN': ((4.0e3, 0.29, -0.66), (0.88, -0.03, 0.19), (3.83, 0.04, -0.025)),
        'SH': ((1.9e4, 0.09, -0.95), (0.70, -0.017, 0.27), (4.43, 0.005, 0.037)),
    }

    def __init__(self, fit='AGN', truncate=2.0, mu_e=1.14, fb=None):
        r"""Battaglia (2016) gas density profile, as an optical depth per comoving Mpc.

        The gas density on $M = M_{200c}$, with $x = r / R_{200c}$ (arXiv:1607.02442, Eqs. A1-A2):

        $$\rho_{\rm gas}(r) = f_b\, \rho_c(z)\, \rho_0 \Big(\frac{x}{x_c}\Big)^\gamma
        \Big[1 + \Big(\frac{x}{x_c}\Big)^\alpha\Big]^{-(\beta+\gamma)/\alpha},
        \qquad x_c = 0.5, \quad \gamma = -0.2,$$

        with $(\rho_0, \alpha, \beta) = A\, (M/10^{14} M_\odot)^{\alpha_M}\, (1+z)^{\alpha_z}$
        (Table 2), truncated at ``truncate`` $\times R_{200c}$.

        The second exponent is $-(\beta+\gamma)/\alpha$, so that $\rho \propto x^{-\beta}$ at large $x$.
        Eq. (A1) of Battaglia (2016) prints $-(\beta-\gamma)/\alpha$, which is a typo (N. Battaglia,
        priv. comm., in arXiv:2509.03458, footnote 1). class_sz uses $-(\beta+\gamma)/\alpha$, as here;
        BaryonForge 1.1.0 (``GasDensity``) uses the printed form, which has about 1/3 fewer electrons
        within $2 R_{200c}$.

        :meth:`density` returns the optical depth per comoving Mpc,
        $\sigma_T\, a\, \rho_{\rm gas} / (\mu_e m_p)$ [Mpc$^{-1}$], so that :meth:`projected`
        is the optical depth $\tau$ through the halo (dimensionless) and :meth:`fourier` is in Mpc$^2$.

        Constructor args:

           - ``fit`` (str): ``'AGN'`` (AGN feedback, default) or ``'SH'`` (shock heating).
           - ``truncate`` (float): truncation radius in units of $R_{200c}$ (default 2).
           - ``mu_e`` (float): mean molecular weight per free electron (default 1.14).
           - ``fb`` (float or None): baryon fraction. If None, uses ``ombh2 / (ombh2 + omch2)``
             from ``cosmo.params`` (consistent with ``Cosmology.rhom_comoving``, which also omits
             massive neutrinos).
        """
        if fit not in self.fits:
            raise ValueError(f"Battaglia: fit={fit!r} not recognized (expected one of {list(self.fits)})")

        self.fit = fit
        self.truncate = float(truncate)
        self.mu_e = float(mu_e)
        self.fb = fb

    def __repr__(self):
        return f'Battaglia(fit={self.fit!r}, truncate={self.truncate}, mu_e={self.mu_e}, fb={self.fb})'

    def baryon_fraction(self, cosmo):
        """Returns $f_b$: the constructor's ``fb``, or ``ombh2 / (ombh2 + omch2)``."""
        if self.fb is not None:
            return float(self.fb)
        p = cosmo.params
        return p.ombh2 / (p.ombh2 + p.omch2)

    def r_t(self, cosmo, *, logm, z):
        """Comoving truncation radius ``truncate * R_200c`` [Mpc]."""
        return self.truncate * r200c(cosmo, logm=logm, z=z)

    def density(self, cosmo, *, r, logm, z):
        """Optical depth per comoving Mpc at comoving radius ``r`` [Mpc]; ``r`` and ``logm`` broadcast, ``z`` is a scalar."""
        (A0, m0, z0), (A1, m1, z1), (A2, m2, z2) = self.fits[self.fit]
        lm, zp = np.asarray(logm, dtype=float), 1.0 + float(z)
        m = 10.0 ** (lm - 14.0)
        rho0 = A0 * m**m0 * zp**z0
        alpha = A1 * m**m1 * zp**z1
        beta = A2 * m**m2 * zp**z2

        x = np.asarray(r, dtype=float) / r200c(cosmo, logm=lm, z=z)
        y = x / 0.5          # x / x_c; below, gamma = -0.2 and -(beta - 0.2)/alpha = -(beta + gamma)/alpha
        rho_gas = self.baryon_fraction(cosmo) * rho_crit(cosmo, z=z) * rho0 * y**-0.2 * (1.0 + y**alpha) ** (-(beta - 0.2) / alpha)
        tau = _SIGMA_T / (self.mu_e * _M_P) * rho_gas / zp            # sigma_T a rho_gas / (mu_e m_p)
        return np.where(x <= self.truncate * (1.0 + 1.0e-9), tau, 0.0)
