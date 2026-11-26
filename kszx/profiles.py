r"""Radial profiles of halos: electron gas (Battaglia), kSZ optical depth, tSZ, and a truncated NFW.

Conventions:

  - Halo masses are $M_{200c}$ in Msol, passed as ``logm`` $= \log_{10}(M_{200c}/M_\odot)$.
  - Radii are comoving Mpc and wavenumbers comoving Mpc$^{-1}$ (no h-units, as in the rest of kszx).
  - The only cosmology needed is a :class:`~kszx.Cosmology`: its ``H(z=...)`` and the parameters
    ``ombh2``, ``omch2``, ``h`` (and ``cosmo.omnuh2``). There is no pyccl dependency.
  - In profile methods, ``cosmo`` is positional and everything else is a keyword (as in
    :class:`~kszx.Cosmology`). ``z`` is a scalar; ``r`` (or ``R``) and ``logm`` broadcast.

Every profile is a :class:`HaloProfile`: a spherical ``density(r)`` truncated at a comoving radius
``r_t``, with its line-of-sight projection ``projected(R)`` and its 3-d Fourier transform ``fourier(k)``.
A halo at comoving distance $\chi$ subtends ``projected(chi * theta)`` at angle $\theta$, and its
harmonic-space profile is ``fourier((l+1/2)/chi) / chi**2``.

Class hierarchy::

    HaloProfile                    density(r) -> projected(R), fourier(k)
    |-- NFW                        unit-mass NFW (satellites, dark matter)
    |-- ElectronProfile            gas model: n_e(r) and P_e(r); density() = comoving n_e
    |   `-- Battaglia              n_e from Battaglia (2016), P_e from Battaglia et al. (2012)
    |-- Tau(gas)                   kSZ: optical depth,   projected() = tau
    `-- TSZ(gas, freq=None)        tSZ: Compton y,       projected() = y  (or Delta T [uK] at freq)

For example, ``Tau(Battaglia()).projected(cosmo, R=R, logm=logm, z=z)`` is the optical depth $\tau(R)$,
and ``TSZ(Battaglia(), freq=150.).fourier(cosmo, k=k, logm=logm, z=z)`` is the tSZ profile at 150 GHz.
"""

import numpy as np
from scipy.special import sici


_C_KMS = 299792.458                                   # speed of light [km/s]
_G_MPC_KMS = 4.30091e-9                               # G [Mpc (km/s)^2 / Msol]
_SIGMA_T = 6.6524587158e-29 / 3.08567758149e22**2     # Thomson cross section [Mpc^2]
_M_P = 1.67262192369e-27 / 1.988409870698051e30       # proton mass [Msol]
_MEC2_KEV = 510.99895                                 # electron rest energy m_e c^2 [keV]
_MSOL_KMS2_KEV = 1.988409870698051e36 / 1.602176634e-16   # 1 Msol (km/s)^2 in keV
_H_OVER_K_GHZ = 6.62607015e-34 / 1.380649e-23 * 1.0e9     # h / k_B [K/GHz]
_T_CMB = 2.7255                                       # CMB temperature [K] (CAMB default)


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


def tsz_spectral_factor(freq, *, T_cmb=_T_CMB):
    r"""Returns the non-relativistic tSZ spectral function $g(x) = x \coth(x/2) - 4$, $x = h\nu / k_B T_{\rm cmb}$.

    The tSZ temperature shift is $\Delta T / T_{\rm cmb} = g(x)\, y$: $g \to -2$ for $\nu \to 0$,
    $g = 0$ at $\nu \approx 217.5$ GHz, and $g(150\ {\rm GHz}) = -0.953$.

    Function args:

       - ``freq`` (float or array): frequency [GHz].
       - ``T_cmb`` (float): CMB temperature [K] (default 2.7255).
    """
    x = _H_OVER_K_GHZ * np.asarray(freq, dtype=float) / T_cmb
    return x / np.tanh(0.5 * x) - 4.0


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

        Gauss-Legendre in $\ln l$ on $[l_0, L]$ with $l_0 = 10^{-9} L$, so the core is resolved at any $R$,
        plus the midpoint rule on $[0, l_0]$ (where $\rho$ is constant to $O(l_0^2/R^2)$). Against adaptive
        quadrature, the relative error is $\lesssim 10^{-11}$ for the profiles in this module, for
        ``truncate`` up to 100 and any $R$ (tested down to $10^{-6} r_t$).
        ``R`` is clipped below at $10^{-6} r_t$, and the result is zero for $R \ge r_t$.
        """
        lm = np.asarray(logm, dtype=float)
        R, rt, lm = np.broadcast_arrays(np.asarray(R, dtype=float), self.r_t(cosmo, logm=lm, z=z), lm)
        R = np.maximum(R, 1.0e-6 * rt)
        L = np.sqrt(np.maximum(rt**2 - R**2, 0.0))
        s, w = np.polynomial.legendre.leggauss(n_l)
        a = 0.5 * np.log(1.0e9)
        l = L[..., None] * np.exp(a * (s - 1.0))
        rho = self.density(cosmo, r=np.hypot(R[..., None], l), logm=lm[..., None], z=z)
        l0 = 1.0e-9 * L
        rho0 = self.density(cosmo, r=np.hypot(R, 0.5 * l0), logm=lm, z=z)
        return 2.0 * (a * (rho * l * w).sum(axis=-1) + l0 * rho0)

    def fourier(self, cosmo, *, k, logm, z):
        r"""Fourier transform $u(k) = 4\pi \int_0^{r_t} r^2 \rho(r)\, \sin(kr)/(kr)\, dr$.

        Function args:

           - ``k`` (float or array): comoving wavenumber [Mpc$^{-1}$].
           - ``logm`` (float or array): $\log_{10}(M_{200c}/M_\odot)$.
           - ``z`` (float): redshift.

        Returns an array of shape ``logm.shape + k.shape``. Units are those of ``density()`` times Mpc$^3$.

        Quadrature: 16-point Gauss-Legendre on at least 64 equal segments of $[0, r_t]$, each
        shorter than 1/3 of a period of $\sin(kr)$ at the largest $k$. The first segment is split
        geometrically (4 per decade, down to $10^{-4}$ of its length), so the core is resolved
        however large $r_t$ is. Against adaptive (QAWO) quadrature, the relative error is
        $\lesssim 10^{-11}$ for the profiles in this module, for ``truncate`` up to 100 and
        $k \le 100$ Mpc$^{-1}$.
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
            edges = np.concatenate(([0.0], h * np.logspace(-4.0, 0.0, 17)[:-1], np.linspace(h, rt, ns)))
            a, d = edges[:-1, None], 0.5 * np.diff(edges)[:, None]
            r = (a + d * (x + 1.0)).ravel()
            f = (d * w).ravel() * r**2 * self.density(cosmo, r=r, logm=m, z=z)
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


class ElectronProfile(HaloProfile):
    r"""Base class for a model of the ionized gas in halos: electron density $n_e(r)$ and pressure $P_e(r)$.

    Subclasses define :meth:`r_t`, :meth:`n_e` and :meth:`P_e`. The observables are built from a gas
    model by wrapping it: :class:`Tau` (kSZ) uses $n_e$, and :class:`TSZ` (tSZ) uses $P_e$.

    As a :class:`HaloProfile`, an ElectronProfile is the electron number density itself:
    :meth:`density` is the comoving density $a^3 n_e$ [Mpc$^{-3}$], so ``fourier(k=0)`` is the number
    of electrons in the halo and ``projected(R)`` is the comoving electron column density [Mpc$^{-2}$].
    """

    def n_e(self, cosmo, *, r, logm, z):
        """Physical electron number density [Mpc$^{-3}$] at comoving radius ``r`` [Mpc]. Must be defined by subclass."""
        raise NotImplementedError(f'{type(self).__name__}.n_e() is not defined')

    def P_e(self, cosmo, *, r, logm, z):
        r"""Physical electron pressure $n_e k_B T_e$ [keV Mpc$^{-3}$] at comoving radius ``r`` [Mpc]. Must be defined by subclass.

        (So ``P_e / n_e`` is the electron temperature $k_B T_e$ in keV.)
        """
        raise NotImplementedError(f'{type(self).__name__}.P_e() is not defined')

    def density(self, cosmo, *, r, logm, z):
        """Comoving electron number density $a^3 n_e$ [Mpc$^{-3}$] at comoving radius ``r`` [Mpc]."""
        return self.n_e(cosmo, r=r, logm=logm, z=z) / (1.0 + float(z))**3


class Battaglia(ElectronProfile):
    # (A, alpha_M, alpha_z) for each of rho_0, alpha, beta (Battaglia 2016, arXiv:1607.02442, Table 2, Delta = 200).
    density_fits = {
        'AGN': ((4.0e3, 0.29, -0.66), (0.88, -0.03, 0.19), (3.83, 0.04, -0.025)),
        'SH': ((1.9e4, 0.09, -0.95), (0.70, -0.017, 0.27), (4.43, 0.005, 0.037)),
    }

    # (A, alpha_M, alpha_z) for each of P_0, x_c, beta (Battaglia et al. 2012, arXiv:1109.3711, Table 1, Delta = 200, AGN feedback).
    pressure_fit = ((18.1, 0.154, -0.758), (0.497, -0.00865, 0.731), (4.35, 0.0393, 0.415))

    def __init__(self, fit='AGN', truncate=2.0, mu_e=1.14, fb=None, X_H=0.76):
        r"""Electron gas in halos: density from Battaglia (2016), pressure from Battaglia et al. (2012).

        Use :class:`Tau` (``Tau(Battaglia())``) for the kSZ optical depth and :class:`TSZ` for the tSZ.
        Both fits are on $M = M_{200c}$ (in Msol), with $x = r / R_{200c}$, and each parameter scales as
        $A\, (M / 10^{14} M_\odot)^{\alpha_M}\, (1+z)^{\alpha_z}$. Both profiles are truncated at
        ``truncate`` $\times R_{200c}$.

        **Density** (arXiv:1607.02442, Eqs. A1-A2 and Table 2; ``fit`` = AGN or SH):

        $$\rho_{\rm gas}(r) = f_b\, \rho_c(z)\, \rho_0 \Big(\frac{x}{x_c}\Big)^\gamma
        \Big[1 + \Big(\frac{x}{x_c}\Big)^\alpha\Big]^{-(\beta+\gamma)/\alpha},
        \qquad x_c = 0.5, \quad \gamma = -0.2, \qquad n_e = \rho_{\rm gas} / (\mu_e m_p).$$

        The second exponent is $-(\beta+\gamma)/\alpha$, so that $\rho \propto x^{-\beta}$ at large $x$.
        Eq. (A1) of Battaglia (2016) prints $-(\beta-\gamma)/\alpha$, which is a typo (N. Battaglia,
        priv. comm., in arXiv:2509.03458, footnote 1). class_sz uses $-(\beta+\gamma)/\alpha$, as here;
        BaryonForge 1.1.0 (``GasDensity``) uses the printed form, which has about 1/3 fewer electrons
        within $2 R_{200c}$.

        **Pressure** (arXiv:1109.3711, Eqs. 10-11 and Table 1; AGN feedback, the only $\Delta = 200$ fit):

        $$P_{\rm th}(r) = P_{200}\, P_0 \Big(\frac{x}{x_c}\Big)^{-0.3} \Big[1 + \frac{x}{x_c}\Big]^{-\beta},
        \qquad P_{200} = \frac{G M_{200}\, 200 \rho_c(z)\, f_b}{2 R_{200}},
        \qquad P_e = \frac{2 (X_H + 1)}{5 X_H + 3} P_{\rm th}$$

        with $R_{200}$ physical. $P_e = 0.518\, P_{\rm th}$ for $X_H = 0.76$ (B12, footnote 3), as in class_sz.
        The pressure fit does not depend on ``fit``: B12 has no shock-heating fit at $\Delta = 200$.

        Constructor args:

           - ``fit`` (str): density fit, ``'AGN'`` (AGN feedback, default) or ``'SH'`` (shock heating).
           - ``truncate`` (float): truncation radius in units of $R_{200c}$ (default 2).
           - ``mu_e`` (float): mean molecular weight per free electron (default 1.14).
           - ``fb`` (float or None): baryon fraction. If None, uses $\Omega_b/\Omega_m$ =
             ``ombh2 / (ombh2 + omch2 + omnuh2)``, with CAMB's massive-neutrino density ``cosmo.omnuh2``
             (as in CAMB's ``omegam``, CCL, class_sz and BaryonForge).
           - ``X_H`` (float): hydrogen mass fraction, used for $P_e / P_{\rm th}$ (default 0.76).
        """
        if fit not in self.density_fits:
            raise ValueError(f"Battaglia: fit={fit!r} not recognized (expected one of {list(self.density_fits)})")

        self.fit = fit
        self.truncate = float(truncate)
        self.mu_e = float(mu_e)
        self.fb = fb
        self.X_H = float(X_H)

    def __repr__(self):
        return f'Battaglia(fit={self.fit!r}, truncate={self.truncate}, mu_e={self.mu_e}, fb={self.fb}, X_H={self.X_H})'

    def baryon_fraction(self, cosmo):
        """Returns $f_b$: the constructor's ``fb``, or ``ombh2 / (ombh2 + omch2 + omnuh2)``."""
        if self.fb is not None:
            return float(self.fb)
        p = cosmo.params
        omnuh2 = getattr(cosmo, 'omnuh2', 0.0)   # 0 for a Cosmology pickled before omnuh2 was added
        return p.ombh2 / (p.ombh2 + p.omch2 + omnuh2)

    def r_t(self, cosmo, *, logm, z):
        """Comoving truncation radius ``truncate * R_200c`` [Mpc]."""
        return self.truncate * r200c(cosmo, logm=logm, z=z)

    def _fit_params(self, fit, logm, z):
        m, zp = 10.0 ** (np.asarray(logm, dtype=float) - 14.0), 1.0 + float(z)
        return [ A * m**am * zp**az for (A, am, az) in fit ]

    def _x(self, cosmo, r, logm, z):
        return np.asarray(r, dtype=float) / r200c(cosmo, logm=logm, z=z)

    def n_e(self, cosmo, *, r, logm, z):
        """Physical electron number density [Mpc$^{-3}$] at comoving radius ``r`` [Mpc]; ``r`` and ``logm`` broadcast."""
        rho0, alpha, beta = self._fit_params(self.density_fits[self.fit], logm, z)
        x = self._x(cosmo, r, logm, z)
        y = x / 0.5          # x / x_c; below, gamma = -0.2 and -(beta - 0.2)/alpha = -(beta + gamma)/alpha
        rho_gas = self.baryon_fraction(cosmo) * rho_crit(cosmo, z=z) * rho0 * y**-0.2 * (1.0 + y**alpha) ** (-(beta - 0.2) / alpha)
        return np.where(x <= self.truncate * (1.0 + 1.0e-9), rho_gas / (self.mu_e * _M_P), 0.0)

    def P_e(self, cosmo, *, r, logm, z):
        """Physical electron pressure [keV Mpc$^{-3}$] at comoving radius ``r`` [Mpc]; ``r`` and ``logm`` broadcast."""
        P0, xc, beta = self._fit_params(self.pressure_fit, logm, z)
        M, zp = 10.0 ** np.asarray(logm, dtype=float), 1.0 + float(z)
        rhoc = rho_crit(cosmo, z=z)
        R = np.cbrt(3.0 * M / (800.0 * np.pi * rhoc))                                     # physical R_200c [Mpc]
        P200 = _G_MPC_KMS * M * 200.0 * rhoc * self.baryon_fraction(cosmo) / (2.0 * R)   # [Msol (km/s)^2 / Mpc^3]
        x = np.asarray(r, dtype=float) / (R * zp)
        y = x / xc
        P_th = _MSOL_KMS2_KEV * P200 * P0 * y**-0.3 * (1.0 + y) ** -beta
        P_e = 2.0 * (self.X_H + 1.0) / (5.0 * self.X_H + 3.0) * P_th
        return np.where(x <= self.truncate * (1.0 + 1.0e-9), P_e, 0.0)


####################################################################################################


class Tau(HaloProfile):
    def __init__(self, gas):
        r"""kSZ optical-depth profile of an :class:`ElectronProfile` ``gas`` (e.g. ``Tau(Battaglia())``).

        :meth:`density` is the optical depth per comoving Mpc, $\sigma_T\, a\, n_e$ [Mpc$^{-1}$],
        so :meth:`projected` is the optical depth $\tau(R)$ (dimensionless) and :meth:`fourier` is in Mpc$^2$.
        """
        if not isinstance(gas, ElectronProfile):
            raise TypeError(f'Tau: expected an ElectronProfile (e.g. Battaglia()), got {gas!r}')
        self.gas = gas

    def __repr__(self):
        return f'Tau({self.gas!r})'

    def r_t(self, cosmo, *, logm, z):
        """Comoving truncation radius of the gas profile [Mpc]."""
        return self.gas.r_t(cosmo, logm=logm, z=z)

    def density(self, cosmo, *, r, logm, z):
        """Optical depth per comoving Mpc at comoving radius ``r`` [Mpc]; ``r`` and ``logm`` broadcast, ``z`` is a scalar."""
        return _SIGMA_T * self.gas.n_e(cosmo, r=r, logm=logm, z=z) / (1.0 + float(z))


class TSZ(HaloProfile):
    def __init__(self, gas, freq=None, T_cmb=_T_CMB):
        r"""Thermal SZ profile of an :class:`ElectronProfile` ``gas`` (e.g. ``TSZ(Battaglia(), freq=150.)``).

        The Compton-y parameter is $y = (\sigma_T / m_e c^2) \int P_e\, dl_{\rm phys}$. If ``freq`` is None,
        :meth:`density` is $y$ per comoving Mpc, $(\sigma_T / m_e c^2)\, a\, P_e$ [Mpc$^{-1}$], so
        :meth:`projected` is $y(R)$ (dimensionless) and :meth:`fourier` is in Mpc$^2$.

        If ``freq`` is given, everything is multiplied by $T_{\rm cmb}\, g(x)$ (see
        :func:`tsz_spectral_factor`), so :meth:`projected` is the CMB temperature shift $\Delta T(R)$ in
        $\mu$K (thermodynamic units, as in kszx CMB maps) and :meth:`fourier` is in $\mu$K Mpc$^2$.
        This is the non-relativistic tSZ at a single frequency: no relativistic corrections and no
        bandpass integration (for a band, pass its effective tSZ frequency).

        Constructor args:

           - ``gas`` (:class:`ElectronProfile`): the gas model, e.g. ``Battaglia()``.
           - ``freq`` (float or None): observing frequency [GHz]. None (default) returns Compton $y$.
           - ``T_cmb`` (float): CMB temperature [K] (default 2.7255).
        """
        if not isinstance(gas, ElectronProfile):
            raise TypeError(f'TSZ: expected an ElectronProfile (e.g. Battaglia()), got {gas!r}')
        self.gas = gas
        self.freq = None if (freq is None) else float(freq)
        self.T_cmb = float(T_cmb)

    def __repr__(self):
        return f'TSZ({self.gas!r}, freq={self.freq}, T_cmb={self.T_cmb})'

    def spectral_factor(self):
        r"""Returns 1 if ``freq`` is None (Compton $y$), else $T_{\rm cmb} g(x)$ in $\mu$K ($\Delta T$ per unit $y$)."""
        if self.freq is None:
            return 1.0
        return 1.0e6 * self.T_cmb * float(tsz_spectral_factor(self.freq, T_cmb=self.T_cmb))

    def r_t(self, cosmo, *, logm, z):
        """Comoving truncation radius of the gas profile [Mpc]."""
        return self.gas.r_t(cosmo, logm=logm, z=z)

    def density(self, cosmo, *, r, logm, z):
        """Compton y (or $\\Delta T$ [$\\mu$K]) per comoving Mpc at comoving radius ``r`` [Mpc]; ``r`` and ``logm`` broadcast."""
        P_e = self.gas.P_e(cosmo, r=r, logm=logm, z=z)
        return self.spectral_factor() * (_SIGMA_T / _MEC2_KEV) * P_e / (1.0 + float(z))
