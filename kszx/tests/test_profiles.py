import functools

import camb
import numpy as np
from scipy.integrate import quad
from scipy.special import j0

from .. import profiles
from ..Cosmology import Cosmology


@functools.cache
def _cosmo():
    return Cosmology('planck18+bao')


def _hankel(prof, cosmo, k, logm, z, n=4000):
    """2 pi int_0^{r_t} R tau(R) J_0(kR) dR, in R = r_t (1 - (1-t)^2), which is smooth at the sqrt edge."""
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (x + 1.0)
    rt = float(prof.r_t(cosmo, logm=logm, z=z))
    R, wR = rt * (1.0 - (1.0 - t)**2), w * rt * (1.0 - t)
    return 2.0 * np.pi * (j0(np.outer(k, R)) @ (wR * R * prof.projected(cosmo, R=R, logm=logm, z=z)))


def test_gas_fourier():
    """fourier() against the Hankel transform of projected(), for Battaglia (n_e), Tau and TSZ."""

    print('test_gas_fourier(): start')
    cosmo = _cosmo()
    gas, gas_sh = profiles.Battaglia(), profiles.Battaglia(fit='SH', truncate=3.0)

    for prof in [ gas, profiles.Tau(gas), profiles.Tau(gas_sh), profiles.TSZ(gas), profiles.TSZ(gas_sh, freq=90.0) ]:
        for logm, z in [ (12.5, 1.0), (14.0, 0.5), (15.0, 0.1) ]:
            rt = float(prof.r_t(cosmo, logm=logm, z=z))
            k = np.array([1.0e-4, 1.0, 3.0, 10.0, 30.0]) / rt
            u = prof.fourier(cosmo, k=k, logm=logm, z=z)
            hankel = _hankel(prof, cosmo, k, logm, z)
            assert u.shape == k.shape
            assert np.all(np.abs(hankel - u) < 1.0e-6 * np.abs(u[0]))

    print('test_gas_fourier(): pass')


def test_battaglia_pressure():
    """Battaglia.P_e() against class_sz's normalization, and Tau/TSZ against n_e/P_e."""

    print('test_battaglia_pressure(): start')
    cosmo = _cosmo()
    gas = profiles.Battaglia()
    fb, h = gas.baryon_fraction(cosmo), cosmo.params.h
    mpc_cm = 3.08567758149e24

    for logm, z in [ (13.0, 0.3), (15.0, 1.0) ]:
        # class_sz: P_e = 2.61051e-18 (100 h E(z))^2 (M / R_200,phys) f_b P(x) in eV/cm^3 (with P_e = 0.5176 P_th).
        M, R = 10.0**logm, float(profiles.r200c(cosmo, logm=logm, z=z)) / (1+z)
        E = float(cosmo.H(z=z) / cosmo.H(z=0.0))
        P0 = 18.1 * (M/1e14)**0.154 * (1+z)**-0.758
        xc = 0.497 * (M/1e14)**-0.00865 * (1+z)**0.731
        beta = 4.35 * (M/1e14)**0.0393 * (1+z)**0.415
        x = np.array([0.05, 0.3, 1.0, 1.9])
        P_class_sz = 2.61051e-18 * (100*h*E)**2 * (M/R) * fb * P0 * (x/xc)**-0.3 * (1 + x/xc)**-beta
        P_e = gas.P_e(cosmo, r=x*R*(1+z), logm=logm, z=z) * 1.0e3 / mpc_cm**3     # keV/Mpc^3 -> eV/cm^3
        assert np.all(np.abs(P_e / P_class_sz - 1.0) < 5.0e-4)

        # Tau and TSZ are sigma_T a n_e and (sigma_T / m_e c^2) a P_e.
        r = x * R * (1+z)
        tau = profiles.Tau(gas).density(cosmo, r=r, logm=logm, z=z)
        y = profiles.TSZ(gas).density(cosmo, r=r, logm=logm, z=z)
        assert np.allclose(tau, profiles._SIGMA_T * gas.n_e(cosmo, r=r, logm=logm, z=z) / (1+z), rtol=1e-14)
        assert np.allclose(y, profiles._SIGMA_T / 510.99895 * gas.P_e(cosmo, r=r, logm=logm, z=z) / (1+z), rtol=1e-14)

    print('test_battaglia_pressure(): pass')


def test_tsz_frequency():
    """tsz_spectral_factor() limits, and TSZ(freq=...) = T_cmb g(nu) y in uK."""

    print('test_tsz_frequency(): start')
    cosmo = _cosmo()
    g = profiles.tsz_spectral_factor

    assert abs(g(1.0e-3) + 2.0) < 1.0e-9                     # Rayleigh-Jeans: Delta T / T = -2y
    assert abs(g(217.5074)) < 1.0e-5                          # tSZ null at x = 3.830016
    assert abs(g(150.0) + 0.95333) < 1.0e-5

    gas = profiles.Battaglia()
    R = np.array([0.05, 0.3, 1.0])
    y = profiles.TSZ(gas).projected(cosmo, R=R, logm=14.0, z=0.5)
    for freq in [ 90.0, 150.0, 220.0 ]:
        dT = profiles.TSZ(gas, freq=freq).projected(cosmo, R=R, logm=14.0, z=0.5)
        assert np.allclose(dT, 2.7255e6 * g(freq) * y, rtol=1e-14)

    try:
        profiles.TSZ(profiles.NFW())
        raise AssertionError('TSZ(NFW()) should raise TypeError')
    except TypeError:
        pass

    print('test_tsz_frequency(): pass')


def test_nfw_fourier():
    """NFW.fourier() (closed form) against the quadrature in HaloProfile.fourier(), and the Hankel transform."""

    print('test_nfw_fourier(): start')
    cosmo = _cosmo()
    logm = np.array([12.0, 13.5, 15.0])
    z = 0.7

    for prof in [ profiles.NFW(), profiles.NFW(truncate=2.0), profiles.NFW(truncate=30.0) ]:
        rt = prof.r_t(cosmo, logm=logm, z=z)
        k = np.geomspace(1.0e-3, 100.0, 40)    # Mpc^{-1}, up to k r_t ~ 10^4
        u = prof.fourier(cosmo, k=k, logm=logm, z=z)
        uq = profiles.HaloProfile.fourier(prof, cosmo, k=k, logm=logm, z=z)
        assert u.shape == uq.shape == (3, 40)
        assert np.all(np.abs(uq / u - 1.0) < 1.0e-9)
        assert np.all(np.abs(prof.fourier(cosmo, k=0.0, logm=logm, z=z) - 1.0) < 1.0e-10)

        # Also checks NFW.projected() (inherited from HaloProfile).
        kh = np.array([1.0e-4, 1.0, 10.0]) / rt[1]
        uh = _hankel(prof, cosmo, kh, logm[1], z)
        assert np.all(np.abs(uh - prof.fourier(cosmo, k=kh, logm=logm[1], z=z)) < 1.0e-5)

    print('test_nfw_fourier(): pass')


def test_quadrature_accuracy():
    """projected() and fourier() against adaptive quadrature (scipy.integrate.quad), at large truncation, small R and high k."""

    print('test_quadrature_accuracy(): start')
    cosmo = _cosmo()
    logm, z = 15.0, 0.1

    for prof in [ profiles.Tau(profiles.Battaglia(truncate=30.0)), profiles.TSZ(profiles.Battaglia(fit='SH', truncate=100.0)) ]:
        rt = float(prof.r_t(cosmo, logm=logm, z=z))
        rho = lambda r: float(prof.density(cosmo, r=r, logm=logm, z=z))
        edges = np.concatenate([[0.0], np.geomspace(1.0e-7 * rt, rt, 40)])

        # fourier(): QUADPACK's QAWO (weight='sin') on geometric pieces of [0, r_t].
        k = np.array([0.01, 1.0, 100.0])    # Mpc^{-1}
        u = prof.fourier(cosmo, k=k, logm=logm, z=z)
        for kk, uu in zip(k, u):
            ref = sum(quad(lambda r: r * rho(r), a, b, weight='sin', wvar=kk, epsabs=0, epsrel=1e-11, limit=200)[0]
                      for a, b in zip(edges[:-1], edges[1:]))
            assert abs(uu / (4.0 * np.pi * ref / kk) - 1.0) < 1.0e-9

        # projected(), down to R = 10^{-6} r_t.
        R = np.array([1.0e-6, 1.0e-3, 0.1, 0.9]) * rt
        p = prof.projected(cosmo, R=R, logm=logm, z=z)
        for RR, pp in zip(R, p):
            L = np.sqrt(rt**2 - RR**2)
            ledges = np.concatenate([[0.0], np.geomspace(1.0e-7 * L, L, 40)])
            ref = 2.0 * sum(quad(lambda l: rho(np.hypot(RR, l)), a, b, epsabs=0, epsrel=1e-12, limit=200)[0]
                            for a, b in zip(ledges[:-1], ledges[1:]))
            assert abs(pp / ref - 1.0) < 1.0e-9

    print('test_quadrature_accuracy(): pass')


def test_profile_shapes():
    """Broadcasting of r200c(), density(), projected(), fourier()."""

    print('test_profile_shapes(): start')
    cosmo = _cosmo()
    logm = np.array([[12.0], [13.0], [14.0]])   # shape (3,1)
    R = np.linspace(0.01, 1.0, 5)               # shape (5,)
    assert profiles.r200c(cosmo, logm=logm, z=0.5).shape == (3,1)

    gas = profiles.Battaglia()
    assert gas.n_e(cosmo, r=R, logm=logm, z=0.5).shape == (3,5)
    assert gas.P_e(cosmo, r=R, logm=logm, z=0.5).shape == (3,5)

    for prof in [ gas, profiles.Tau(gas), profiles.TSZ(gas), profiles.TSZ(gas, freq=150.0) ]:
        assert prof.density(cosmo, r=R, logm=logm, z=0.5).shape == (3,5)
        assert prof.projected(cosmo, R=R, logm=logm, z=0.5).shape == (3,5)
        assert prof.fourier(cosmo, k=R, logm=logm[:,0], z=0.5).shape == (3,5)
        assert prof.fourier(cosmo, k=1.0, logm=13.0, z=0.5).shape == ()

        # Zero outside the truncation radius.
        rt = float(prof.r_t(cosmo, logm=13.0, z=0.5))
        assert prof.density(cosmo, r=1.01*rt, logm=13.0, z=0.5) == 0.0
        assert prof.projected(cosmo, R=1.01*rt, logm=13.0, z=0.5) == 0.0

    print('test_profile_shapes(): pass')


def test_baryon_fraction():
    """Battaglia's default f_b is Omega_b / Omega_m with CAMB's Omega_m (which includes massive neutrinos)."""

    print('test_baryon_fraction(): start')
    cosmo = _cosmo()
    p = cosmo.params
    cp = camb.set_params(H0=100*p.h, ombh2=p.ombh2, omch2=p.omch2, mnu=p.mnu, nnu=p.nnu)
    fb = p.ombh2 / p.h**2 / cp.omegam

    assert cosmo.omnuh2 > 0.0
    assert abs(profiles.Battaglia().baryon_fraction(cosmo) / fb - 1.0) < 1.0e-10
    assert profiles.Battaglia(fb=0.15).baryon_fraction(cosmo) == 0.15

    print('test_baryon_fraction(): pass')
