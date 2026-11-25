import functools

import numpy as np
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


def test_battaglia_fourier():
    """Battaglia.fourier() against the Hankel transform of Battaglia.projected()."""

    print('test_battaglia_fourier(): start')
    cosmo = _cosmo()

    for prof in [ profiles.Battaglia(), profiles.Battaglia(fit='SH', truncate=3.0) ]:
        for logm, z in [ (12.5, 1.0), (14.0, 0.5), (15.0, 0.1) ]:
            rt = float(prof.r_t(cosmo, logm=logm, z=z))
            k = np.array([1.0e-4, 1.0, 3.0, 10.0, 30.0]) / rt
            u = prof.fourier(cosmo, k=k, logm=logm, z=z)
            hankel = _hankel(prof, cosmo, k, logm, z)
            assert u.shape == k.shape
            assert np.all(np.abs(hankel - u) < 1.0e-6 * u[0])

    print('test_battaglia_fourier(): pass')


def test_nfw_fourier():
    """NFW.fourier() (closed form) against the quadrature in HaloProfile.fourier(), and the Hankel transform."""

    print('test_nfw_fourier(): start')
    cosmo = _cosmo()
    logm = np.array([12.0, 13.5, 15.0])
    z = 0.7

    for prof in [ profiles.NFW(), profiles.NFW(truncate=2.0) ]:
        rt = prof.r_t(cosmo, logm=logm, z=z)
        k = np.geomspace(1.0e-3, 50.0, 40) / rt.min()
        u = prof.fourier(cosmo, k=k, logm=logm, z=z)
        uq = profiles.HaloProfile.fourier(prof, cosmo, k=k, logm=logm, z=z)
        assert u.shape == uq.shape == (3, 40)
        assert np.all(np.abs(u - uq) < 1.0e-6)
        assert np.all(np.abs(prof.fourier(cosmo, k=0.0, logm=logm, z=z) - 1.0) < 1.0e-10)

        # Also checks NFW.projected() (inherited from HaloProfile).
        kh = np.array([1.0e-4, 1.0, 10.0]) / rt[1]
        uh = _hankel(prof, cosmo, kh, logm[1], z)
        assert np.all(np.abs(uh - prof.fourier(cosmo, k=kh, logm=logm[1], z=z)) < 1.0e-5)

    print('test_nfw_fourier(): pass')


def test_profile_shapes():
    """Broadcasting of r200c(), density(), projected(), fourier()."""

    print('test_profile_shapes(): start')
    cosmo = _cosmo()
    prof = profiles.Battaglia()
    logm = np.array([[12.0], [13.0], [14.0]])   # shape (3,1)
    R = np.linspace(0.01, 1.0, 5)               # shape (5,)

    assert profiles.r200c(cosmo, logm=logm, z=0.5).shape == (3,1)
    assert prof.density(cosmo, r=R, logm=logm, z=0.5).shape == (3,5)
    assert prof.projected(cosmo, R=R, logm=logm, z=0.5).shape == (3,5)
    assert prof.fourier(cosmo, k=R, logm=logm[:,0], z=0.5).shape == (3,5)
    assert prof.fourier(cosmo, k=1.0, logm=13.0, z=0.5).shape == ()

    # Zero outside the truncation radius.
    rt = float(prof.r_t(cosmo, logm=13.0, z=0.5))
    assert prof.density(cosmo, r=1.01*rt, logm=13.0, z=0.5) == 0.0
    assert prof.projected(cosmo, R=1.01*rt, logm=13.0, z=0.5) == 0.0

    print('test_profile_shapes(): pass')
