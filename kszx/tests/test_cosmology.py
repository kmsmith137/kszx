import functools

import numpy as np

from ..Cosmology import Cosmology


@functools.cache
def _cosmo():
    return Cosmology('planck18+bao')


def test_frsd():
    """frsd() (fitting function) against the growth rate of D(z), which comes from CAMB's P(k,z) at k ~ 0.1 Mpc^{-1}.

    With massive neutrinos in Omega_m the agreement is <= 0.08% for z <= 1 (planck18+bao); without them it was 0.27%.
    """

    print('test_frsd(): start')
    cosmo = _cosmo()
    eps = 1.0e-3

    for z in [ 0.0, 0.25, 0.5, 0.75, 1.0 ]:
        zm, zp = max(z-eps, 0.0), z+eps
        f = -(1+z) * (np.log(cosmo.D(z=zp, z0norm=True)) - np.log(cosmo.D(z=zm, z0norm=True))) / (zp-zm)
        assert abs(cosmo.frsd(z=z) / f - 1.0) < 1.5e-3

    print('test_frsd(): pass')
