"""Compare kszx.quijote.read_pk() with direct numpy.loadtxt.

Requires: Quijote power spectrum data on disk.

Usage:
    python scripts/test_pylians3_equivalence/test_pk.py
    python scripts/test_pylians3_equivalence/test_pk.py --sim_type Om_p --realization 42 --redshift 1.0
"""

import argparse
import numpy as np
import kszx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_type', default='fiducial')
    parser.add_argument('--realization', type=int, default=0)
    parser.add_argument('--redshift', type=float, default=0.0)
    args = parser.parse_args()

    sim_type = args.sim_type
    realization = args.realization
    redshift = args.redshift
    h = kszx.quijote._get_h(sim_type)

    print(f'Testing Pk: sim_type={sim_type}, realization={realization}, redshift={redshift}')

    # --- Real-space ---
    pk = kszx.quijote.read_pk(sim_type, realization, redshift, space='real')

    zstr = f'{redshift:g}'
    path = kszx.quijote._quijote_path(f'Pk/{sim_type}/{realization}/Pk_m_z={zstr}.txt')
    raw = np.loadtxt(path)
    k_expected = raw[:, 0] * h         # h/Mpc -> Mpc^-1
    pk_expected = raw[:, 1] / h**3     # (Mpc/h)^3 -> Mpc^3

    np.testing.assert_allclose(pk['k'], k_expected, rtol=1e-10, err_msg='k mismatch')
    np.testing.assert_allclose(pk['pk'], pk_expected, rtol=1e-10, err_msg='P(k) mismatch')
    print('  PASS: Real-space power spectrum unit conversion correct')

    # --- Redshift-space ---
    try:
        pk_rsd = kszx.quijote.read_pk(sim_type, realization, redshift, space='redshift')
        path_rsd = kszx.quijote._quijote_path(
            f'Pk/{sim_type}/{realization}/Pk2D_m_z={zstr}_axis=z.txt')
        raw_rsd = np.loadtxt(path_rsd)

        np.testing.assert_allclose(pk_rsd['k'], raw_rsd[:, 0] * h, rtol=1e-10,
                                   err_msg='k mismatch (rsd)')
        np.testing.assert_allclose(pk_rsd['pk0'], raw_rsd[:, 1] / h**3, rtol=1e-10,
                                   err_msg='P0(k) mismatch')
        np.testing.assert_allclose(pk_rsd['pk2'], raw_rsd[:, 2] / h**3, rtol=1e-10,
                                   err_msg='P2(k) mismatch')
        np.testing.assert_allclose(pk_rsd['pk4'], raw_rsd[:, 3] / h**3, rtol=1e-10,
                                   err_msg='P4(k) mismatch')
        print('  PASS: Redshift-space power spectrum unit conversion correct')
    except RuntimeError:
        print('  SKIP: Redshift-space Pk file not found on disk')

    print('PASS: Power spectrum tests complete')


if __name__ == '__main__':
    main()
