"""Compare kszx.quijote.read_halos() with Pylians3 readfof.

Requires: Pylians3 installed, Quijote FoF data on disk.

Usage:
    python scripts/test_pylians3_equivalence/test_fof.py
    python scripts/test_pylians3_equivalence/test_fof.py --sim_type Om_p --realization 42 --redshift 1.0
"""

import argparse
import numpy as np
import kszx
import readfof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_type', default='fiducial')
    parser.add_argument('--realization', type=int, default=0)
    parser.add_argument('--redshift', type=float, default=0.0)
    args = parser.parse_args()

    sim_type = args.sim_type
    realization = args.realization
    redshift = args.redshift
    snapnum = kszx.quijote._snap_num(redshift)

    print(f'Testing FoF: sim_type={sim_type}, realization={realization}, redshift={redshift}')

    # --- kszx ---
    cat = kszx.quijote.read_halos(sim_type, realization, redshift)

    # --- Pylians3 ---
    snapdir = kszx.quijote._quijote_path(f'Halos/FoF/{sim_type}/{realization}')
    FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False, swap=False, SFR=False, read_IDs=False)

    # Pylians3 returns raw Gadget units; convert to match kszx conventions
    h = kszx.quijote._get_h(sim_type)
    pos_pylians = FoF.GroupPos / (1e3 * h)                      # kpc/h -> Mpc
    mass_pylians = FoF.GroupMass * 1e10 / h                     # 10^10 M_sun/h -> M_sun
    vel_pylians = FoF.GroupVel * (1 + redshift) / 299792.458    # v_pec*a -> v/c
    npart_pylians = FoF.GroupLen

    # --- Compare ---
    assert len(cat.x) == len(pos_pylians), f'Count mismatch: {len(cat.x)} vs {len(pos_pylians)}'
    print(f'  Halo count: {len(cat.x)}')

    np.testing.assert_allclose(cat.x, pos_pylians[:, 0], rtol=0, atol=0, err_msg='x mismatch')
    np.testing.assert_allclose(cat.y, pos_pylians[:, 1], rtol=0, atol=0, err_msg='y mismatch')
    np.testing.assert_allclose(cat.z, pos_pylians[:, 2], rtol=0, atol=0, err_msg='z mismatch')
    np.testing.assert_allclose(cat.vx, vel_pylians[:, 0], rtol=0, atol=0, err_msg='vx mismatch')
    np.testing.assert_allclose(cat.vy, vel_pylians[:, 1], rtol=0, atol=0, err_msg='vy mismatch')
    np.testing.assert_allclose(cat.vz, vel_pylians[:, 2], rtol=0, atol=0, err_msg='vz mismatch')
    np.testing.assert_allclose(cat.mass, mass_pylians, rtol=0, atol=0, err_msg='mass mismatch')
    np.testing.assert_array_equal(cat.npart, npart_pylians, err_msg='npart mismatch')

    print('PASS: FoF halo catalog matches Pylians3')


if __name__ == '__main__':
    main()
