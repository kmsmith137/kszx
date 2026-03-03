"""Compare kszx.quijote.read_snapshot() with Pylians3 readgadget.

Requires: Pylians3 installed, hdf5plugin installed, Quijote snapshot data on disk.

Usage:
    python scripts/test_pylians3_equivalence/test_snapshot.py
    python scripts/test_pylians3_equivalence/test_snapshot.py --sim_type fiducial --realization 0 --redshift 0.0
"""

import argparse
import numpy as np
import kszx
import readgadget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_type', default='fiducial')
    parser.add_argument('--realization', type=int, default=0)
    parser.add_argument('--redshift', type=float, default=0.0)
    parser.add_argument('--ptype', type=int, default=1, help='Particle type (1=CDM, 2=neutrinos)')
    args = parser.parse_args()

    sim_type = args.sim_type
    realization = args.realization
    redshift = args.redshift
    ptype = args.ptype
    snapnum = kszx.quijote._snap_num(redshift)

    print(f'Testing snapshot: sim_type={sim_type}, realization={realization}, redshift={redshift}, ptype={ptype}')

    # --- kszx ---
    data = kszx.quijote.read_snapshot(sim_type, realization, redshift,
                                       fields=('pos', 'vel', 'ids'), ptype=ptype)

    # --- Pylians3 ---
    snapdir = kszx.quijote._quijote_path(f'Snapshots/{sim_type}/{realization}')
    snapshot = f'{snapdir}/snapdir_{snapnum:03d}/snap_{snapnum:03d}'

    header = readgadget.header(snapshot)
    h = header.hubble

    pos_pylians = readgadget.read_block(snapshot, "POS ", [ptype])  # kpc/h
    vel_pylians = readgadget.read_block(snapshot, "VEL ", [ptype])  # km/s (peculiar)
    ids_pylians = readgadget.read_block(snapshot, "ID  ", [ptype])

    # Convert Pylians3 output to kszx conventions
    if header.boxsize > 10000:
        pos_pylians = pos_pylians / (1e3 * h)   # kpc/h -> Mpc
    else:
        pos_pylians = pos_pylians / h            # Mpc/h -> Mpc

    # Pylians3's read_block("VEL ") returns peculiar km/s
    # (it applies the sqrt(a) correction internally); kszx returns v/c
    vel_pylians = vel_pylians / 299792.458

    # --- Compare ---
    n = len(data['pos'])
    print(f'  Particle count: {n}')
    assert n == len(pos_pylians), f'Count mismatch: {n} vs {len(pos_pylians)}'

    np.testing.assert_allclose(data['pos'], pos_pylians, rtol=0, atol=0,
                               err_msg='pos mismatch')
    np.testing.assert_allclose(data['vel'], vel_pylians, rtol=1e-6,
                               err_msg='vel mismatch')
    np.testing.assert_array_equal(data['ids'], ids_pylians, err_msg='ids mismatch')

    print('PASS: Snapshot matches Pylians3')


if __name__ == '__main__':
    main()
