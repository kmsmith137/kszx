#!/usr/bin/env python
"""
Estimate the matter power spectrum P_mm(k) from a Quijote snapshot at z=0,
and compare to Cosmology.Plin_z0 (CAMB linear theory).

Usage:
    conda activate kszx && source .venv/bin/activate
    python scripts/quijote_matter_pk.py
"""

import numpy as np
import matplotlib.pyplot as plt
import kszx

# --- Parameters ---
sim_type = 'fiducial'
realization = 42
redshift = 0.0
kmax = 0.3     # Mpc^{-1}

# Quijote fiducial cosmology
h = 0.6711
Lbox = 1000.0 / h   # Mpc

# Grid: 256^3 gives k_nyq ~ 0.54 Mpc^{-1}, comfortably above kmax
npix = 256
pixsize = Lbox / npix

# --- Build Cosmology matching Quijote fiducial ---
cosmo = kszx.quijote.cosmology(sim_type)

# --- Read snapshot ---
print('Reading snapshot (this may take a minute)...')
snap = kszx.quijote.read_snapshot(sim_type, realization, redshift, fields=('pos',))
pos = snap['pos']   # shape (Npart, 3), units Mpc
Npart = pos.shape[0]

nbar = Npart / Lbox**3
print(f'Npart  = {Npart}')
print(f'nbar   = {nbar:.4e} Mpc^-3')
print(f'1/nbar = {1/nbar:.1f} Mpc^3')

# --- Grid particles and estimate P(k) ---
box = kszx.Box(npix=(npix, npix, npix), pixsize=pixsize)

# Weight by 1/nbar so P(k) is in standard units (Mpc^3)
fk = kszx.grid_points(box, pos, kernel='cubic', fft=True,
                       compensate=True, periodic=True, wscal=1.0/nbar)

del pos, snap   # free memory

kfund = 2 * np.pi / Lbox
dk = 2 * kfund
kbin_edges = np.arange(kfund, kmax + dk, dk)

pk_measured = kszx.estimate_power_spectrum(box, fk, kbin_edges)
k_centers = 0.5 * (kbin_edges[:-1] + kbin_edges[1:])

# --- Theory curves ---
pk_lin = cosmo.Plin_z0(k_centers)

# Also load the simulation's own CAMB linear Pk for cross-check
pk_camb_sim = kszx.quijote.read_linear_pk(sim_type)
pk_sim_interp = np.interp(k_centers, pk_camb_sim['k'], pk_camb_sim['pk'])

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 7),
    gridspec_kw={'height_ratios': [3, 1]}, sharex=True
)

ax1.loglog(k_centers, pk_measured, 'k.', ms=3, label=r'$P_{mm}(k)$ measured')
ax1.loglog(k_centers, pk_lin, 'r-', lw=1.5, label=r'Cosmology.Plin\_z0 (CAMB)')
ax1.loglog(k_centers, pk_sim_interp, 'b--', lw=1, alpha=0.6,
           label=r'read\_linear\_pk (sim ICs)')
ax1.axhline(1.0 / nbar, color='gray', ls=':', lw=1,
            label=rf'$1/\bar{{n}} = {1/nbar:.1f}$ Mpc$^3$')
ax1.set_ylabel(r'$P(k)$ [Mpc$^3$]')
ax1.set_title(f'Quijote {sim_type} r={realization} z={redshift}')
ax1.legend(fontsize=9)
ax1.set_xlim(kbin_edges[0], kmax)

ratio = pk_measured / pk_lin
ratio_sim = pk_measured / pk_sim_interp
ax2.plot(k_centers, ratio, 'r.', ms=3, label='measured / Cosmology.Plin')
ax2.plot(k_centers, ratio_sim, 'b.', ms=3, alpha=0.5, label='measured / sim linear Pk')
ax2.axhline(1.0, color='gray', ls='-', lw=0.5)
ax2.set_xlabel(r'$k$ [Mpc$^{-1}$]')
ax2.set_ylabel(r'$P_{mm} / P_{\rm lin}$')
ax2.set_ylim(0.8, 1.5)
ax2.set_xscale('log')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('quijote_matter_pk.pdf')
print('Saved quijote_matter_pk.pdf')
plt.show()
