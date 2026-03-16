#!/usr/bin/env python
"""
Estimate the halo power spectrum P_hh(k) from a Quijote FoF catalog at z=0,
and compare to the simple bias model  P_hh(k) = bg^2 * P_mm(k) + 1/nbar.

Usage:
    conda run -n kszx python scripts/quijote_halo_pk.py

Data is auto-downloaded via Globus on first run (requires Globus Connect Personal).
"""

import numpy as np
import matplotlib.pyplot as plt
import kszx

# --- Parameters ---
sim_type = 'fiducial'
realization = 42
redshift = 0.0
bg = 1.5       # linear halo bias (placeholder, to be fine-tuned)
kmax = 0.3     # Mpc^{-1}

# Quijote fiducial cosmology: h = 0.6711, box = 1 h^{-1} Gpc
h = 0.6711
Lbox = 1000.0 / h   # Mpc

# Grid: 256^3 gives k_nyq ~ 0.54 Mpc^{-1}, comfortably above kmax
npix = 256
pixsize = Lbox / npix

# --- Read halo catalog ---
print('Reading halo catalog...')
cat = kszx.quijote.read_halos(sim_type, realization, redshift)

nbar = cat.size / Lbox**3
print(f'Nhalos = {cat.size}')
print(f'nbar   = {nbar:.4e} Mpc^-3')
print(f'1/nbar = {1/nbar:.1f} Mpc^3')

# --- Grid halos and estimate P(k) ---
box = kszx.Box(npix=(npix, npix, npix), pixsize=pixsize)

points = np.column_stack([cat.x, cat.y, cat.z])

# Weight each halo by 1/nbar so the gridded field has units of overdensity+1,
# and the estimated P(k) directly gives P_hh(k) in Mpc^3.
fk = kszx.grid_points(box, points, kernel='cubic', fft=True,
                       compensate=True, periodic=True, wscal=1.0/nbar)

kfund = 2 * np.pi / Lbox
dk = 2 * kfund
kbin_edges = np.arange(kfund, kmax + dk, dk)

pk_halo = kszx.estimate_power_spectrum(box, fk, kbin_edges)
k_centers = 0.5 * (kbin_edges[:-1] + kbin_edges[1:])

# --- Read matter P(k) from the same realization ---
print('Reading matter power spectrum...')
pk_data = kszx.quijote.read_pk(sim_type, realization, redshift, )

# Interpolate matter Pk onto our k-bin centers
pk_mm = np.interp(k_centers, pk_data['k'], pk_data['pk'])

# --- Fit bg on large scales (k < 0.1) where linear bias should hold ---
fit_mask = k_centers < 0.1
bg_fit = np.sqrt(np.mean((pk_halo[fit_mask] - 1.0/nbar) / pk_mm[fit_mask]))
print(f'Best-fit bg (k < 0.1) = {bg_fit:.3f}  (initial guess was {bg})')
bg = bg_fit

# --- Model ---
pk_model = bg**2 * pk_mm + 1.0 / nbar

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 7),
    gridspec_kw={'height_ratios': [3, 1]}, sharex=True
)

ax1.loglog(k_centers, pk_halo, 'k.', ms=3, label=r'$P_{hh}(k)$ measured')
ax1.loglog(k_centers, pk_model, 'r-', lw=1.5,
           label=rf'$b_g^2 P_{{mm}} + 1/\bar{{n}}$  ($b_g={bg:.2f}$)')
ax1.loglog(k_centers, bg**2 * pk_mm, 'b--', lw=1, alpha=0.5,
           label=rf'$b_g^2 P_{{mm}}$')
ax1.axhline(1.0 / nbar, color='gray', ls=':', lw=1,
            label=rf'$1/\bar{{n}} = {1/nbar:.0f}$ Mpc$^3$')
ax1.set_ylabel(r'$P(k)$ [Mpc$^3$]')
ax1.set_title(f'Quijote {sim_type} r={realization} z={redshift}')
ax1.legend(fontsize=9)
ax1.set_xlim(kbin_edges[0], kmax)

ratio = pk_halo / pk_model
ax2.plot(k_centers, ratio, 'k.', ms=3)
ax2.axhline(1.0, color='r', ls='-', lw=1)
ax2.set_xlabel(r'$k$ [Mpc$^{-1}$]')
ax2.set_ylabel(r'$P_{hh} / P_{\rm model}$')
ax2.set_ylim(0.8, 1.2)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig('quijote_halo_pk.pdf')
print('Saved quijote_halo_pk.pdf')
plt.show()
