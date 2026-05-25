"""
Orientation sweep at a fixed position — what the spherical EEG model says.

The original intent: turn the Q-B realistic-cortex radial-dipole failure
(~62mm vertex error) into a continuum, expecting tangential = easy →
radial = hard.

What we actually find here: the spherical 4-shell EEG model does NOT
reproduce that observability gradient. At every depth tested (10–78mm),
the radial dipole produces ~1.6–2.0x MORE scalp signal than the tangential
dipole of the same moment magnitude. This is the (n+1) prefactor on the
radial term in the de Munck series (see Mosher 1999, eq. 22; matches
spherical_forward.py:182). The "radial dipoles are weak for EEG" intuition
is often imported from MEG — Sarvas (1987) showed radial dipoles produce
zero magnetic field outside a sphere — but for spherical EEG, the radial
term carries an extra (n+1) factor.

Localization error in this script therefore mildly DECREASES as the dipole
rotates toward radial, tracking the SNR rise. The Q-B vertex failure must
be specific to realistic anatomy (sulcal geometry, BEM mesh, anatomical
skull thickness, vertex cortical-normal direction not exactly aligned with
the head-center radial). To turn that anecdote into a continuum the right
venue is BEM, not the analytic sphere.

Setup:
  - Free-orientation leadfield (3 cols/source). Inverse basis can
    represent any dipole orientation; isolates *scalp observability* of
    orientation from any basis-mismatch artefact.
  - Fixed dipole position on a 10mm grid (floor localization error = 0).
  - Noise std calibrated to tangential signal *RMS* (matched-filter SNR
    convention; std would overstate the radial SNR advantage because
    radial scalp patterns are less zero-mean across this montage).
  - 5 noise seeds → mean ± std bands. Not a beats-baseline claim; a
    geometry property — but the seeds protect against single-draw
    artefacts as per CLAUDE.md.

Two-panel figure:
  TOP    — localization error vs orientation, 4 methods, depth=50mm.
  BOTTOM — scalp signal RMS vs orientation at 3 depths (20/50/70mm), the
           diagnostic that explains the top panel.
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spherical_forward import (
    SphericalHeadModel, forward_potential, make_standard_montage_64
)
from inverse_solvers import mne_operator, sloreta_operator, eloreta_operator


def build_source_space(model: SphericalHeadModel, spacing_mm: float = 10.0) -> np.ndarray:
    """Symmetric origin-centered grid so (50,0,0)/(20,0,0)/(70,0,0) land on grid points."""
    R_brain = model.radii_mm[0]
    R_max = R_brain - 5.0  # 76 mm
    n_half = int(np.floor(R_max / spacing_mm))
    coords = np.arange(-n_half, n_half + 1) * spacing_mm
    XX, YY, ZZ = np.meshgrid(coords, coords, coords, indexing='ij')
    positions = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    return positions[np.linalg.norm(positions, axis=1) <= R_max]


def build_free_leadfield(sources: np.ndarray, electrodes: np.ndarray,
                          model: SphericalHeadModel) -> np.ndarray:
    """Free-orientation leadfield: 3 cols (x,y,z unit dipoles) per source."""
    n_elec = electrodes.shape[0]
    n_src = sources.shape[0]
    L = np.zeros((n_elec, n_src, 3))
    for i, src_pos in enumerate(sources):
        for k in range(3):
            ehat = np.zeros(3); ehat[k] = 1.0
            L[:, i, k] = forward_potential(src_pos, ehat, electrodes, model=model)
        if (i + 1) % 200 == 0:
            print(f"  leadfield source {i+1}/{n_src}", end='\r', flush=True)
    print()
    return L.reshape(n_elec, n_src * 3)


def vector_localization_error(J_flat: np.ndarray, sources: np.ndarray,
                               true_pos: np.ndarray) -> float:
    n_src = sources.shape[0]
    mags = np.linalg.norm(J_flat.reshape(n_src, 3), axis=1)
    peak = int(np.argmax(mags))
    return float(np.linalg.norm(sources[peak] - true_pos))


def main():
    print("=" * 70)
    print("Orientation sweep (spherical EEG, free-orientation inverse)")
    print("=" * 70)

    model = SphericalHeadModel()
    electrodes = make_standard_montage_64(model)
    sources = build_source_space(model, spacing_mm=10.0)
    print(f"Source space: {sources.shape[0]} candidates (10mm grid)")

    # Verify the depths we'll probe are on-grid (zero floor error).
    for d in (20.0, 50.0, 70.0):
        p = np.array([d, 0.0, 0.0])
        on = bool(np.any(np.all(sources == p, axis=1)))
        floor = float(np.linalg.norm(sources - p, axis=1).min())
        print(f"  ({d:>4.0f},0,0): on-grid={on}, floor={floor:.3f} mm")

    print("\nBuilding free-orientation leadfield (3 cols/source)...")
    L = build_free_leadfield(sources, electrodes, model)
    print(f"  leadfield shape: {L.shape}")

    print("\nBuilding inverse operators (snr=3.0)...")
    operators = {
        'MNE':     mne_operator(L, snr=3.0),
        'wMNE':    mne_operator(L, snr=3.0, depth_weighting=0.8),
        'sLORETA': sloreta_operator(L, snr=3.0),
        'eLORETA': eloreta_operator(L, snr=3.0, max_iter=50, verbose=False),
    }

    moment_mag = 1e-8                              # 10 nA.m
    t_hat = np.array([0.0, 0.0, 1.0])              # tangential
    angles_deg = np.linspace(0.0, 90.0, 13)        # 7.5-deg steps
    seeds = [11, 23, 47, 101, 1729]

    # ------------ BOTTOM-PANEL DIAGNOSTIC: signal RMS vs angle at 3 depths.
    diag_depths = [20.0, 50.0, 70.0]
    diag_rms = {d: np.zeros(len(angles_deg)) for d in diag_depths}
    for d in diag_depths:
        pos = np.array([d, 0.0, 0.0])
        r_hat_d = pos / np.linalg.norm(pos)
        for ai, theta_deg in enumerate(angles_deg):
            th = np.deg2rad(theta_deg)
            m = moment_mag * (np.cos(th) * t_hat + np.sin(th) * r_hat_d)
            V = forward_potential(pos, m, electrodes, model=model)
            diag_rms[d][ai] = float(np.sqrt(np.mean(V ** 2)))

    print("\n--- Scalp signal RMS (V) vs angle, three depths ---")
    print(f"{'angle deg':>10} " + " ".join(f"{'d=' + str(int(d)) + 'mm':>16s}"
                                            for d in diag_depths))
    for ai, theta_deg in enumerate(angles_deg):
        line = f"{theta_deg:>10.1f} "
        for d in diag_depths:
            line += f"{diag_rms[d][ai]:>16.3e} "
        print(line)
    for d in diag_depths:
        r = diag_rms[d][-1] / diag_rms[d][0]
        print(f"  ratio (radial/tangential) at depth {int(d):>2}mm: {r:.2f}x")

    # ------------ TOP-PANEL: orientation sweep at fixed mid-depth, all methods.
    true_pos = np.array([50.0, 0.0, 0.0])
    r_hat = true_pos / np.linalg.norm(true_pos)

    # Calibrate noise to tangential signal RMS (matched-filter SNR convention).
    V_tang = forward_potential(true_pos, moment_mag * t_hat,
                                electrodes, model=model)
    sig_rms_tang = float(np.sqrt(np.mean(V_tang ** 2)))
    noise_std = sig_rms_tang / np.sqrt(10.0)   # power-SNR=10 at tangential
    print(f"\nTop panel: true_pos = {tuple(true_pos.astype(int))} mm")
    print(f"  tangential signal RMS = {sig_rms_tang:.3e} V")
    print(f"  fixed noise std       = {noise_std:.3e} V  "
          f"(matched-filter SNR=sqrt(10) at tangential)")

    results = {name: np.zeros((len(angles_deg), len(seeds))) for name in operators}
    amp_snr_mf = np.zeros(len(angles_deg))  # matched-filter SNR = RMS/sigma
    sig_rms = np.zeros(len(angles_deg))

    print(f"\nSweeping {len(angles_deg)} angles x {len(seeds)} seeds...")
    for ai, theta_deg in enumerate(angles_deg):
        th = np.deg2rad(theta_deg)
        m = moment_mag * (np.cos(th) * t_hat + np.sin(th) * r_hat)
        V_clean = forward_potential(true_pos, m, electrodes, model=model)
        sig_rms[ai] = float(np.sqrt(np.mean(V_clean ** 2)))
        amp_snr_mf[ai] = sig_rms[ai] / noise_std
        for si, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            V = V_clean + rng.normal(0, noise_std, size=V_clean.shape)
            for name, K in operators.items():
                J_hat = K @ V
                results[name][ai, si] = vector_localization_error(
                    J_hat, sources, true_pos
                )

    # Numeric table.
    print("\n" + "=" * 92)
    print(f"{'angle':>6} {'sigRMS':>11} {'ampSNR':>8} " +
          " ".join(f"{n:>14s}" for n in operators))
    print("-" * 92)
    for ai, theta_deg in enumerate(angles_deg):
        row = f"{theta_deg:>6.1f} {sig_rms[ai]:>11.3e} {amp_snr_mf[ai]:>8.2f} "
        for name in operators:
            m_ = results[name][ai].mean()
            s_ = results[name][ai].std()
            row += f" {m_:>6.1f} +/- {s_:>4.1f} "
        print(row)
    print("=" * 92)

    # ----------------------------- Plot -----------------------------------
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 8.5),
        gridspec_kw={'height_ratios': [3, 2]}, sharex=True
    )
    colors = {'MNE': 'C0', 'wMNE': 'C1', 'sLORETA': 'C2', 'eLORETA': 'C3'}
    for name in operators:
        m_ = results[name].mean(axis=1)
        s_ = results[name].std(axis=1)
        ax_top.plot(angles_deg, m_, '-o', label=name,
                    color=colors[name], linewidth=2)
        ax_top.fill_between(angles_deg, m_ - s_, m_ + s_,
                             color=colors[name], alpha=0.18)
    ax_top.axhline(0.0, color='gray', linestyle=':', linewidth=1,
                    label='grid floor (0 mm)')
    ax_top.set_ylabel('Localization error (mm)')
    ax_top.set_title(
        f'Spherical EEG, free-orientation inverse. Fixed pos '
        f'{tuple(true_pos.astype(int))} mm (depth {np.linalg.norm(true_pos):.0f} mm).  '
        f'Mean +/- std across {len(seeds)} noise seeds.\n'
        f'Expected tangential-easy -> radial-hard NOT reproduced — '
        f'in this model radial scalp signal is ~2x stronger (see bottom).'
    )
    ax_top.legend(loc='upper right', fontsize=9, ncol=2)
    ax_top.grid(True, alpha=0.3)

    depth_colors = {20.0: 'C5', 50.0: 'k', 70.0: 'C6'}
    for d in diag_depths:
        ax_bot.plot(angles_deg, diag_rms[d], '-s',
                    color=depth_colors[d], linewidth=1.6,
                    label=f'depth = {int(d)} mm')
    ax_bot.set_xlabel('Dipole orientation theta (deg)        '
                       '0 = tangential (+z)        90 = radial (+x)')
    ax_bot.set_ylabel('Scalp potential RMS (V)\nunit moment, no noise')
    ax_bot.set_title(
        'Diagnostic: scalp signal RMS vs orientation at three depths.  '
        'Radial > tangential by ~1.6-2x at every depth tested\n'
        '(de Munck (n+1) prefactor on radial term).  '
        'This is why the top-panel error mildly decreases with rotation toward radial.',
        fontsize=10
    )
    ax_bot.legend(loc='upper left', fontsize=9)
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), '..', 'figures',
                            'q_b_orientation_sweep.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"\nFigure saved to: {out_path}")
    print("\n" + "=" * 70)
    print("Done. Interpretation: the spherical model does not motivate the")
    print("observability gradient in the direction Q-B suggested. The vertex")
    print("failure is realistic-anatomy/BEM-specific. Right venue for a")
    print("convincing continuum: rerun this sweep in realistic geometry,")
    print("not the analytic sphere.")
    print("=" * 70)


if __name__ == '__main__':
    main()
