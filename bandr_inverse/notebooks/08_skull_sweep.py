"""
Q-C extension: continuous skull-conductivity robustness sweep.

Q-C Exp 3 established the calibrated-open-loop targeting result at three
discrete skull scales {0.7, 1.0, 1.3}: ~85% signed delivery across all
three, vs uncalibrated {64%, 85%, 103%}. The Monte-Carlo self-check at
those three points (5 seeds) showed σ ~ 0.1pp — the calibration draw is
not the noise source.

This notebook upgrades that to a continuous sweep over s ∈ [0.6, 1.4] in
steps of 0.05 (17 scales), with 5 seeds per scale, so the paper can say
"robust across conductivity in [X, Y]" instead of "robust at three
points." Watching specifically for delivery roll-off at the extremes:

  - Low s (high skull resistance): calibration may saturate / regularization
    may start to dominate the open-loop solve.
  - High s (low skull resistance): less natural attenuation, less headroom
    for the safety budget, may overshoot.

If the calibrated line bends at either end that's a real operating-range
limit and is reported as the headline, NOT smoothed over.

Reuses calibration.simulate_calibration_data + calibration.fit_conductivity
+ closed_loop.open_loop_target verbatim. Geometry helpers are inlined
copies of those in 03_system_id.py (small enough that a shared module
would be overkill).

Run
---
    $env:MPLBACKEND='Agg'
    & .\.venv\Scripts\python.exe .\bandr_inverse\notebooks\08_skull_sweep.py
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

from spherical_forward import (
    SphericalHeadModel, forward_potential, make_standard_montage_64
)
from calibration import (
    simulate_calibration_data, fit_conductivity
)
from closed_loop import open_loop_target


# ----- geometry helpers (copied from 03_system_id.py) -----

def build_source_space(model: SphericalHeadModel,
                        spacing_mm: float = 12.0) -> np.ndarray:
    R_brain = model.radii_mm[0]
    R_max = R_brain - 5.0
    coords = np.arange(-R_max, R_max + spacing_mm, spacing_mm)
    XX, YY, ZZ = np.meshgrid(coords, coords, coords, indexing='ij')
    positions = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    mask = np.linalg.norm(positions, axis=1) <= R_max
    return positions[mask]


def build_leadfield(source_positions: np.ndarray,
                     electrode_positions: np.ndarray,
                     model: SphericalHeadModel) -> np.ndarray:
    n_elec = electrode_positions.shape[0]
    n_src = source_positions.shape[0]
    L = np.zeros((n_elec, n_src))
    for i, src_pos in enumerate(source_positions):
        r = np.linalg.norm(src_pos)
        moment = (src_pos / r) if r > 1e-9 else np.array([0.0, 0.0, 1.0])
        L[:, i] = forward_potential(src_pos, moment, electrode_positions, model=model)
    return L


def scaled_skull_model(template: SphericalHeadModel, scale: float) -> SphericalHeadModel:
    sigmas = list(template.sigmas)
    sigmas[2] = template.sigmas[2] * scale
    return replace(template, sigmas=tuple(sigmas))


# ============================================================
# Sweep
# ============================================================

def run_sweep(
    electrodes: np.ndarray,
    sources: np.ndarray,
    L_template: np.ndarray,
    template_model: SphericalHeadModel,
    target_pos: np.ndarray,
    scales: np.ndarray,
    seeds: list[int],
    K_cal: int = 80,
) -> dict:
    """Sweep true skull scale; for each scale run calibration + open-loop
    across multiple seeds. Also record uncalibrated open-loop (deterministic
    per scale)."""
    target_idx = int(np.argmin(np.linalg.norm(sources - target_pos, axis=1)))
    target_pos_actual = sources[target_idx]

    # Feasible target = what L_template alone, with safety headroom, can
    # deliver at this voxel. Same formula as Q-C Exp 3. Depends on L_template
    # only, so constant across the sweep.
    L_target_col = L_template[:, target_idx]
    norm_sq = float(np.dot(L_target_col, L_target_col))
    I_unc = L_target_col / (norm_sq + 1e-3 ** 2)
    budget = 2e-3
    headroom = 0.5
    scale_factor = headroom * budget / np.sum(np.abs(I_unc))
    feasible_target = norm_sq * scale_factor / (norm_sq + 1e-3 ** 2)
    print(f"  target voxel: {target_pos_actual} mm (idx {target_idx})")
    print(f"  feasible_target = {feasible_target:.4e} V/m")
    print(f"  sweep: scales={scales[0]:.2f}..{scales[-1]:.2f} step "
          f"{scales[1]-scales[0]:.2f} ({len(scales)} points)")
    print(f"  seeds: {seeds}")
    print(f"  K_cal = {K_cal}")

    E_target = np.zeros(sources.shape[0])
    E_target[target_idx] = feasible_target
    target_mask = np.zeros(sources.shape[0], dtype=bool)
    target_mask[target_idx] = True

    cal_signed = np.zeros((len(scales), len(seeds)))
    cal_theta = np.zeros((len(scales), len(seeds)))
    uncal_signed = np.zeros(len(scales))

    print()
    for i, s in enumerate(scales):
        true_model = scaled_skull_model(template_model, s)
        L_true = build_leadfield(sources, electrodes, true_model)

        # Uncalibrated open-loop: deterministic, one shot.
        I_uncal = open_loop_target(L_template, E_target,
                                    target_mask=target_mask,
                                    total_budget_A=2e-3,
                                    per_elec_max_A=1e-3,
                                    regularization=1e-3)
        E_actual_uncal = L_true.T @ I_uncal
        uncal_signed[i] = float(E_actual_uncal[target_idx] / feasible_target)

        # Calibrated open-loop: vary seed.
        for j, seed in enumerate(seeds):
            cal_data = simulate_calibration_data(
                true_model=true_model,
                electrode_pos_mm=electrodes,
                n_measurements=K_cal,
                noise_snr_power=100.0,
                seed=seed,
            )
            theta_hat, fitted_model, _ = fit_conductivity(
                cal_data, template_model, electrodes,
                parameterization='global_skull',
            )
            cal_theta[i, j] = float(theta_hat[0])
            L_hat = build_leadfield(sources, electrodes, fitted_model)
            I_cal = open_loop_target(L_hat, E_target,
                                      target_mask=target_mask,
                                      total_budget_A=2e-3,
                                      per_elec_max_A=1e-3,
                                      regularization=1e-3)
            E_actual_cal = L_true.T @ I_cal
            cal_signed[i, j] = float(E_actual_cal[target_idx] / feasible_target)

        cmean = cal_signed[i].mean()
        cstd = cal_signed[i].std()
        tmean = cal_theta[i].mean()
        tstd = cal_theta[i].std()
        print(f"  s={s:.2f}  theta_hat={tmean:.4f}±{tstd:.4f}  "
              f"calibrated={100*cmean:6.2f}% ± {100*cstd:4.2f}pp  "
              f"uncalibrated={100*uncal_signed[i]:6.2f}%")

    return {
        'scales': scales,
        'seeds': seeds,
        'target_pos': target_pos_actual,
        'target_idx': target_idx,
        'feasible_target': feasible_target,
        'cal_signed': cal_signed,        # (n_scales, n_seeds)
        'cal_theta': cal_theta,
        'uncal_signed': uncal_signed,    # (n_scales,)
    }


# ============================================================
# Reporting
# ============================================================

def print_table(res: dict):
    scales = res['scales']
    cal = res['cal_signed']
    th = res['cal_theta']
    unc = res['uncal_signed']
    print()
    print("=" * 78)
    print("Per-scale table (calibration: 5 seeds; uncalibrated: deterministic)")
    print("=" * 78)
    print(f"  {'scale':>6s}  {'theta_hat (mean±std)':>22s}  "
          f"{'calibrated signed delivery':>30s}  {'uncalibrated':>14s}")
    print("  " + "-" * 76)
    for i, s in enumerate(scales):
        cmean = cal[i].mean()
        cstd = cal[i].std()
        cmin = cal[i].min()
        cmax = cal[i].max()
        tmean = th[i].mean()
        tstd = th[i].std()
        print(f"  {s:>6.2f}  {tmean:>10.4f} ± {tstd:>8.4f}  "
              f"{100*cmean:>10.2f}% ± {100*cstd:>5.2f}pp  "
              f"[{100*cmin:>5.2f},{100*cmax:>6.2f}]  "
              f"{100*unc[i]:>10.2f}%")
    print()
    cmean_all = cal.mean(axis=1)
    cstd_all = cal.std(axis=1)
    print(f"  Overall calibrated:   mean across scales = {100*cmean_all.mean():.2f}%  "
          f"range = [{100*cmean_all.min():.2f}%, {100*cmean_all.max():.2f}%]  "
          f"spread = {100*(cmean_all.max()-cmean_all.min()):.2f}pp")
    print(f"  Overall uncalibrated: range = [{100*unc.min():.2f}%, {100*unc.max():.2f}%]  "
          f"spread = {100*(unc.max()-unc.min()):.2f}pp")
    print(f"  Worst per-scale seed std (calibrated): {100*cstd_all.max():.2f}pp")
    # Flag any operating-range bends.
    # Look at the slope of the mean curve near the extremes vs the middle.
    mid_lo = np.searchsorted(scales, 0.85)
    mid_hi = np.searchsorted(scales, 1.15)
    middle = cmean_all[mid_lo:mid_hi+1].mean()
    low_end = cmean_all[0]
    hi_end = cmean_all[-1]
    print(f"  Mean delivery at low end (s={scales[0]:.2f}):  {100*low_end:.2f}%")
    print(f"  Mean delivery in middle (s in [0.85,1.15]):    {100*middle:.2f}%")
    print(f"  Mean delivery at high end (s={scales[-1]:.2f}): {100*hi_end:.2f}%")
    print(f"  Δ(low − middle)  = {100*(low_end-middle):+.2f}pp")
    print(f"  Δ(high − middle) = {100*(hi_end-middle):+.2f}pp")


def plot_sweep(res: dict, out_dir: str):
    scales = res['scales']
    cal = res['cal_signed']
    unc = res['uncal_signed']

    cal_mean = cal.mean(axis=1)
    cal_std = cal.std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: signed delivery vs skull scale.
    ax = axes[0]
    ax.plot(scales, 100 * cal_mean, 'o-', color='C0',
            label='calibrated open-loop (mean ± std, 5 seeds)',
            linewidth=2.0, markersize=6)
    ax.fill_between(scales,
                     100 * (cal_mean - cal_std),
                     100 * (cal_mean + cal_std),
                     color='C0', alpha=0.25)
    ax.plot(scales, 100 * unc, 's-', color='C3',
            label='uncalibrated open-loop (deterministic)',
            linewidth=1.5, markersize=5)
    ax.axhline(100, color='gray', linestyle='--', alpha=0.6,
                label='target = 100% of feasible')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.4,
                label='nominal skull conductivity')
    # Mark the three original Q-C points.
    for s_orig in (0.7, 1.0, 1.3):
        ax.axvline(s_orig, color='black', linestyle=':', alpha=0.2)
    ax.set_xlabel('True skull conductivity scale (× nominal)')
    ax.set_ylabel('Signed delivery (% of feasible target)')
    ax.set_title('Q-C robustness sweep: open-loop targeting vs skull conductivity\n'
                 'shallow target at (50, 0, 30) mm; K=80 calibration measurements')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right panel: zoom on the calibrated line (drop uncalibrated) so the
    # ± std band and any bending at the extremes are visible.
    ax = axes[1]
    ax.plot(scales, 100 * cal_mean, 'o-', color='C0',
            linewidth=2.0, markersize=6, label='calibrated mean')
    ax.fill_between(scales,
                     100 * (cal_mean - cal_std),
                     100 * (cal_mean + cal_std),
                     color='C0', alpha=0.25, label='± 1 std (5 seeds)')
    # Per-seed scatter so you can see if there are tail outliers.
    for j in range(cal.shape[1]):
        ax.plot(scales, 100 * cal[:, j], '.', color='C0', alpha=0.35, markersize=3)
    ax.axhline(100, color='gray', linestyle='--', alpha=0.6)
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.4)
    for s_orig in (0.7, 1.0, 1.3):
        ax.axvline(s_orig, color='black', linestyle=':', alpha=0.2)
    ax.set_xlabel('True skull conductivity scale (× nominal)')
    ax.set_ylabel('Calibrated signed delivery (% of feasible)')
    ax.set_title('Calibrated-only zoom (per-seed dots + mean line + ±std band)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'q_c_skull_sweep.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"\nFigure saved to: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Q-C continuous-skull-conductivity robustness sweep")
    print("=" * 60)

    template_model = SphericalHeadModel()
    electrodes = make_standard_montage_64(template_model)
    print(f"\nTemplate model: 4-shell sphere")
    print(f"  sigmas (S/m) = {template_model.sigmas}")
    print(f"  electrodes = {electrodes.shape[0]}")

    print("\nBuilding source space (12mm spacing — matches Q-C Exp 3)...")
    sources = build_source_space(template_model, spacing_mm=12.0)
    print(f"  {sources.shape[0]} candidate sources")

    print("\nBuilding template leadfield...")
    L_template = build_leadfield(sources, electrodes, template_model)
    print(f"  L_template shape: {L_template.shape}, cond = {np.linalg.cond(L_template):.2e}")

    target_pos = np.array([50.0, 0.0, 30.0])
    scales = np.round(np.arange(0.60, 1.40 + 1e-9, 0.05), 2)
    seeds = [42, 43, 44, 45, 46]

    res = run_sweep(
        electrodes=electrodes,
        sources=sources,
        L_template=L_template,
        template_model=template_model,
        target_pos=target_pos,
        scales=scales,
        seeds=seeds,
        K_cal=80,
    )

    print_table(res)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    plot_sweep(res, out_dir)

    print("\n" + "=" * 60)
    print("Q-C sweep complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
