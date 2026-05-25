"""
Day-1 sanity check: spherical head + dipole recovery.

This is THE script to run when you sit down with the project. It does:
  1. Set up a 4-shell spherical head with 64 electrodes
  2. Build a leadfield numerically from analytic forward solutions to a
     grid of candidate source locations
  3. Place a known dipole, generate synthetic scalp data (with optional
     noise)
  4. Recover the source using MNE, sLORETA, eLORETA
  5. Compute localization error, PSF, depth bias for each method
  6. Plot the result

If this runs end-to-end and the methods recover the source within reason,
the unit-test level of the pipeline is working. Then we can move on.
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from spherical_forward import (
    SphericalHeadModel, forward_potential, make_standard_montage_64
)
from inverse_solvers import mne_operator, sloreta_operator, eloreta_operator
from metrics import summary


def build_source_space(model: SphericalHeadModel, spacing_mm: float = 8.0) -> np.ndarray:
    """
    Generate candidate source locations on a regular grid inside the
    brain compartment.

    Returns
    -------
    source_pos : (n_sources, 3) array of positions in mm
    """
    R_brain = model.radii_mm[0]
    # Conservative inner boundary so we don't sit on the brain/CSF interface
    R_max = R_brain - 5.0
    coords_1d = np.arange(-R_max, R_max + spacing_mm, spacing_mm)
    XX, YY, ZZ = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    positions = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    # Keep only points inside the brain
    radii = np.linalg.norm(positions, axis=1)
    mask = radii <= R_max
    return positions[mask]


def build_leadfield(
    source_positions: np.ndarray,
    electrode_positions: np.ndarray,
    model: SphericalHeadModel,
    orientation: str = 'radial',
) -> np.ndarray:
    """
    Build the (n_elec, n_sources) leadfield matrix.

    For 'radial' orientation, each source is a unit radial dipole.
    For 'free', returns (n_elec, n_sources, 3) for x/y/z components.
    """
    n_elec = electrode_positions.shape[0]
    n_src = source_positions.shape[0]

    if orientation == 'radial':
        L = np.zeros((n_elec, n_src))
        for i, src_pos in enumerate(source_positions):
            r = np.linalg.norm(src_pos)
            if r < 1e-9:
                # Source at center: take z-direction as nominal radial
                moment = np.array([0.0, 0.0, 1.0])
            else:
                moment = src_pos / r  # unit radial
            V = forward_potential(
                src_pos, moment, electrode_positions, model=model
            )
            L[:, i] = V
            if (i + 1) % 50 == 0:
                print(f"  leadfield column {i+1}/{n_src}", end='\r', flush=True)
        print()
        return L
    else:
        raise NotImplementedError("Only radial orientation implemented yet")


def plot_recovery_slice(ax, sources, J_hat, true_pos, method_name,
                         loc_err, R_head, R_brain):
    """xz-plane slice (y=0): reconstruction magnitude as scatter, true source as star.

    Each panel is independently normalized to its own peak — the brightness
    encodes shape and concentration, not absolute amplitude. The y=0 plane
    is chosen because all three test dipoles lie in it.
    """
    in_slice = np.abs(sources[:, 1]) <= 6.0  # ±6mm catches one grid layer
    pts = sources[in_slice][:, [0, 2]]
    mags = np.abs(J_hat[in_slice])
    mags_norm = mags / mags.max() if mags.max() > 0 else mags
    ax.scatter(pts[:, 0], pts[:, 1], c=mags_norm, s=40, cmap='magma',
               vmin=0, vmax=1, alpha=0.95, edgecolors='none')
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R_head * np.cos(theta), R_head * np.sin(theta),
            color='black', linewidth=0.8)
    ax.plot(R_brain * np.cos(theta), R_brain * np.sin(theta),
            color='black', linewidth=0.4, linestyle='--', alpha=0.4)
    ax.scatter(true_pos[0], true_pos[2], marker='*', s=320,
               color='cyan', edgecolors='black', linewidths=1.2, zorder=10)
    ax.set_title(f'{method_name}   err = {loc_err:.1f} mm', fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.tick_params(labelsize=8)


def main():
    print("=" * 60)
    print("Day-1 sanity check: spherical head + dipole recovery")
    print("=" * 60)

    # 1. Head model and electrodes
    model = SphericalHeadModel()
    print(f"\nHead model: {model.n_shells}-shell")
    print(f"  Radii (mm): {model.radii_mm}")
    print(f"  Sigmas (S/m): {model.sigmas}")

    electrodes = make_standard_montage_64(model)
    print(f"\nElectrodes: {electrodes.shape[0]} channels")

    # 2. Source space
    print("\nBuilding source space (8mm grid)...")
    sources = build_source_space(model, spacing_mm=8.0)
    print(f"  {sources.shape[0]} candidate source locations")

    # 3. Leadfield
    print("\nBuilding leadfield (n_elec x n_sources)...")
    L = build_leadfield(sources, electrodes, model, orientation='radial')
    print(f"  Leadfield shape: {L.shape}")
    print(f"  Condition number: {np.linalg.cond(L):.2e}")
    print(f"  ||L|| = {np.linalg.norm(L):.3e}")

    # 4. Place a known dipole and simulate
    # Try a few different depths
    test_positions = [
        np.array([50.0, 0.0, 0.0]),   # shallow, lateral
        np.array([0.0, 0.0, 50.0]),   # shallow, superior
        np.array([20.0, 0.0, 0.0]),   # deep
    ]
    test_labels = ['shallow lateral', 'shallow superior', 'deep']

    method_names = ['MNE', 'wMNE', 'sLORETA', 'eLORETA']
    fig, axes = plt.subplots(len(test_positions), len(method_names),
                             figsize=(3.5 * len(method_names),
                                      3.5 * len(test_positions)),
                             squeeze=False)
    R_head = model.head_radius_mm
    R_brain = model.radii_mm[0]

    print("\n" + "=" * 60)
    print("Recovery tests")
    print("=" * 60)

    for row_idx, (true_pos, label) in enumerate(zip(test_positions,
                                                     test_labels)):
        print(f"\n--- Source: {label} at {true_pos} "
              f"(depth = {np.linalg.norm(true_pos):.1f} mm) ---")

        # Generate scalp data: unit radial dipole at true_pos
        r = np.linalg.norm(true_pos)
        moment = (true_pos / r) * 1e-8  # 10 nA·m, typical dipole strength
        V_clean = forward_potential(true_pos, moment, electrodes, model=model)

        # Add noise (SNR ~ 10)
        signal_power = np.var(V_clean)
        noise_power = signal_power / 10.0
        rng = np.random.default_rng(42)
        V = V_clean + rng.normal(0, np.sqrt(noise_power), size=V_clean.shape)

        # Recover with each method
        operators = {
            'MNE': mne_operator(L, snr=3.0),
            'wMNE': mne_operator(L, snr=3.0, depth_weighting=0.8),
            'sLORETA': sloreta_operator(L, snr=3.0),
            'eLORETA': eloreta_operator(L, snr=3.0, max_iter=50,
                                         verbose=True),
        }

        for col_idx, name in enumerate(method_names):
            K = operators[name]
            J_hat = K @ V
            metrics = summary(J_hat, sources, true_pos)
            print(f"  {name:8s}: loc_err = {metrics['localization_error_mm']:5.1f} mm, "
                  f"PSF FWHM = {metrics['psf_fwhm_mm']:5.1f} mm, "
                  f"depth_bias = {metrics['depth_bias_mm']:+5.1f} mm")
            plot_recovery_slice(axes[row_idx, col_idx], sources, J_hat,
                                 true_pos, name,
                                 metrics['localization_error_mm'],
                                 R_head, R_brain)

        axes[row_idx, 0].set_ylabel(
            f'{label}\n(r = {np.linalg.norm(true_pos):.0f} mm)\n\nZ (mm)',
            fontsize=10
        )

    for ax in axes[-1]:
        ax.set_xlabel('X (mm)', fontsize=10)
    fig.suptitle('xz-plane slice (y=0). Cyan star = true source. '
                 'Magnitudes normalized per panel.', fontsize=10, y=1.00)

    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__), '..', 'figures', 'day1_sphere_sanity.png'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"\nFigure saved to: {out_path}")

    print("\n" + "=" * 60)
    print("Day-1 sanity check complete.")
    print("=" * 60)
    print("\nIf each method's bright cluster sits near its cyan star,")
    print("the pipeline is working. See loc_err in each panel title")
    print("for the numeric distance from peak to true source.")
    print("Next: Question B (realistic head geometry) and Question C")
    print("(system ID via active calibration). See README.md.")


if __name__ == '__main__':
    main()
