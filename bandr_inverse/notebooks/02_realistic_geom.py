"""
Q-B: realistic head geometry sanity check.

Same structure as 01_sphere_sanity.py, but uses MNE-Python's BEM
forward solution on the sample subject's MRI instead of the analytic
spherical solution. Source space is the actual cortical surface.

What this tests
---------------
The same four solvers we validated against the analytic sphere in Q-A
should still recover known cortical sources from synthetic data on
real geometry. If they do, the methods generalize from sphere to
brain. If they don't, there's a real-geometry-specific issue worth
understanding before Q-C.

Failure modes worth watching for
--------------------------------
- Different SNR sensitivity: real leadfields are more ill-conditioned
  than spherical, so eLORETA in particular may need different snr.
- Different depth bias direction: cortical sources are all "shallow"
  in the sphere sense (they're on the cortex), but cortical-surface
  geometry creates a different kind of depth-from-scalp variation
  (gyrus vs sulcus).
- Source-space sampling artifacts: cortical mesh has non-uniform
  density. PSF metric may behave weirdly near low-density regions.

Run
---
    python notebooks/02_realistic_geom.py

First run downloads the MNE sample dataset (~1.5 GB), cached after.
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from realistic_geom import (
    load_mne_sample, find_nearest_source, simulate_scalp_data
)
from inverse_solvers import mne_operator, sloreta_operator, eloreta_operator
from metrics import summary


def plot_cortical_slice(
    ax,
    source_positions: np.ndarray,
    magnitudes: np.ndarray,
    true_pos: np.ndarray,
    method_name: str,
    loc_err: float,
    slice_axis: str = 'y',
    slice_value: float = 0.0,
    slice_thickness: float = 8.0,
):
    """
    Slice plot in 2D, analogous to plot_recovery_slice in Q-A.

    Because cortical sources don't fill a regular grid, the visualization
    is sparser than Q-A's, but the same idea: pick a plane near the true
    source, scatter source-space points colored by magnitude, mark the
    true source with a cyan star.
    """
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[slice_axis]
    plot_axes = [i for i in [0, 1, 2] if i != axis_idx]

    # Select sources within the slice
    in_slice = np.abs(source_positions[:, axis_idx] - slice_value) <= slice_thickness
    pts = source_positions[in_slice][:, plot_axes]
    mags = magnitudes[in_slice]
    if mags.max() > 0:
        mags_norm = mags / mags.max()
    else:
        mags_norm = mags

    ax.scatter(pts[:, 0], pts[:, 1], c=mags_norm, s=12, cmap='magma',
               vmin=0, vmax=1, alpha=0.85, edgecolors='none')
    # True source projected onto the slice plane
    ax.scatter(true_pos[plot_axes[0]], true_pos[plot_axes[1]],
               marker='*', s=320, color='cyan', edgecolors='black',
               linewidths=1.2, zorder=10)
    ax.set_title(f'{method_name}   err = {loc_err:.1f} mm', fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlabel({'x': 'X', 'y': 'Y', 'z': 'Z'}[chr(ord('x') + plot_axes[0])] + ' (mm)', fontsize=8)
    ax.set_ylabel({'x': 'X', 'y': 'Y', 'z': 'Z'}[chr(ord('x') + plot_axes[1])] + ' (mm)', fontsize=8)
    ax.tick_params(labelsize=8)


def main():
    print("=" * 60)
    print("Q-B: Realistic head geometry (MNE sample subject)")
    print("=" * 60)

    print("\nLoading sample subject and building BEM forward solution...")
    print("(First run downloads ~1.5 GB. Subsequent runs use cache.)")
    geom = load_mne_sample(spacing='oct6', use_eeg_only=True,
                            fixed_orientation=True, verbose=False)
    print(f"\n{geom}")
    print(f"  Source positions range (mm): "
          f"X [{geom.source_pos_mm[:,0].min():.0f}, {geom.source_pos_mm[:,0].max():.0f}], "
          f"Y [{geom.source_pos_mm[:,1].min():.0f}, {geom.source_pos_mm[:,1].max():.0f}], "
          f"Z [{geom.source_pos_mm[:,2].min():.0f}, {geom.source_pos_mm[:,2].max():.0f}]")
    print(f"  Electrode count: {geom.n_electrodes}")

    # Pick test sources at anatomically interpretable positions in HEAD
    # coordinates. HEAD coords for the sample subject have +X right,
    # +Y forward (nasion), +Z up. Approximate locations:
    test_targets_mm = [
        np.array([55.0,   0.0,  20.0]),   # right lateral somatomotor-ish
        np.array([ 0.0,   0.0,  80.0]),   # vertex / SMA
        np.array([ 0.0,  20.0,  10.0]),   # medial frontal (deeper)
    ]
    test_labels = ['right lateral cortex', 'vertex', 'medial frontal']

    method_names = ['MNE', 'wMNE', 'sLORETA', 'eLORETA']

    fig, axes = plt.subplots(len(test_targets_mm), len(method_names),
                             figsize=(3.5 * len(method_names),
                                      3.5 * len(test_targets_mm)),
                             squeeze=False)

    print("\n" + "=" * 60)
    print("Recovery tests")
    print("=" * 60)

    rng = np.random.default_rng(42)

    for row_idx, (target, label) in enumerate(zip(test_targets_mm,
                                                    test_labels)):
        # Snap to nearest cortical vertex
        src_idx, snap_dist = find_nearest_source(target, geom)
        true_pos = geom.source_pos_mm[src_idx]

        print(f"\n--- Target: {label} ---")
        print(f"  Requested:  {target} mm")
        print(f"  Snapped to: {true_pos} mm  (vertex {src_idx}, "
              f"snap dist {snap_dist:.1f} mm)")

        # Simulate scalp data (10 dB power SNR, matches Q-A)
        V_noisy, V_clean = simulate_scalp_data(
            geom, source_idx=src_idx, moment_amplitude=10e-9,
            noise_snr_power=10.0, rng=rng,
        )

        # Recover with each method (snr=3.0 ≈ sqrt(10), matches Q-A convention)
        L = geom.leadfield
        operators = {
            'MNE': mne_operator(L, snr=3.0),
            'wMNE': mne_operator(L, snr=3.0, depth_weighting=0.8),
            'sLORETA': sloreta_operator(L, snr=3.0),
            'eLORETA': eloreta_operator(L, snr=3.0, max_iter=50,
                                         verbose=False),
        }

        # Pick a slice axis for plotting: use the axis where the
        # true source has the smallest absolute coordinate (so we
        # slice through the source plane)
        slice_axis = 'xyz'[int(np.argmin(np.abs(true_pos)))]
        slice_value = true_pos['xyz'.index(slice_axis)]

        for col_idx, name in enumerate(method_names):
            K = operators[name]
            J_hat = K @ V_noisy
            metrics_d = summary(J_hat, geom.source_pos_mm, true_pos)
            print(f"  {name:8s}: loc_err = {metrics_d['localization_error_mm']:5.1f} mm, "
                  f"PSF FWHM = {metrics_d['psf_fwhm_mm']:5.1f} mm, "
                  f"depth_bias = {metrics_d['depth_bias_mm']:+5.1f} mm")
            plot_cortical_slice(
                axes[row_idx, col_idx], geom.source_pos_mm,
                np.abs(J_hat), true_pos, name,
                metrics_d['localization_error_mm'],
                slice_axis=slice_axis, slice_value=slice_value,
                slice_thickness=10.0,
            )

        axes[row_idx, 0].set_ylabel(
            f'{label}\n(slice: {slice_axis}={slice_value:.0f} mm)',
            fontsize=10
        )

    fig.suptitle(
        'Q-B: cortical-source recovery on MNE sample subject. '
        'Cyan star = true source. Per-panel normalization.',
        fontsize=10, y=1.00
    )
    plt.tight_layout()

    out_path = os.path.join(
        os.path.dirname(__file__), '..', 'figures',
        'q_b_realistic_geom.png'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"\nFigure saved to: {out_path}")

    print("\n" + "=" * 60)
    print("Q-B sanity check complete.")
    print("=" * 60)
    print("\nIf each method's bright cluster sits near its cyan star,")
    print("the pipeline generalizes from sphere to brain.")
    print("Compare numbers with Q-A: ranges should be roughly")
    print("comparable for cortical sources (no truly deep sources here")
    print("since the source space is the cortical surface).")
    print("Next: Q-C (system ID via active calibration). The contribution.")


if __name__ == '__main__':
    main()
