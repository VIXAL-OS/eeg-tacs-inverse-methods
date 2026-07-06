"""
Q-D: distributed-network recovery on the spherical head.

Compares standard per-voxel sLORETA / eLORETA against a network-prior MNE
built on a blind k-means parcellation (K=24) of the 8 mm source grid.
Ground truth is a 12-node co-active network of radial dipoles near the
xz-plane (so it stays visible in the slice plot reused from Q-A) with
correlated low-pass time-courses (shared_fraction=0.6, so neither rank-1
nor independent).

The blind k-means parcellation is computed from source coordinates only —
no peeking at the true network positions or activity. As a ceiling
reference we also run an "oracle Voronoi" parcellation that assigns each
grid point to its nearest true node. The oracle is NOT a real method
(real clinical use has no access to the true node positions); it is
included only to quantify how much the realistic blind partition's
mismatch with biology costs us — i.e. to make the partition-dependence
limitation honest and quantitative.

Outputs
-------
- bandr_inverse/figures/day4_network_recovery.png — xz-plane slices
- bandr_inverse/figures/day4_network_recovery_parcels.png — bar chart
- stdout: parcel-detection precision/recall/F1, parcel time-course
  correlation, per-node localization error.
"""
from __future__ import annotations
import sys
import os
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from spherical_forward import SphericalHeadModel, make_gsn_montage_64
from inverse_solvers import sloreta_operator, eloreta_operator
from network_recovery import (
    sample_network_positions, make_correlated_node_timecourses,
    simulate_network_scalp_data,
    parcellate_source_space, network_prior_operator,
    per_source_amplitude, parcel_mean_amplitude,
    parcel_time_course, true_parcel_assignment,
    parcel_detection_pr, parcel_timecourse_correlation,
)


def _load_sanity_helpers():
    """Import build_source_space + build_leadfield from 01_sphere_sanity.py
    without renaming the file (a digit-leading basename blocks `import`).
    """
    path = os.path.join(os.path.dirname(__file__), '01_sphere_sanity.py')
    spec = importlib.util.spec_from_file_location('sphere_sanity', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_source_space, mod.build_leadfield


def plot_slice(ax, sources, per_src_map, node_pos, title,
               R_head, R_brain, y_slab_mm=6.0):
    """xz-plane scatter of recovered per-source amplitudes (per-panel
    normalization) with true network nodes overlaid as cyan stars.
    Matches the conventions of plot_recovery_slice in 01_sphere_sanity.py.
    """
    in_slice = np.abs(sources[:, 1]) <= y_slab_mm
    pts = sources[in_slice][:, [0, 2]]
    vals = per_src_map[in_slice]
    vmax = vals.max() if vals.max() > 0 else 1.0
    ax.scatter(pts[:, 0], pts[:, 1], c=vals / vmax, s=40, cmap='magma',
               vmin=0, vmax=1, alpha=0.95, edgecolors='none')
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R_head * np.cos(theta), R_head * np.sin(theta),
            color='black', linewidth=0.8)
    ax.plot(R_brain * np.cos(theta), R_brain * np.sin(theta),
            color='black', linewidth=0.4, linestyle='--', alpha=0.4)
    in_node_slice = np.abs(node_pos[:, 1]) <= y_slab_mm
    ax.scatter(node_pos[in_node_slice, 0], node_pos[in_node_slice, 2],
               marker='*', s=180, color='cyan', edgecolors='black',
               linewidths=1.0, zorder=10)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.tick_params(labelsize=8)


def per_node_recovery_scores(
    per_source: np.ndarray,                # (n_src,) RMS amplitudes
    node_positions: np.ndarray,
    source_positions: np.ndarray,
) -> np.ndarray:
    """For each true node, recovered amplitude at the nearest grid point,
    normalized by the brain-wide max. Range [0, 1].

    Partition-free metric of whether the network's node was preserved as
    distinct activity. A method that smears the network into a blob
    centered between nodes will have R_k ~ 0.2-0.5 at every node (no
    single node dominates); a method that correctly preserves the network
    will have R_k ~ 1 at every node.
    """
    n_max = max(per_source.max(), 1e-30)
    out = np.empty(len(node_positions))
    for i, npos in enumerate(node_positions):
        nearest = int(np.argmin(np.linalg.norm(source_positions - npos, axis=1)))
        out[i] = per_source[nearest] / n_max
    return out


def network_mass_fraction(
    per_source: np.ndarray,
    node_positions: np.ndarray,
    source_positions: np.ndarray,
    radius_mm: float = 15.0,
) -> tuple[float, float]:
    """Fraction of total recovered mass within radius_mm of any true node.

    Returns (mass_fraction, volume_baseline). Methods that focus on the
    network have mass_fraction >> volume_baseline; methods that smear
    diffusely have mass_fraction ~ volume_baseline.
    """
    near = np.zeros(len(source_positions), dtype=bool)
    for p in node_positions:
        near |= (np.linalg.norm(source_positions - p, axis=1) <= radius_mm)
    total = per_source.sum()
    mass = per_source[near].sum() / max(total, 1e-30)
    vol = float(near.sum()) / len(source_positions)
    return float(mass), vol


def per_node_timecourse_correlation(
    J_hat: np.ndarray,                     # (n_src, T)
    node_positions: np.ndarray,
    source_positions: np.ndarray,
    true_amps: np.ndarray,                 # (n_nodes, T) ground-truth signals
) -> np.ndarray:
    """Per-node Pearson correlation between the recovered time course at
    each node's nearest grid point and the true per-node time course.
    Partition-free; tests whether the node-level dynamics were preserved.
    """
    out = np.full(len(node_positions), np.nan)
    for i, npos in enumerate(node_positions):
        nearest = int(np.argmin(np.linalg.norm(source_positions - npos, axis=1)))
        a, b = J_hat[nearest], true_amps[i]
        if a.std() < 1e-30 or b.std() < 1e-30:
            continue
        out[i] = float(np.corrcoef(a, b)[0, 1])
    return out


def main():
    print("=" * 68)
    print("Q-D: distributed-network recovery on the spherical head")
    print("=" * 68)

    # ----- 1. Head, electrodes, source space, leadfield ------------------
    model = SphericalHeadModel()
    electrodes = make_gsn_montage_64(model)
    build_source_space, build_leadfield = _load_sanity_helpers()
    print("\nBuilding source space (8mm grid)...")
    sources = build_source_space(model, spacing_mm=8.0)
    print(f"  {sources.shape[0]} source positions")
    print("Building leadfield (the slow step — ~1-2 min)...")
    L = build_leadfield(sources, electrodes, model, orientation='radial')
    print(f"  L shape={L.shape}, cond(L)={np.linalg.cond(L):.2e}")

    # ----- 2. Ground-truth network ---------------------------------------
    rng = np.random.default_rng(20260524)
    n_nodes = 12
    node_pos = sample_network_positions(
        model, n_nodes=n_nodes, min_separation_mm=18.0, margin_mm=8.0,
        slab_y_mm=5.0, rng=rng,
    )
    node_radii = np.linalg.norm(node_pos, axis=1)
    print(f"\nGround-truth network: {n_nodes} radial dipoles, |y|<=5mm")
    print(f"  radii (mm): {np.sort(node_radii).round(1)}")

    # Inverse-crime check: confirm nodes are OFF the inversion grid
    nearest_grid_dist = np.array([
        np.linalg.norm(sources - p, axis=1).min() for p in node_pos
    ])
    print(f"  off-grid offsets (mm): min={nearest_grid_dist.min():.2f}, "
          f"median={np.median(nearest_grid_dist):.2f}, "
          f"max={nearest_grid_dist.max():.2f}")
    if nearest_grid_dist.min() < 0.05:
        print("  WARNING: a node lies on the inversion grid — this is an "
              "inverse crime (same positions used for forward and inverse).")

    n_times = 256
    # shared_fraction=0.3 gives moderate inter-node correlation (mean r ~0.2).
    # Higher correlation (0.6+) drives the scalp signal toward rank-1, which
    # makes "all nodes equal" and "few nodes equal-sum" indistinguishable to
    # any inverse — and lets smearing methods "win" parcel detection by
    # broad coverage. Lower correlation gives the recovery enough rank for
    # localization to actually pay off.
    amps = make_correlated_node_timecourses(
        n_nodes, n_times, shared_fraction=0.3, rng=rng,
    ) * 1e-8  # 10 nA·m per-node scale (same order as Q-A)
    inter_node_corr = np.corrcoef(amps)
    off_diag = inter_node_corr[~np.eye(n_nodes, dtype=bool)]
    print(f"  inter-node time-course corr: mean={off_diag.mean():.2f}, "
          f"min={off_diag.min():.2f}, max={off_diag.max():.2f}")

    V, V_clean = simulate_network_scalp_data(
        node_pos, amps, electrodes, model,
        amplitude_snr=3.0, rng=rng,
    )
    measured_snr = (np.sqrt(np.mean(V_clean ** 2))
                    / np.sqrt(np.mean((V - V_clean) ** 2)))
    print(f"\nScalp data: V shape={V.shape}, "
          f"V_clean RMS={np.sqrt(np.mean(V_clean**2)):.2e} V, "
          f"amplitude SNR={measured_snr:.2f}")

    # ----- 3. Blind k-means parcellation ---------------------------------
    n_parcels = 24
    labels = parcellate_source_space(sources, n_parcels, random_state=42)
    parcel_sizes = np.bincount(labels, minlength=n_parcels)
    nonempty = parcel_sizes > 0
    print(f"\nBlind k-means parcellation: K={n_parcels} requested, "
          f"non-empty={int(nonempty.sum())}, "
          f"sizes min/median/max = "
          f"{int(parcel_sizes[nonempty].min())}/"
          f"{int(np.median(parcel_sizes[nonempty]))}/"
          f"{int(parcel_sizes.max())}")
    node_parcels = true_parcel_assignment(node_pos, sources, labels)
    true_active = np.unique(node_parcels)
    print(f"  true nodes land in {len(true_active)} distinct parcels "
          f"({n_nodes - len(true_active)} collisions)")
    print(f"  random-baseline precision @ top-K = "
          f"{len(true_active)/n_parcels:.2f}")

    # ----- 4. Inverse operators ------------------------------------------
    print("\nBuilding inverse operators...")
    K_slo = sloreta_operator(L, snr=3.0)
    K_elo = eloreta_operator(L, snr=3.0, max_iter=50, verbose=False)
    K_net_voxel, K_net_parcel, _ = network_prior_operator(
        L, labels, snr=3.0, n_parcels=n_parcels, normalize='unit_l2',
    )

    # Oracle Voronoi partition: each source assigned to its nearest true
    # node. NOT realistic — used only as a ceiling reference to make the
    # blind-partition cost explicit.
    voronoi_labels = np.array([
        int(np.argmin(np.linalg.norm(node_pos - p, axis=1))) for p in sources
    ])
    K_ora_voxel, K_ora_parcel, _ = network_prior_operator(
        L, voronoi_labels, snr=3.0, n_parcels=n_nodes, normalize='unit_l2',
    )

    method_specs = [
        ('sLORETA',                         K_slo),
        ('eLORETA',                         K_elo),
        ('network-prior (blind k-means)',   K_net_voxel),
        ('network-prior (oracle Voronoi)',  K_ora_voxel),
    ]

    # ----- 5. Reconstruction + evaluation --------------------------------
    method_results = {}
    for name, K in method_specs:
        J_hat = K @ V                            # (n_src, T)
        per_src = per_source_amplitude(J_hat)    # RMS over time
        per_parcel = parcel_mean_amplitude(per_src, labels, n_parcels)
        tc_recov = parcel_time_course(J_hat, labels, n_parcels)
        method_results[name] = dict(J_hat=J_hat, per_src=per_src,
                                    per_parcel=per_parcel, tc_recov=tc_recov)

    # True per-source map / per-parcel map / per-parcel time-course built by
    # snapping each node to its nearest grid point. Used ONLY in evaluation.
    nearest_idx = np.array([
        int(np.argmin(np.linalg.norm(sources - p, axis=1))) for p in node_pos
    ])
    J_true_on_grid = np.zeros((sources.shape[0], n_times))
    for k in range(n_nodes):
        J_true_on_grid[nearest_idx[k]] += amps[k]
    true_per_src = per_source_amplitude(J_true_on_grid)
    true_per_parcel = parcel_mean_amplitude(true_per_src, labels, n_parcels)
    true_tc = parcel_time_course(J_true_on_grid, labels, n_parcels)

    # Oracle sanity check: does MNE on the 12-column oracle leadfield
    # actually recover the per-node amplitudes? If a_hat correlates with
    # the true amps, the oracle's recovery is genuine; if it collapses to a
    # uniform vector regardless of input, the inverse setup has an issue.
    a_hat_oracle = K_ora_parcel @ V         # (n_nodes, T)
    # Permutation of true amps to align with Voronoi cell indices. Each true
    # node maps to its own Voronoi cell, but k=0..n_nodes-1 indexes them in
    # the order they appear in node_pos — so K_ora_parcel[k] estimates the
    # amplitude assigned to whatever single node sits in Voronoi cell k.
    # Since voronoi_labels uses argmin(node_pos - p) indexing, cell k
    # contains node k — no permutation needed.
    per_node_corr_oracle = np.array([
        np.corrcoef(a_hat_oracle[k], amps[k])[0, 1] for k in range(n_nodes)
    ])
    a_hat_rms = np.sqrt(np.mean(a_hat_oracle ** 2, axis=1))
    print(f"\nOracle sanity: per-cell a_hat-vs-truth correlation: "
          f"median={np.median(per_node_corr_oracle):.2f}, "
          f"min={per_node_corr_oracle.min():.2f}, "
          f"max={per_node_corr_oracle.max():.2f}")
    print(f"  a_hat RMS across cells: min={a_hat_rms.min():.2e}, "
          f"max={a_hat_rms.max():.2e} "
          f"(ratio max/min = {a_hat_rms.max()/a_hat_rms.min():.2f})")

    print("\n" + "=" * 68)
    print("Per-node recovery (partition-free)")
    print("=" * 68)
    print(f"{'method':38s}  {'mean R':>7s}  {'min R':>6s}  "
          f"{'>0.3':>5s}  {'mean r(tc)':>10s}")
    for name, _ in method_specs:
        res = method_results[name]
        R = per_node_recovery_scores(res['per_src'], node_pos, sources)
        tc_corr = per_node_timecourse_correlation(
            res['J_hat'], node_pos, sources, amps
        )
        tc_corr_clean = tc_corr[~np.isnan(tc_corr)]
        n_detected = int((R > 0.3).sum())
        print(f"{name:38s}  {R.mean():7.2f}  {R.min():6.2f}  "
              f"{n_detected:2d}/{n_nodes}  "
              f"{(tc_corr_clean.mean() if tc_corr_clean.size else float('nan')):10.3f}")

    print("\nNetwork mass concentration (mass within 15mm of any true node)")
    _, vol_base = network_mass_fraction(
        np.ones(len(sources)), node_pos, sources, radius_mm=15.0,
    )
    print(f"  volume baseline (uniform map): {vol_base:.2%}")
    print(f"{'method':38s}  {'mass frac':>10s}  {'/baseline':>10s}")
    for name, _ in method_specs:
        m, _ = network_mass_fraction(
            method_results[name]['per_src'], node_pos, sources,
            radius_mm=15.0,
        )
        print(f"{name:38s}  {m:10.2%}  {m / max(vol_base, 1e-30):>10.2f}x")

    print("\nParcel-detection (top-K parcels by amplitude on blind k-means)")
    print("Note: oracle Voronoi spreads its mass uniformly across its OWN 12")
    print("cells, so reaggregating to the blind 24-cell partition averages it")
    print("out — this metric is biased against partition-mismatched methods.")
    print(f"{'method':38s}  {'prec':>5s}  {'recall':>6s}  "
          f"{'F1':>5s}  {'mean r(tc)':>10s}")
    for name, _ in method_specs:
        res = method_results[name]
        prec, rec, f1, _sel = parcel_detection_pr(
            res['per_parcel'], true_active, top_k=len(true_active),
        )
        r_tc = parcel_timecourse_correlation(res['tc_recov'], true_tc,
                                              true_active)
        print(f"{name:38s}  {prec:5.2f}  {rec:6.2f}  "
              f"{f1:5.2f}  {r_tc:10.3f}")
    print(f"{'(random baseline)':38s}  "
          f"{len(true_active)/n_parcels:5.2f}  "
          f"{len(true_active)/n_parcels:6.2f}  "
          f"{len(true_active)/n_parcels:5.2f}  "
          f"{0.0:10.3f}")

    # ----- 6. Spatial-recovery figure ------------------------------------
    print("\nBuilding figure...")
    R_head = model.head_radius_mm
    R_brain = model.radii_mm[0]
    method_names = [name for name, _ in method_specs]

    fig, axes = plt.subplots(1, 1 + len(method_names),
                             figsize=(4.6 * (1 + len(method_names)), 5.0))
    plot_slice(axes[0], sources, true_per_src, node_pos,
               f'ground truth ({n_nodes} nodes)', R_head, R_brain)
    for ax, name in zip(axes[1:], method_names):
        plot_slice(ax, sources, method_results[name]['per_src'], node_pos,
                   name, R_head, R_brain)
    axes[0].set_ylabel('Z (mm)', fontsize=11)
    for ax in axes:
        ax.set_xlabel('X (mm)', fontsize=11)
    fig.suptitle(
        f'Q-D network recovery — xz-plane slice (|y|<=6mm), '
        f'{n_nodes}-node network. Per-panel-normalized RMS amplitude over '
        f'{n_times} samples.',
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__), '..', 'figures', 'day4_network_recovery.png'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out_path}")

    # ----- 7. Per-parcel amplitude bar chart -----------------------------
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    width = 0.16
    x = np.arange(n_parcels)
    truth_norm = true_per_parcel / max(true_per_parcel.max(), 1e-30)
    ax2.bar(x - 2 * width, truth_norm, width, color='black',
            label='ground truth')
    palette = ['#4C72B0', '#DD8452', '#55A467', '#C44E52']
    for i, name in enumerate(method_names):
        v = method_results[name]['per_parcel']
        v_norm = v / max(v.max(), 1e-30)
        ax2.bar(x + (i - 1) * width, v_norm, width, color=palette[i],
                label=name)
    for k in true_active:
        ax2.axvspan(k - 0.5, k + 0.5, color='yellow', alpha=0.15, zorder=0)
    ax2.set_xlabel('parcel index (blind k-means)')
    ax2.set_ylabel('per-parcel RMS amplitude (normalized to method max)')
    ax2.set_title('Per-parcel recovered amplitude vs ground truth. '
                  'Yellow bands = parcels containing a true node.')
    ax2.legend(fontsize=8, loc='upper right', ncol=2)
    ax2.set_xticks(x)
    ax2.tick_params(axis='x', labelsize=8)
    plt.tight_layout()
    out_path2 = os.path.join(
        os.path.dirname(__file__), '..', 'figures',
        'day4_network_recovery_parcels.png'
    )
    plt.savefig(out_path2, dpi=120, bbox_inches='tight')
    plt.close(fig2)
    print(f"  saved {out_path2}")

    print("\n" + "=" * 68)
    print("Q-D complete. Compare the four spatial panels for the smearing")
    print("story; read the precision/recall and per-node tables above to")
    print("see the partition-dependence cost (blind vs oracle).")
    print("=" * 68)


if __name__ == '__main__':
    main()
