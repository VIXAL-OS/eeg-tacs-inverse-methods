"""
Q-D follow-up: tight sub-cluster recovery.

Q-D-distributed (04_network_recovery.py) showed that on a 12-node spatially-
spread network, sLORETA/eLORETA's depth-normalized smearing prior already
matches a distributed source pattern, so no method dramatically wins. This
script tests the opposite regime: 3 tight sub-clusters of 4 nodes each
(total 12), each cluster's nodes within ~12 mm of a shared center. Per-voxel
methods are expected to fuse each cluster into one peak (4 underlying nodes,
1 recovered blob), while a fine-partition network-prior can — at minimum —
distribute the recovered amplitude across multiple parcels per cluster.

Methods compared
----------------
  - sLORETA   (per-voxel, depth-normalized)
  - eLORETA   (per-voxel, adaptive-alpha)
  - network-prior MNE      (blind k-means K=24)   — the original Q-D config
  - network-prior MNE      (blind k-means K=60)   — fine partition
  - network-prior sLORETA  (blind k-means K=24)   — fix attempt @ K=24
  - network-prior sLORETA  (blind k-means K=60)   — fix attempt @ K=60
  - network-prior MNE oracle Voronoi(true nodes)  — ceiling reference, NOT a
    real method; included only to bound what's achievable after correct
    partitioning.

The sLORETA-on-reduced variant is the new variable here. The prediction:
MNE on a reduced leadfield with several correlated columns (parcels in the
same cluster region) has a winner-take-all bias — the regularized
least-squares solution shrinks all but the best-aligned column toward
zero. sLORETA's per-row standardization (divide by sqrt(diag(R))) is
explicitly designed to cancel that bias, so applying it on the reduced
system should reallocate amplitude across same-cluster parcels.

Inverse-crime guards (same as 04_network_recovery.py)
-----------------------------------------------------
  - True node positions are sampled off-grid (Gaussian jitter inside each
    cluster radius); confirmed by printing nearest-grid distances.
  - Both blind k-means parcellations are computed from source coordinates
    only — no peeking at the true cluster centers or node positions.
  - Oracle Voronoi is clearly labeled and reported with a disclaimer.

Headline metric
---------------
"Within-cluster recovery R_k": recovered amplitude at the source nearest to
node k divided by the max recovered amplitude inside node k's cluster
region. R_k near 1 = node k is distinctly visible inside its cluster; R_k
near 0 = there's a brighter spot elsewhere in the cluster (fused). Count
of nodes with R_k > 0.7 = number of "distinct" nodes resolved across the
network. A method that fuses each 4-node cluster into one peak resolves
~3/12 (one per cluster); a method that splits across multiple parcels per
cluster resolves more.
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
    sample_clustered_network_positions, make_correlated_node_timecourses,
    simulate_network_scalp_data,
    parcellate_source_space, network_prior_operator,
    per_source_amplitude,
    per_node_within_cluster_recovery,
)


def _load_sanity_helpers():
    path = os.path.join(os.path.dirname(__file__), '01_sphere_sanity.py')
    spec = importlib.util.spec_from_file_location('sphere_sanity', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_source_space, mod.build_leadfield


def per_node_recovery_scores(per_source, node_positions, source_positions):
    n_max = max(per_source.max(), 1e-30)
    out = np.empty(len(node_positions))
    for i, npos in enumerate(node_positions):
        nearest = int(np.argmin(np.linalg.norm(source_positions - npos, axis=1)))
        out[i] = per_source[nearest] / n_max
    return out


def plot_slice(ax, sources, per_src_map, node_pos, cluster_ids, title,
               R_head, R_brain, y_slab_mm=6.0):
    """xz-slice; nodes colored by cluster_id."""
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
    star_colors = ['cyan', 'lime', 'magenta', 'yellow']
    in_node_slice = np.abs(node_pos[:, 1]) <= y_slab_mm
    for c in np.unique(cluster_ids):
        mask = (cluster_ids == c) & in_node_slice
        if not mask.any():
            continue
        ax.scatter(node_pos[mask, 0], node_pos[mask, 2],
                   marker='*', s=180,
                   color=star_colors[c % len(star_colors)],
                   edgecolors='black', linewidths=1.0, zorder=10,
                   label=f'cluster {c}' if title.startswith('ground') else None)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.tick_params(labelsize=8)


def main():
    print("=" * 68)
    print("Q-D cluster follow-up: 3 tight clusters x 4 nodes")
    print("=" * 68)

    # ----- 1. Head, electrodes, source space, leadfield ------------------
    model = SphericalHeadModel()
    electrodes = make_gsn_montage_64(model)
    build_source_space, build_leadfield = _load_sanity_helpers()
    print("\nBuilding source space (8mm grid)...")
    sources = build_source_space(model, spacing_mm=8.0)
    print(f"  {sources.shape[0]} source positions")
    print("Building leadfield (~1-2 min)...")
    L = build_leadfield(sources, electrodes, model, orientation='radial')
    print(f"  L shape={L.shape}, cond(L)={np.linalg.cond(L):.2e}")

    # ----- 2. Ground-truth clustered network -----------------------------
    rng = np.random.default_rng(20260524)
    cluster_centers = np.array([
        [ 45.0, 0.0,  10.0],   # shallow right
        [-30.0, 0.0,  45.0],   # superior left
        [ 10.0, 0.0,  -5.0],   # deep central
    ])
    nodes_per_cluster = 4
    node_pos, cluster_ids = sample_clustered_network_positions(
        model,
        cluster_centers_mm=cluster_centers,
        nodes_per_cluster=nodes_per_cluster,
        cluster_radius_mm=12.0,
        min_separation_mm=6.0,
        margin_mm=8.0,
        slab_y_mm=4.0,
        rng=rng,
    )
    n_nodes = len(node_pos)
    print(f"\nGround-truth: {len(cluster_centers)} clusters x "
          f"{nodes_per_cluster} nodes = {n_nodes} nodes total")
    for c, ctr in enumerate(cluster_centers):
        pts = node_pos[cluster_ids == c]
        radii = np.linalg.norm(pts, axis=1)
        pairs = [np.linalg.norm(pts[i] - pts[j])
                 for i in range(len(pts)) for j in range(i+1, len(pts))]
        print(f"  cluster {c} @ {ctr}: |radius|={radii.mean():.1f}mm, "
              f"within-cluster spacing={min(pairs):.1f}-{max(pairs):.1f}mm")

    # Inverse-crime check: nodes off the inversion grid
    nearest_grid = np.array([
        np.linalg.norm(sources - p, axis=1).min() for p in node_pos
    ])
    print(f"  off-grid offsets: min={nearest_grid.min():.2f}mm, "
          f"median={np.median(nearest_grid):.2f}mm, "
          f"max={nearest_grid.max():.2f}mm")
    if nearest_grid.min() < 0.05:
        print("  WARNING: a node lies on the inversion grid (inverse crime)")

    # Time-courses + simulated scalp data
    n_times = 256
    amps = make_correlated_node_timecourses(
        n_nodes, n_times, shared_fraction=0.3, rng=rng,
    ) * 1e-8
    inter = np.corrcoef(amps)[~np.eye(n_nodes, dtype=bool)]
    print(f"  inter-node tc correlation: mean={inter.mean():.2f}, "
          f"min={inter.min():.2f}, max={inter.max():.2f}")
    V, V_clean = simulate_network_scalp_data(
        node_pos, amps, electrodes, model,
        amplitude_snr=3.0, rng=rng,
    )
    snr_measured = (np.sqrt(np.mean(V_clean**2))
                    / np.sqrt(np.mean((V - V_clean)**2)))
    print(f"\nScalp data: V shape={V.shape}, amplitude SNR={snr_measured:.2f}")

    # ----- 3. Inverse operators ------------------------------------------
    print("\nBuilding inverse operators...")
    K_slo = sloreta_operator(L, snr=3.0)
    K_elo = eloreta_operator(L, snr=3.0, max_iter=50, verbose=False)

    labels_k24 = parcellate_source_space(sources, 24, random_state=42)
    labels_k60 = parcellate_source_space(sources, 60, random_state=42)
    K_net24_mne, _, _ = network_prior_operator(
        L, labels_k24, snr=3.0, n_parcels=24, normalize='unit_l2',
        method='mne')
    K_net60_mne, _, _ = network_prior_operator(
        L, labels_k60, snr=3.0, n_parcels=60, normalize='unit_l2',
        method='mne')
    K_net24_slo, _, _ = network_prior_operator(
        L, labels_k24, snr=3.0, n_parcels=24, normalize='unit_l2',
        method='sloreta')
    K_net60_slo, _, _ = network_prior_operator(
        L, labels_k60, snr=3.0, n_parcels=60, normalize='unit_l2',
        method='sloreta')

    voronoi_labels = np.array([
        int(np.argmin(np.linalg.norm(node_pos - p, axis=1))) for p in sources
    ])
    K_ora_mne, K_ora_parcel, _ = network_prior_operator(
        L, voronoi_labels, snr=3.0, n_parcels=n_nodes, normalize='unit_l2',
        method='mne')
    K_ora_slo, _, _ = network_prior_operator(
        L, voronoi_labels, snr=3.0, n_parcels=n_nodes, normalize='unit_l2',
        method='sloreta')

    method_specs = [
        ('sLORETA',                                   K_slo),
        ('eLORETA',                                   K_elo),
        ('network-prior MNE  (blind K=24)',           K_net24_mne),
        ('network-prior MNE  (blind K=60)',           K_net60_mne),
        ('network-prior sLOR (blind K=24)',           K_net24_slo),
        ('network-prior sLOR (blind K=60)',           K_net60_slo),
        ('network-prior MNE  (oracle Voronoi)',       K_ora_mne),
        ('network-prior sLOR (oracle Voronoi)',       K_ora_slo),
    ]

    # Per-parcel sizes printout for context
    for K, lbl in [(24, labels_k24), (60, labels_k60)]:
        sizes = np.bincount(lbl, minlength=K)
        nz = sizes[sizes > 0]
        print(f"  blind k-means K={K}: "
              f"non-empty={len(nz)}, "
              f"sizes min/median/max = "
              f"{int(nz.min())}/{int(np.median(nz))}/{int(nz.max())}, "
              f"approx parcel radius ~ {(3 * nz.mean()/(4*np.pi) * 8**3)**(1/3):.1f}mm")

    # Oracle diagnostic
    a_hat_ora = K_ora_parcel @ V
    corrs = np.array([np.corrcoef(a_hat_ora[k], amps[k])[0, 1]
                      for k in range(n_nodes)])
    print(f"\nOracle sanity: per-cell a_hat-vs-truth correlation: "
          f"median={np.median(corrs):.2f}, "
          f"min={corrs.min():.2f}, max={corrs.max():.2f}")

    # ----- 4. Reconstruction + evaluation --------------------------------
    method_results = {}
    for name, K in method_specs:
        J_hat = K @ V
        per_src = per_source_amplitude(J_hat)
        method_results[name] = dict(J_hat=J_hat, per_src=per_src)

    print("\n" + "=" * 68)
    print("Per-node detection")
    print("=" * 68)
    print(f"{'method':38s}  {'mean R':>7s}  {'min R':>6s}  "
          f"{'>0.3':>5s}  | within-cluster: "
          f"{'mean Rc':>7s}  {'>0.7':>5s}")
    for name, _ in method_specs:
        per_src = method_results[name]['per_src']
        R = per_node_recovery_scores(per_src, node_pos, sources)
        Rc = per_node_within_cluster_recovery(
            per_src, node_pos, cluster_ids, sources,
            cluster_search_radius_mm=25.0,
        )
        n_det = int((R > 0.3).sum())
        n_distinct = int((Rc > 0.7).sum())
        print(f"{name:38s}  {R.mean():7.2f}  {R.min():6.2f}  "
              f"{n_det:2d}/{n_nodes}  | "
              f"            {Rc.mean():7.2f}  {n_distinct:2d}/{n_nodes}")

    # Per-cluster mass: how does each method distribute amplitude across the
    # 3 clusters? Smearing methods should concentrate in one cluster less than
    # network-prior. Per-cluster mass = sum of per_src in cluster region /
    # total per_src. (Cluster region = within 25mm of any node in that cluster.)
    print("\nPer-cluster mass fraction (sum of per_src within 25mm of cluster)")
    print(f"{'method':38s}  " + "  ".join(f"  c{c}  " for c in range(len(cluster_centers))) + "  out-of-net")
    for name, _ in method_specs:
        per_src = method_results[name]['per_src']
        total = per_src.sum()
        cluster_frac = []
        in_any = np.zeros(len(sources), dtype=bool)
        for c in range(len(cluster_centers)):
            in_c = np.zeros(len(sources), dtype=bool)
            for p in node_pos[cluster_ids == c]:
                in_c |= (np.linalg.norm(sources - p, axis=1) <= 25.0)
            cluster_frac.append(per_src[in_c].sum() / max(total, 1e-30))
            in_any |= in_c
        out_frac = per_src[~in_any].sum() / max(total, 1e-30)
        print(f"{name:38s}  " + "  ".join(f"{x:6.1%}" for x in cluster_frac)
              + f"  {out_frac:8.1%}")

    # ----- 5. Spatial figure ---------------------------------------------
    print("\nBuilding figure...")
    R_head = model.head_radius_mm
    R_brain = model.radii_mm[0]
    names_for_plot = [name for name, _ in method_specs]

    # True per-source map for the ground-truth panel
    nearest_idx = np.array([
        int(np.argmin(np.linalg.norm(sources - p, axis=1))) for p in node_pos
    ])
    J_true = np.zeros((sources.shape[0], n_times))
    for k in range(n_nodes):
        J_true[nearest_idx[k]] += amps[k]
    true_per_src = per_source_amplitude(J_true)

    n_panels = 1 + len(names_for_plot)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.8 * n_rows))
    axes = axes.flatten()
    plot_slice(axes[0], sources, true_per_src, node_pos, cluster_ids,
               f'ground truth ({n_nodes} nodes, 3 clusters)', R_head, R_brain)
    for ax, name in zip(axes[1:n_panels], names_for_plot):
        plot_slice(ax, sources, method_results[name]['per_src'],
                   node_pos, cluster_ids, name, R_head, R_brain)
    for ax in axes[n_panels:]:
        ax.axis('off')
    for ax in axes[:n_panels]:
        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Z (mm)', fontsize=10)
    axes[0].legend(loc='upper right', fontsize=7, framealpha=0.85)
    fig.suptitle(
        f'Q-D cluster follow-up — xz-plane slice (|y|<=6mm), '
        f'3 clusters x {nodes_per_cluster} nodes. Per-panel-normalized RMS.',
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__), '..', 'figures', 'day5_network_clusters.png'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out_path}")

    print("\n" + "=" * 68)
    print("Done. Read the within-cluster Rc column above: that's whether each")
    print("method preserves multiple distinct nodes per cluster, or fuses each")
    print("cluster into a single blob.")
    print("=" * 68)


if __name__ == '__main__':
    main()
