"""
Q-D robustness sweep: is network-prior sLOR-on-reduced a plateau or a peak?

05_network_clusters.py showed network-prior sLORETA-on-reduced at K=24 beats
per-voxel sLORETA on within-cluster Rc. But that's one K, one k-means seed,
one network draw. Two failure modes to rule out before claiming the result:

  1. K=24 placed parcel boundaries that happened to carve THIS network's
     clusters favorably (soft partition-quality leak — not the hard inverse
     crime that the network-prior method guards against, but a milder "the
     blind k-means seed got lucky" effect).
  2. The win is a knife-edge at K=24 that evaporates at K=18 or K=30.

Plus the obvious Monte-Carlo concern: one cluster-center draw is not a
distribution. Re-drawing the network puts it under the same standard as
the calibration result.

What this script does
---------------------
  - Sweep K in {12, 18, 24, 36, 48, 60}
  - Sweep k-means seeds (4 of them) per K
  - Sweep network realizations (8 of them: new cluster centers, new node
    positions inside each cluster, new time courses, new noise)
  - For each (network, seed, K, method-on-reduced), apply the operator
    and compute within-cluster Rc averaged over nodes
  - Plot mean ± std across realizations × seeds, with per-voxel sLORETA
    and eLORETA as horizontal reference bands
  - Also count win-rate: fraction of (realization × seed) pairs where
    network-prior sLOR beats per-voxel sLORETA on the same realization

Decision rule
-------------
If network-prior sLOR sits clearly above per-voxel sLORETA across a broad
K range with band overlap near zero AND per-realization win-rate ≥ 80%,
the claim "network-prior sLOR beats per-voxel sLORETA for clustered
networks" is defensible. If the win only exists at K=24 or only on a
handful of realizations, it's a coincidence.
"""
from __future__ import annotations
import sys
import os
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from spherical_forward import SphericalHeadModel, make_standard_montage_64
from inverse_solvers import sloreta_operator, eloreta_operator
from network_recovery import (
    sample_clustered_network_positions, make_correlated_node_timecourses,
    simulate_network_scalp_data,
    parcellate_source_space, network_prior_operator,
    per_source_amplitude, per_node_within_cluster_recovery,
)


def _load_sanity_helpers():
    path = os.path.join(os.path.dirname(__file__), '01_sphere_sanity.py')
    spec = importlib.util.spec_from_file_location('sphere_sanity', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_source_space, mod.build_leadfield


def sample_cluster_centers(
    model: SphericalHeadModel,
    n_clusters: int = 3,
    min_separation_mm: float = 32.0,
    margin_mm: float = 18.0,
    slab_y_mm: float = 4.0,
    rng: np.random.Generator | None = None,
    max_attempts: int = 50000,
) -> np.ndarray:
    """Sample n_clusters centers inside the brain compartment with minimum
    pairwise separation. margin_mm keeps each cluster (radius ~12mm) inside
    the brain. slab_y_mm constrains centers to |y|<=slab_y_mm for the
    xz-plane slice visualization (when used)."""
    if rng is None:
        rng = np.random.default_rng(0)
    R_max = model.radii_mm[0] - margin_mm
    pts: list = []
    attempts = 0
    while len(pts) < n_clusters and attempts < max_attempts:
        attempts += 1
        cand = rng.uniform(-R_max, R_max, size=3)
        cand[1] = rng.uniform(-slab_y_mm, slab_y_mm)
        if np.linalg.norm(cand) > R_max:
            continue
        if pts and np.min(np.linalg.norm(np.array(pts) - cand, axis=1)) < min_separation_mm:
            continue
        pts.append(cand)
    if len(pts) < n_clusters:
        raise RuntimeError(
            f"Could not place {n_clusters} cluster centers in {max_attempts} attempts"
        )
    return np.array(pts)


def main():
    print("=" * 72)
    print("Q-D network-prior robustness sweep: K × kmeans seed × network draw")
    print("=" * 72)

    # ----- 1. Head, electrodes, source space, leadfield ------------------
    model = SphericalHeadModel()
    electrodes = make_standard_montage_64(model)
    build_source_space, build_leadfield = _load_sanity_helpers()
    print("\nBuilding source space (8mm grid)...")
    sources = build_source_space(model, spacing_mm=8.0)
    print(f"  {sources.shape[0]} sources")
    print("Building leadfield (~1-2 min)...")
    L = build_leadfield(sources, electrodes, model, orientation='radial')
    print(f"  L shape={L.shape}")

    # ----- 2. Per-voxel operators (depend only on L, build once) ---------
    K_slo = sloreta_operator(L, snr=3.0)
    K_elo = eloreta_operator(L, snr=3.0, max_iter=50, verbose=False)

    # ----- 3. Sweep grid -------------------------------------------------
    K_values = [12, 18, 24, 36, 48, 60]
    kmeans_seeds = [42, 7, 1234, 999]
    network_seeds = list(range(8))

    n_clusters = 3
    nodes_per_cluster = 4
    n_nodes = n_clusters * nodes_per_cluster

    # ----- 4. Precompute network-prior operators per (K, kmeans_seed, method)
    # These depend only on L and labels, not on the network realization, so
    # we build them once and reuse across realizations.
    np_methods = ('mne', 'sloreta', 'eloreta')
    print(f"\nPrecomputing network-prior operators: "
          f"{len(K_values)} × {len(kmeans_seeds)} × {len(np_methods)} methods = "
          f"{len(K_values) * len(kmeans_seeds) * len(np_methods)} operators...")
    np_operators: dict = {}
    for K in K_values:
        for sd in kmeans_seeds:
            labels = parcellate_source_space(sources, K, random_state=sd)
            for method in np_methods:
                K_voxel, _, _ = network_prior_operator(
                    L, labels, snr=3.0, n_parcels=K,
                    normalize='unit_l2', method=method,
                )
                np_operators[(K, sd, method)] = K_voxel
    print(f"  done")

    # ----- 5. Sweep over network realizations -----------------------------
    # Structures to fill:
    #   rc_pv[method_name]            -> array(n_realizations,)        per-voxel
    #   rc_np[(method, K)]            -> array(n_realizations, n_seeds) network-prior
    rc_pv = {
        'per-voxel sLORETA': np.empty(len(network_seeds)),
        'per-voxel eLORETA': np.empty(len(network_seeds)),
    }
    rc_np = {(m, K): np.empty((len(network_seeds), len(kmeans_seeds)))
             for m in np_methods for K in K_values}

    # Also keep cluster-center positions and node positions for diagnostics
    network_summary = []

    print(f"\nRunning {len(network_seeds)} network realizations × "
          f"{len(kmeans_seeds)} k-means seeds...")
    for ni, nseed in enumerate(network_seeds):
        rng = np.random.default_rng(20260000 + nseed)
        try:
            cluster_centers = sample_cluster_centers(
                model, n_clusters=n_clusters,
                min_separation_mm=32.0, margin_mm=18.0,
                slab_y_mm=4.0, rng=rng,
            )
            node_pos, cluster_ids = sample_clustered_network_positions(
                model, cluster_centers_mm=cluster_centers,
                nodes_per_cluster=nodes_per_cluster,
                cluster_radius_mm=12.0, min_separation_mm=6.0,
                margin_mm=8.0, slab_y_mm=4.0, rng=rng,
            )
        except RuntimeError as e:
            print(f"  realization {ni+1} (seed {nseed}) sampling FAILED: {e}")
            # Fill with NaN and continue
            for name in rc_pv:
                rc_pv[name][ni] = np.nan
            for key in rc_np:
                rc_np[key][ni, :] = np.nan
            network_summary.append(None)
            continue
        amps = make_correlated_node_timecourses(
            n_nodes, 256, shared_fraction=0.3, rng=rng,
        ) * 1e-8
        V, _ = simulate_network_scalp_data(
            node_pos, amps, electrodes, model,
            amplitude_snr=3.0, rng=rng,
        )

        network_summary.append({
            'centers': cluster_centers,
            'node_pos': node_pos,
            'cluster_ids': cluster_ids,
        })

        # Per-voxel methods (kmeans-seed-independent)
        for K_op, name in [(K_slo, 'per-voxel sLORETA'),
                           (K_elo, 'per-voxel eLORETA')]:
            J_hat = K_op @ V
            per_src = per_source_amplitude(J_hat)
            Rc = per_node_within_cluster_recovery(
                per_src, node_pos, cluster_ids, sources,
                cluster_search_radius_mm=25.0,
            )
            rc_pv[name][ni] = float(np.nanmean(Rc))

        # Network-prior methods
        for si, sd in enumerate(kmeans_seeds):
            for K in K_values:
                for method in np_methods:
                    K_op = np_operators[(K, sd, method)]
                    J_hat = K_op @ V
                    per_src = per_source_amplitude(J_hat)
                    Rc = per_node_within_cluster_recovery(
                        per_src, node_pos, cluster_ids, sources,
                        cluster_search_radius_mm=25.0,
                    )
                    rc_np[(method, K)][ni, si] = float(np.nanmean(Rc))
        print(f"  realization {ni+1}/{len(network_seeds)} done "
              f"(per-voxel sLORETA Rc = {rc_pv['per-voxel sLORETA'][ni]:.3f})")

    # ----- 6. Summary stats ----------------------------------------------
    print("\n" + "=" * 72)
    print("Summary — within-cluster Rc, mean ± std")
    print("=" * 72)
    for name, vals in rc_pv.items():
        v = vals[~np.isnan(vals)]
        print(f"  {name:32s}  mean={v.mean():.3f}  std={v.std():.3f}  "
              f"n={len(v)}")
    print()
    pv_slo = rc_pv['per-voxel sLORETA']
    pv_elo = rc_pv['per-voxel eLORETA']
    # Each network-prior method is compared against its same-family per-voxel
    # baseline (the natural Greg-test). MNE-on-reduced has no per-voxel MNE
    # row in this sweep, so we keep its historical comparison vs per-voxel
    # sLORETA (a documented bound on what a per-voxel solver achieves).
    baseline_for = {
        'sloreta': ('per-voxel sLORETA', pv_slo),
        'eloreta': ('per-voxel eLORETA', pv_elo),
        'mne':     ('per-voxel sLORETA', pv_slo),
    }
    print(f"  {'network-prior method':28s}  K   mean   std   n   "
          f"win-rate vs same-family per-voxel baseline")
    for method in ('sloreta', 'eloreta', 'mne'):
        baseline_name, pv = baseline_for[method]
        for K in K_values:
            arr = rc_np[(method, K)]   # (n_real, n_seeds)
            flat = arr[~np.isnan(arr)]
            # Per-realization-per-seed comparison against the per-voxel value
            # for that same realization
            pv_broadcast = np.broadcast_to(pv[:, None], arr.shape)
            valid = ~np.isnan(arr) & ~np.isnan(pv_broadcast)
            n_compared = int(valid.sum())
            n_wins = int(((arr > pv_broadcast) & valid).sum())
            wr = n_wins / max(n_compared, 1)
            print(f"  np-{method.upper():7s} on reduced   "
                  f"K={K:3d}  {flat.mean():.3f}  {flat.std():.3f}  "
                  f"{len(flat):3d}   {n_wins:3d}/{n_compared}  ({wr:.0%}) "
                  f"vs {baseline_name}")
        print()

    # ----- 7. Plot 1: mean ± std band vs K -------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    Kx = np.array(K_values)

    # Per-voxel reference bands
    for name, color, ls in [
        ('per-voxel sLORETA', '#264653', '--'),
        ('per-voxel eLORETA', '#6A6D77', ':'),
    ]:
        vals = rc_pv[name][~np.isnan(rc_pv[name])]
        m, s = vals.mean(), vals.std()
        ax.axhline(m, color=color, linestyle=ls, linewidth=1.6,
                   label=f'{name}  ({m:.2f}±{s:.2f}, n={len(vals)})')
        ax.fill_between([K_values[0] - 2, K_values[-1] + 2],
                        m - s, m + s, color=color, alpha=0.10)

    # Network-prior curves
    for method, color, marker, label in [
        ('sloreta', '#2A9D8F', 'o', 'network-prior sLOR on reduced'),
        ('eloreta', '#8E44AD', '^', 'network-prior eLOR on reduced'),
        ('mne',     '#E76F51', 's', 'network-prior MNE on reduced'),
    ]:
        means = np.array([np.nanmean(rc_np[(method, K)]) for K in K_values])
        stds = np.array([np.nanstd(rc_np[(method, K)]) for K in K_values])
        ns = np.array([(~np.isnan(rc_np[(method, K)])).sum() for K in K_values])
        ax.plot(Kx, means, '-' + marker, color=color, linewidth=2.2,
                markersize=8, label=f'{label}  (n={ns[0]} per K)')
        ax.fill_between(Kx, means - stds, means + stds, color=color, alpha=0.22)

    ax.set_xlabel('blind k-means K (parcels)', fontsize=11)
    ax.set_ylabel('within-cluster Rc (mean across nodes)', fontsize=11)
    ax.set_title(
        f'Within-cluster recovery Rc vs K — '
        f'{len(network_seeds)} network draws × {len(kmeans_seeds)} k-means seeds.\n'
        f'Shaded = ±1 std across all (realization × seed) pairs.',
        fontsize=11,
    )
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_values)
    ax.set_xlim(K_values[0] - 2, K_values[-1] + 2)
    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__), '..', 'figures', 'day6_network_prior_sweep.png'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSweep figure: {out_path}")

    # ----- 8. Plot 2: per-realization win-rate heatmaps ------------------
    # Each cell = win-rate across k-means seeds for network-prior method >
    # same-family per-voxel baseline on this realization. Two panels:
    # sLOR-vs-sLOR (the original Q-D comparison) and eLOR-vs-eLOR (the
    # Greg test — does the prior help the *good* inverse?).
    def _winrate_grid(method_key: str, baseline_vec: np.ndarray) -> np.ndarray:
        arr = np.full((len(network_seeds), len(K_values)), np.nan)
        for ni in range(len(network_seeds)):
            if np.isnan(baseline_vec[ni]):
                continue
            for ki, K in enumerate(K_values):
                seeds_arr = rc_np[(method_key, K)][ni, :]
                valid = ~np.isnan(seeds_arr)
                if not valid.any():
                    continue
                wins = int((seeds_arr[valid] > baseline_vec[ni]).sum())
                arr[ni, ki] = wins / int(valid.sum())
        return arr

    win_slo = _winrate_grid('sloreta', pv_slo)
    win_elo = _winrate_grid('eloreta', pv_elo)

    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))
    for ax_w, win_arr, title in [
        (axes2[0], win_slo, 'np-sLOR > per-voxel sLORETA'),
        (axes2[1], win_elo, 'np-eLOR > per-voxel eLORETA'),
    ]:
        im = ax_w.imshow(win_arr, aspect='auto', cmap='RdYlGn',
                         vmin=0, vmax=1, interpolation='nearest')
        for ni in range(len(network_seeds)):
            for ki in range(len(K_values)):
                v = win_arr[ni, ki]
                txt = 'n/a' if np.isnan(v) else f'{v:.0%}'
                ax_w.text(ki, ni, txt, ha='center', va='center', fontsize=9,
                          color='black' if 0.25 < v < 0.75 else 'white')
        ax_w.set_xticks(range(len(K_values)))
        ax_w.set_xticklabels([f'K={K}' for K in K_values])
        ax_w.set_yticks(range(len(network_seeds)))
        ax_w.set_yticklabels([f'net seed {s}' for s in network_seeds])
        ax_w.set_title(
            f'{title}\n(fraction of {len(kmeans_seeds)} k-means seeds per cell)',
            fontsize=10,
        )
        plt.colorbar(im, ax=ax_w, label='win fraction')
    plt.tight_layout()
    out_path2 = os.path.join(
        os.path.dirname(__file__), '..', 'figures',
        'day6_network_prior_winrate.png'
    )
    plt.savefig(out_path2, dpi=120, bbox_inches='tight')
    plt.close(fig2)
    print(f"Win-rate heatmap: {out_path2}")

    # ----- 9. Verdict ----------------------------------------------------
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    def _verdict(method_key: str, baseline_name: str, baseline_vec: np.ndarray):
        b_mean = float(np.nanmean(baseline_vec))
        b_std = float(np.nanstd(baseline_vec))
        label = method_key.upper()
        print(f"\n  [{label}-on-reduced vs {baseline_name}]")
        print(f"    {baseline_name} baseline: {b_mean:.3f} ± {b_std:.3f}")
        print(f"    K values where np-{label} mean > {baseline_name} mean:")
        for K in K_values:
            m = float(np.nanmean(rc_np[(method_key, K)]))
            s = float(np.nanstd(rc_np[(method_key, K)]))
            flag = (' <-- exceeds baseline by >1 std'
                    if m > b_mean + b_std else '')
            bf = '+' if m > b_mean else ' '
            print(f"      K={K:3d}: {m:.3f} ± {s:.3f}  ({bf}vs baseline){flag}")
        overall_wins = sum(
            int(((rc_np[(method_key, K)] > baseline_vec[:, None])
                 & ~np.isnan(rc_np[(method_key, K)]))[
                ~np.isnan(rc_np[(method_key, K)])].sum())
            for K in K_values
        )
        overall_total = sum(
            int((~np.isnan(rc_np[(method_key, K)])).sum()) for K in K_values
        )
        print(f"    Overall np-{label} > {baseline_name} win-rate "
              f"(all K × seeds × realizations): {overall_wins}/{overall_total} = "
              f"{overall_wins/max(overall_total,1):.0%}")

    # The Greg test: does the region-grouping prior help the *good* inverse?
    # Reported alongside the original sLORETA comparison.
    _verdict('sloreta', 'per-voxel sLORETA', pv_slo)
    _verdict('eloreta', 'per-voxel eLORETA', pv_elo)
    print("=" * 72)


if __name__ == '__main__':
    main()
