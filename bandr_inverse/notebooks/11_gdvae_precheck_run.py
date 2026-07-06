r"""
Q-D GD-VAE pre-check driver — run the training-free observability bound on the
REAL spherical leadfield used by the Q-D network-recovery experiments.

This wires the actual construction (no np.load stubs, no synthetic --demo
leadfield) into src/gdvae_precheck.py's bracket:

  raw L          -> SNR-limited observable rank of L (the ceiling, r_L)
  oracle-d       -> top-d right singular vectors of L (best ANY d-dim linear
                    prior can do; reached only by spread/superficial patterns)
  parcel (K=24)  -> the localized linear prior that already hit the 34% wall
  oracle-Voronoi -> the BEST PHYSICAL linear prior (each true node = its own
                    cell); the d=12 prior that actually hit the overlap floor
  random-d       -> a generic d-subspace (the floor)

Everything is rebuilt from the same deterministic helpers the Q-D notebooks
use, so the leadfield, the K=24 blind k-means parcellation (seed 42), and the
12-node oracle-Voronoi network (seed 20260524) are bit-identical to 04/06.

Run:
    $env:MPLBACKEND='Agg'
    & .\.venv\Scripts\python.exe .\bandr_inverse\notebooks\11_gdvae_precheck_run.py
"""
from __future__ import annotations
import sys
import os
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from spherical_forward import SphericalHeadModel, make_standard_montage_64
from network_recovery import (
    sample_network_positions, parcellate_source_space, parcel_indicator_matrix,
)
from gdvae_precheck import svals, orth, eff_rank, principal_cos, observable_basis


def _load_sanity_helpers():
    """build_source_space + build_leadfield from 01_sphere_sanity.py."""
    path = os.path.join(os.path.dirname(__file__), '01_sphere_sanity.py')
    spec = importlib.util.spec_from_file_location('sphere_sanity', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_source_space, mod.build_leadfield


def full_probe(name, L, Phi, snr, obs_V):
    """Richer probe: returns (eff_rank, cond(L_eff), observable_frac, d)."""
    Q = orth(Phi)
    s_eff = svals(L @ Q)
    er = eff_rank(s_eff, snr)
    cond = s_eff.max() / max(s_eff.min(), 1e-300)
    cos = principal_cos(Q, obs_V)
    obs_frac = float(np.mean(cos ** 2))
    return dict(name=name, d=Q.shape[1], eff_rank=er, cond=cond,
               obs_frac=obs_frac)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    cache_L = os.path.join(out_dir, 'leadfield.npy')
    cache_G = os.path.join(out_dir, 'parcel_G_K24.npy')
    cache_Gora = os.path.join(out_dir, 'oracle_voronoi_G_K12.npy')

    print("=" * 72)
    print("Q-D GD-VAE pre-check — real spherical leadfield")
    print("=" * 72)

    model = SphericalHeadModel()
    electrodes = make_standard_montage_64(model)
    build_source_space, build_leadfield = _load_sanity_helpers()

    print("\nBuilding source space (8mm grid)...")
    sources = build_source_space(model, spacing_mm=8.0)
    print(f"  {sources.shape[0]} sources")

    if os.path.exists(cache_L):
        L = np.load(cache_L)
        print(f"Loaded cached leadfield {L.shape} from {cache_L}")
    else:
        print("Building leadfield (~1-2 min)...")
        L = build_leadfield(sources, electrodes, model, orientation='radial')
        np.save(cache_L, L)
        print(f"  L shape={L.shape}, saved to {cache_L}")

    assert L.shape[0] == electrodes.shape[0] == 64, "expected 64 electrodes"
    assert L.shape[1] == sources.shape[0], "L cols must match #sources"

    # ---- parcel basis G (K=24, blind k-means seed 42, unit_l2) -- matches 04
    K = 24
    labels = parcellate_source_space(sources, K, random_state=42)
    G = parcel_indicator_matrix(labels, n_parcels=K, normalize='unit_l2')
    np.save(cache_G, G)

    # ---- oracle-Voronoi G (12-node network, seed 20260524) -- matches 04 ----
    rng = np.random.default_rng(20260524)
    node_pos = sample_network_positions(
        model, n_nodes=12, min_separation_mm=18.0, margin_mm=8.0,
        slab_y_mm=5.0, rng=rng,
    )
    voronoi_labels = np.array([
        int(np.argmin(np.linalg.norm(node_pos - p, axis=1))) for p in sources
    ])
    G_ora = parcel_indicator_matrix(voronoi_labels, n_parcels=12,
                                    normalize='unit_l2')
    np.save(cache_Gora, G_ora)

    print(f"\nBases:  parcel G {G.shape} (K={K}, blind k-means seed 42)")
    print(f"        oracle-Voronoi G {G_ora.shape} (12 nodes, seed 20260524)")
    print(f"  full cond(L)            = {np.linalg.cond(L):.3e}")
    print(f"  cond(L_net) parcel K24  = "
          f"{np.linalg.cond(L @ orth(G)):.3e}")
    print(f"  cond(L_net) oracle-Voro = "
          f"{np.linalg.cond(L @ orth(G_ora)):.3e}")

    # ---- full singular spectrum of L (the cliff) ------------------------
    s_full = svals(L)
    print(f"\nSingular spectrum of L (top 30 of {s_full.size}):")
    norm_s = s_full / s_full.max()
    for j in range(0, min(30, s_full.size), 5):
        chunk = " ".join(f"{norm_s[j+t]:.2e}" if j + t < s_full.size else ""
                         for t in range(5))
        print(f"  s[{j:2d}:{j+5:2d}]/s0 = {chunk}")

    # ---- the bracket, swept over SNR ------------------------------------
    rng_rand = np.random.default_rng(1)
    for snr in (3.0, 10.0, 30.0):
        obs_V, r_L = observable_basis(L, snr)
        d = K
        oracle24 = full_probe("oracle-d=24 (best 24-sub)", L,
                              obs_V[:, :min(d, obs_V.shape[1])], snr, obs_V)
        parcel = full_probe("parcel basis (K=24)", L, G, snr, obs_V)
        oracle12 = full_probe("oracle-d=12 (best 12-sub)", L,
                              obs_V[:, :min(12, obs_V.shape[1])], snr, obs_V)
        voro = full_probe("oracle-Voronoi (K=12)", L, G_ora, snr, obs_V)
        rand24 = int(np.median([
            full_probe("r", L, rng_rand.standard_normal((L.shape[1], 24)),
                       snr, obs_V)['eff_rank'] for _ in range(7)]))
        rand12 = int(np.median([
            full_probe("r", L, rng_rand.standard_normal((L.shape[1], 12)),
                       snr, obs_V)['eff_rank'] for _ in range(7)]))

        print("\n" + "-" * 72)
        print(f"SNR = {snr:g}   |   SNR-limited observable rank r_L = "
              f"{r_L} of 64")
        print("-" * 72)
        hdr = f"  {'basis':<26} {'d':>3} {'eff_rank':>8} {'obs_frac':>9} {'cond(L_eff)':>13}"
        print(hdr)
        for p in (oracle24, parcel, oracle12, voro):
            print(f"  {p['name']:<26} {p['d']:>3} {p['eff_rank']:>8} "
                  f"{p['obs_frac']:>9.3f} {p['cond']:>13.2e}")
        print(f"  {'random-d=24 (floor)':<26} {24:>3} {rand24:>8}")
        print(f"  {'random-d=12 (floor)':<26} {12:>3} {rand12:>8}")

        gap24 = oracle24['eff_rank'] - parcel['eff_rank']
        gap12 = oracle12['eff_rank'] - voro['eff_rank']
        print(f"\n  VERDICT numbers @ SNR={snr:g}:")
        print(f"    r_L (ceiling)                 = {r_L}")
        print(f"    gap_24  = oracle24 - parcelK24 = "
              f"{oracle24['eff_rank']} - {parcel['eff_rank']} = {gap24}")
        print(f"    gap_12  = oracle12 - VoronoiK12 = "
              f"{oracle12['eff_rank']} - {voro['eff_rank']} = {gap12}")
        verdict = ("SKIP (parcel sits at oracle ceiling)" if gap24 <= 1
                   else f"headroom {gap24} dims (mostly un-physical)")
        print(f"    -> {verdict}")

    print("\n" + "=" * 72)
    print("Saved arrays: leadfield.npy, parcel_G_K24.npy, "
          "oracle_voronoi_G_K12.npy (in src/)")
    print("=" * 72)


if __name__ == '__main__':
    main()
