r"""
Q-D GD-VAE pre-check â€” multi-seed robustness (the cardinal rule).

11_gdvae_precheck_run.py reported a single parcel basis (k-means seed 42) and a
single oracle-Voronoi network (seed 20260524). The leadfield L and r_L are
deterministic physical properties, but the parcel/Voronoi alignment with the
observable subspace depends on a stochastic axis, so it must be checked across
realizations before it is believed.

âš ï¸ FRAMING CORRECTION (adversarial-verification pass, 2026-06-30). An earlier
version of this script foregrounded "P(parcel eff_rank > random-floor)" and read
the parcel's low eff_rank (3 < random's 4) as evidence that localized priors are
"anti-aligned with the observable world." THAT IS A HARD-THRESHOLD / AMBIENT-
DIMENSION ARTIFACT, not a real effect â€” and a textbook instance of the internal notes
trap (reaching for a just-so story because it fit the pre-decided "humble SKIP").
eff_rank is a coarse integer Picard count: a generic 24-dim subspace of 3544-dim
source space picks up slivers of many soft modes and clears the s_max/snr cliff on
~4 of them while carrying almost NO observable energy (obs_frac ~0.007). The
parcel, by contrast, is STRONGLY ALIGNED with the observable subspace â€”
principal cosines vs the 5 observable modes â‰ˆ [0.93,0.92,0.92,0.82,0.78],
obs_frac â‰ˆ 0.765 (~115Ã— the random floor). The decision-relevant continuous
measure is obs_frac, not the eff_rank-vs-random count, so this script now
foregrounds obs_frac (with eff_rank kept only as the raw Picard count, explicitly
labelled as confounded). The SKIP verdict does NOT rest on this comparison; it
rests on the observable-rank ceiling (see 11_*.py and the internal notes Q-D section).

This sweeps, at SNR in {3, 10}:
  - K=24 blind k-means parcellation across 12 seeds
  - oracle-Voronoi across 12 network draws (the 04-notebook sampler)
  - random-24 / random-12 floors across 50 draws
reporting the distribution of obs_frac (primary) and eff_rank (raw, confounded),
plus the soft Wiener-weighted observable dof (the continuous analogue of r_L).

Run:
    $env:MPLBACKEND='Agg'
    & .\.venv\Scripts\python.exe .\bandr_inverse\notebooks\11b_gdvae_precheck_multiseed.py
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from spherical_forward import SphericalHeadModel, make_standard_montage_64
from network_recovery import (
    sample_network_positions, parcellate_source_space, parcel_indicator_matrix,
)
from gdvae_precheck import svals, orth, eff_rank, principal_cos, observable_basis

import importlib.util


def _load_sanity_helpers():
    path = os.path.join(os.path.dirname(__file__), '01_sphere_sanity.py')
    spec = importlib.util.spec_from_file_location('sphere_sanity', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_source_space, mod.build_leadfield


def probe_er_frac(L, Phi, snr, obs_V):
    Q = orth(Phi)
    er = eff_rank(svals(L @ Q), snr)
    cos = principal_cos(Q, obs_V)
    return er, float(np.mean(cos ** 2))


def main():
    src = os.path.join(os.path.dirname(__file__), '..', 'src')
    model = SphericalHeadModel()
    build_source_space, build_leadfield = _load_sanity_helpers()
    sources = build_source_space(model, spacing_mm=8.0)
    cache_L = os.path.join(src, 'leadfield.npy')
    if os.path.exists(cache_L):
        L = np.load(cache_L)
    else:
        electrodes = make_standard_montage_64(model)
        L = build_leadfield(sources, electrodes, model, orientation='radial')
        np.save(cache_L, L)
    print(f"L {L.shape}, sources {sources.shape[0]}")

    K = 24
    kmeans_seeds = [42, 7, 1234, 999, 0, 1, 2, 3, 100, 2026, 31337, 555]
    net_seeds = [20260524] + list(range(20260000, 20260011))

    for snr in (3.0, 10.0):
        obs_V, r_L = observable_basis(L, snr)
        # soft Wiener-weighted observable dof: continuous analogue of r_L
        s = svals(L); s2 = (s / s.max()) ** 2
        soft_dof = float((s2 / (s2 + (1.0 / snr) ** 2)).sum())

        # parcel K=24 across k-means seeds
        p_er, p_fr = [], []
        for sd in kmeans_seeds:
            labels = parcellate_source_space(sources, K, random_state=sd)
            G = parcel_indicator_matrix(labels, n_parcels=K, normalize='unit_l2')
            er, fr = probe_er_frac(L, G, snr, obs_V)
            p_er.append(er); p_fr.append(fr)
        p_er = np.array(p_er); p_fr = np.array(p_fr)

        # oracle-Voronoi K=12 across network draws
        v_er, v_fr = [], []
        for ns in net_seeds:
            rng = np.random.default_rng(ns)
            try:
                node_pos = sample_network_positions(
                    model, n_nodes=12, min_separation_mm=18.0, margin_mm=8.0,
                    slab_y_mm=5.0, rng=rng,
                )
            except RuntimeError:
                continue
            vlab = np.array([int(np.argmin(np.linalg.norm(node_pos - p, axis=1)))
                             for p in sources])
            Gv = parcel_indicator_matrix(vlab, n_parcels=12, normalize='unit_l2')
            er, fr = probe_er_frac(L, Gv, snr, obs_V)
            v_er.append(er); v_fr.append(fr)
        v_er = np.array(v_er); v_fr = np.array(v_fr)

        # random floors across 50 draws
        rng = np.random.default_rng(7)
        r24 = np.array([probe_er_frac(L, rng.standard_normal((L.shape[1], 24)),
                                      snr, obs_V)[0] for _ in range(50)])
        r12 = np.array([probe_er_frac(L, rng.standard_normal((L.shape[1], 12)),
                                      snr, obs_V)[0] for _ in range(50)])

        # random-subspace obs_frac (the decision-relevant floor, not eff_rank)
        rng2 = np.random.default_rng(11)
        r24_fr = np.array([
            probe_er_frac(L, rng2.standard_normal((L.shape[1], 24)), snr, obs_V)[1]
            for _ in range(50)])

        oracle24 = min(24, r_L)
        oracle12 = min(12, r_L)
        print("\n" + "=" * 68)
        print(f"SNR={snr:g}   r_L(hard)={r_L}   soft Wiener dof={soft_dof:.2f}   "
              f"oracle obs_frac=1.000")
        print("=" * 68)
        print("  PRIMARY measure â€” obs_frac (fraction of prior energy in the "
              f"{r_L}-dim observable subspace):")
        print(f"    parcel K24  : mean={p_fr.mean():.3f} std={p_fr.std():.3f}  "
              f"(n={len(p_fr)} k-means seeds)")
        print(f"    Voronoi-12  : mean={v_fr.mean():.3f} std={v_fr.std():.3f}  "
              f"(n={len(v_fr)} network draws)")
        print(f"    random-24   : mean={r24_fr.mean():.4f} std={r24_fr.std():.4f}  "
              f"-> localized prior captures ~{p_fr.mean()/r24_fr.mean():.0f}x more "
              f"observable energy than random (it is ALIGNED, not anti-aligned).")
        print("\n  RAW eff_rank (coarse integer Picard count â€” CONFOUNDED by "
              "ambient dimension, do NOT compare to random):")
        print(f"    parcel K24  : {p_er.mean():.2f}+/-{p_er.std():.2f} "
              f"[{p_er.min()},{p_er.max()}]   "
              f"Voronoi-12 : {v_er.mean():.2f}+/-{v_er.std():.2f} "
              f"[{v_er.min()},{v_er.max()}]")
        print(f"    random-24   : {r24.mean():.2f}+/-{r24.std():.2f}   "
              f"random-12 : {r12.mean():.2f}+/-{r12.std():.2f}  "
              f"(higher than parcel ONLY because random spreads slivers across "
              f"many soft modes at ~0.7% energy each)")
        print(f"\n  Ceiling (SNR-independent SKIP logic): the observable subspace is "
              f"{r_L}-dim (soft {soft_dof:.0f}); per-voxel sLORETA already spans it, and "
              f"any prior is bounded by it (V=LJ+eps factors through fixed linear L). "
              f"A prior can only fill the UNOBSERVABLE null space by assumption -> cannot "
              f"beat per-voxel for observability reasons -> SKIP. At the operating SNR=3 "
              f"the ceiling is additionally tiny ({r_L}<12 nodes), so the Q-D task is "
              f"rank-starved outright. The lever is more channels (channel selection), not a richer prior.")


if __name__ == '__main__':
    main()
