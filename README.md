# EEG/tACS Inverse Methods — Validation Pipeline

Simulation and validation code for an EEG source-localization and
transcranial alternating current stimulation (tACS) targeting pipeline.
Establishes which inverse methods recover sources reliably under
realistic noise, quantifies why a forward model is necessary for
multi-channel stimulation, shows that active calibration enables
robust open-loop field targeting on a realistic head geometry, and
characterizes the fundamental **observability ceiling** that bounds any
read-side inverse — a fixed handful of observable spatial degrees of
freedom that no richer prior, parcellation, or spectral trick can enlarge.

This repo is the methods/validation slice of a larger system. The
device-faithful control law simulation and the electrode-subset
optimization layer are not included here (see *What's not in this repo*
below).

## What's in here

Five validation experiments, an observability-ceiling analysis, and one
motivation figure. Each is a self-contained script (in
`bandr_inverse/notebooks/` or `bandr_inverse/qe/`) that produces a
figure in `bandr_inverse/figures/`.

### Q-A — sphere sanity check
`01_sphere_sanity.py` → `figures/day1_sphere_sanity.png`

Inverse methods (MNE, wMNE, sLORETA, eLORETA) recover known dipoles in
a 4-shell analytic sphere with the Ary/de Munck forward. eLORETA
matches sLORETA on shallow sources and beats it 3.5× on a deep one
(depth-independence). The unit test that establishes the inversion
stack is correctly implemented; a regression here invalidates everything
downstream.

### Q-B — realistic geometry
`02_realistic_geom.py`, `10_orientation_sweep.py` →
`figures/q_b_realistic_geom.png`, `figures/q_b_orientation_sweep.png`

Same solvers on a BEM head model (MNE-Python `sample` subject).
Surfaces a near-vertex source mislocalization (~62mm error). The
orientation sweep corrects a common misconception: the failure is
**not** "radial dipoles are weak for EEG" — that's the MEG result
(Sarvas 1987: radial dipoles produce zero magnetic field outside a
sphere). In spherical EEG the radial dipole is *not* weak: the de Munck
series weights the radial term by `n` against the tangential term's `1`
per series order, but integrated over the montage the radial and
tangential scalp signals come out roughly equal (~1:1 RMS, flat across
depth). So radial sources are well-observed at the scalp, and the vertex
failure is realistic-anatomy specific (cortical normals not aligning
with head-center radials, sulcal geometry, skull thickness variation),
not generic EEG physics.

### Q-C — active leadfield calibration
`03_system_id.py`, `08_skull_sweep.py` →
`figures/q_c_system_id.png`, `figures/q_c_skull_sweep.png`

Active tACS calibration recovers skull conductivity to <0.3% from 5
calibration measurements (a ~5000:1 over-determined fit at ~20dB SNR —
the expected CRLB regime). Calibrated open-loop field targeting then
delivers ~85% of feasible at the target, flat across skull-conductivity
scale [0.6, 1.4] (52× spread reduction vs uncalibrated; confirmed
across 5 seeds, σ ~0.1pp). The flatness is structurally forced:
calibration recovers the conductivity scalar to <0.3% at every scale,
so open-loop-through-the-calibrated-leadfield hits the regularization
floor regardless.

The closed-loop targeting experiment in `closed_loop.py` is an
instructive negative: closing the loop through source-localization
feedback delivers only ~2% of target field at a focal deep target. The
source-localization observer's resolution-matrix diagonal
`R[t,t] ≈ 0.12` creates a null space that traps the regularized
controller at a spread-out, low-amplitude fixed point. Calibration does
not rescue this — the smearing is intrinsic to source-localization on
this geometry, not to leadfield mismatch. Architectural takeaway:
calibrate, then run open-loop through the calibrated leadfield; do
*not* close the loop through source-localization feedback for focal
targets.

Caveats kept load-bearing: the robustness claim is bounded ([0.6, 1.4]
skull-scale range, not unbounded); and the calibration parameterization
is a single global skull scalar, matching the form of the ground-truth
perturbation. Model-form robustness (regional/anisotropic skull) is a
separate question.

### Q-D — network-pattern recovery
`04_network_recovery.py`, `05_network_clusters.py`,
`06_network_prior_sweep.py` → `day4_network_recovery*.png`,
`day5_network_clusters.png`, `day6_network_prior_*.png`

Blind k-means parcellation + network-prior operators on a reduced
`L @ G` system, with the partition defined blind to ground-truth source
locations (anti-inverse-crime guard). **Honest result after multi-seed
sweep:** the network-prior does *not* beat per-voxel sLORETA at 64
electrodes — sLORETA-on-reduced runs ~3% below per-voxel at every K,
with a 34% win-rate across 8 networks × 4 partition seeds × K ∈
{8, 16, 24, 32}. An initial single-shot K=24 result (Rc 0.93 vs 0.83)
was a favorable noise draw. Per-voxel sLORETA already recovers
distributed sources; sub-cluster splits are rank-limited at 64
electrodes regardless of algorithm (an oracle-Voronoi probe doesn't
beat the blind partition either). The binding constraint is
observability, not algorithmic.

### The observability ceiling — why a richer prior can't help
`qe/realistic_rank.py`, `qe/realistic_rank_plot.py`,
`notebooks/11_gdvae_precheck_run.py`, `notebooks/11b_gdvae_precheck_multiseed.py`,
`freq_leadfield_sim.py`

Q-D's "observability, not algorithm" conclusion is a specific,
quantifiable claim: at realistic SNR the 64-channel scalp leadfield `L`
transmits only a handful of effective spatial degrees of freedom. Its
singular-value spectrum falls off a cliff — dropping below the SNR floor
after ~3–5 modes (`r_L ≈ 3–5`). Everything a linear inverse can recover
lives in that observable subspace; the rest is null space. Three
independent checks triangulate the same ceiling:

**Realistic-anatomy rank** (`realistic_rank*.py`). The `r_L ≈ 3–5` cliff
is not an artifact of the idealized sphere. An SVD of a realistic
finite-element-method leadfield — heterogeneous skull, free source
orientation, more channels — sits in the same handful-of-modes regime.
Realism *lowers* the ceiling if anything; it does not raise it.

**Prior expressiveness — the GD-VAE pre-check** (`11_gdvae_precheck_run.py`,
`11b_*`). Before building a *GD-VAE* source prior — a Geometric Dynamic
Variational Autoencoder (Lopez & Atzberger 2022, arXiv:2206.05183; publ.
*J. Comput. Phys.* 2025): a variational autoencoder (Kingma & Welling
2013, *Auto-Encoding Variational Bayes*, arXiv:1312.6114) whose latent
lives on a specified geometric/topological manifold, e.g. a torus — here
proposed as a *nonlinear* replacement for a linear source prior like
sLORETA — we ran a training-free go/no-go: *could
such a model even beat linear localization here?* The answer is no, and
the argument needs no training. The measurement factors through a fixed
linear map, `V = L·J + ε`. By the **data-processing inequality** (Cover &
Thomas, *Elements of Information Theory*), no estimator `Ĵ = f(V)` —
linear, nonlinear, manifold, or GD-VAE — can recover more about `J` than
`V` carries, which is bounded by `L`'s SNR-observable subspace. A
nonlinear decoder `J = g(z)` enters the forward model only as `L·g(z)`,
so its reachable directions still lie inside `colspace(L)`. Per-voxel
sLORETA already spans that subspace, so a richer prior can only fill the
*unobservable null space by assumption* — precisely the inverse-crime-
adjacent failure that produced Q-D's 34% wall, not an observability gain.
**Decision: skip the build.** The lever that raises `r_L` is more (or
better-placed) channels, not a more expressive prior.

**Spectral coloring** (`freq_leadfield_sim.py`, with
`freq_conductivity_review.md`). A third axis: does frequency-dependent
conductivity/permittivity open new observable directions per band — a
"colored-filters" resolution gain? No. Frequency only reweights the same
spatial (Legendre) basis through the per-degree shell gain `gₙ(ω)`; the
per-band leadfields stay near-collinear, and stacking nine bands adds
zero resolvable dimensions at every realistic SNR. (Permittivity is a
write-side dose-accuracy concern, not a read-side resolution lever.)

**The unification.** Three orthogonal levers — a richer prior, a
parcellation (Q-D), and multi-band spectral coloring — each fail to
enlarge the same low-dimensional observable subspace. The 64-channel
scalp leadfield has a fixed handful of observable spatial degrees of
freedom; only channel count and placement move it. This is a
property-of-the-physics floor, so — like Q0 — it hardens under scrutiny
rather than flaking.

### Q0 — locality motivation
`07_locality_motivation.py` → `figures/q0_locality_motivation.png`

Quantifies why a forward model is necessary for multi-channel
stimulation. Under a linear-response stand-in, the stim→measurement
transfer is `M = L Lᵀ`. Locality fraction
`|M[i,i]| / Σⱼ |M[j,i]| ≈ 0.04` — the diagonal is ~1/25 of the column
sum (mean off-diagonal magnitude is 0.38× diagonal, ×63 off-diagonals).
A model-free "stim where the error is" controller would misroute ~96%
of its effect, and has no stability justification against the
off-diagonal coupling. Montage-robust (std 0.0001 across 5
realizations) — this is a skull-low-pass property, not a montage
artifact. A property-of-the-physics floor, not a margin over baseline,
so it hardens under scrutiny rather than flaking.

## Methodological discipline

Single-shot improvements over baselines are not believed without
multi-realization verification. Three times in this project that
discipline changed a conclusion:

- A network-prior result that beat per-voxel sLORETA by Rc 0.93 vs 0.83
  in a single shot reverted to a 34% win-rate across 32 realizations —
  a favorable noise draw, not a real win. (Q-D)
- A calibrated open-loop delivery result survived 5 seeds at
  84.5–85.2% (σ ~0.1pp) — a confirmed positive. (Q-C)
- A "model-free beats model-based under noise" result turned out to be
  a regularization-tuning artifact: the regularizer had been picked
  from a single-seed scan that stopped at the wrong end. An honest
  8-seed sweep moved the optimum and erased the reversal.

The third one is the subtlest: the multi-realization rule applies to
**hyperparameters as well as RNG seeds**, and applies regardless of
whether the result is flattering or humbling. Over-caution is as much
a bias as over-enthusiasm — the bad-tuning result slid through
precisely because it *looked* like a cautious negative ("fancy method
loses to crude heuristic"), the shape this work had pre-decided to
trust. A cherry-picked hyperparameter is a cherry-picked realization.
Don't tune and evaluate on the same draws.

## What's not in this repo

Two pieces of the broader system are intentionally excluded:

- **Device-faithful control law simulation.** Real-time control
  simulation for the device this work supports is held back for IP
  reasons. The `closed_loop.py` simulation included here is a
  source-space control experiment (Q-C Exp 3), a different question
  from device control.
- **Electrode-subset optimization.** A mixed-integer optimization layer
  for choosing which electrodes to activate from a larger montage. In
  the internal roadmap; not yet implemented in code.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r bandr_inverse/requirements.txt
```

On Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r bandr_inverse/requirements.txt
```

Sanity-check the full pipeline (~30s):
```powershell
$env:MPLBACKEND='Agg'
python .\bandr_inverse\notebooks\01_sphere_sanity.py
```

Each notebook is a standalone script — run any of them the same way;
figures land in `bandr_inverse/figures/`. The Q-B/Q-C scripts require
MNE-Python's `sample` dataset, which downloads automatically on first
run (~1.5 GB).

## Author Dr. Sarah Case

Built May 2026. Contact: `s.case.103@gmail.com`.

## License

MIT. See [`LICENSE`](LICENSE).
