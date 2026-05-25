"""
Q-C: System identification — three experiments.

The actual contribution of the white paper lives here. Q-A and Q-B
were validation infrastructure; Q-C is "does active stimulation
calibration actually improve the inverse problem?"

Experiment 1: Forward-model sensitivity
---------------------------------------
Build L_true with skull conductivity scaled by s ∈ {0.5, 0.7, 1.0,
1.3, 1.5, 2.0}, simulate scalp data from a known source, invert using
L_template (built with s=1.0 default). Plot localization error vs s
for each solver. Quantifies the "Sapien-Labs pitfall" empirically.

Experiment 2: Active leadfield estimation
-----------------------------------------
Fix s=0.7. Generate K calibration measurements from L_true. Fit
conductivity-correction parameters under two parameterizations:
global skull scaling (1 param) and full compartmental scaling (4
params). Plot leadfield-error reduction vs K, one line per
parameterization. This is the headline.

Experiment 3: Closed-loop targeting robustness
----------------------------------------------
For several conductivity perturbations, compare open-loop (one-shot
inverse using L_template) and closed-loop (iterative correction)
targeting of a specific brain region. Plot targeting error over
iterations.

Run
---
    python notebooks/03_system_id.py

Total runtime ~3-5 minutes on a laptop.
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
from inverse_solvers import (
    mne_operator, sloreta_operator, eloreta_operator
)
from metrics import summary
from calibration import (
    simulate_calibration_data, fit_conductivity, leadfield_error
)
from closed_loop import open_loop_target, closed_loop_target


def build_source_space(model: SphericalHeadModel,
                        spacing_mm: float = 12.0) -> np.ndarray:
    """Same as Q-A. Coarse grid for speed."""
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
    """Radial-orientation leadfield from analytic spherical forward."""
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
# Experiment 1
# ============================================================

def run_experiment_1(electrodes: np.ndarray,
                      sources: np.ndarray,
                      L_template: np.ndarray,
                      template_model: SphericalHeadModel) -> dict:
    """Forward-model sensitivity to skull conductivity scaling."""
    print("\n" + "=" * 60)
    print("Experiment 1: forward-model sensitivity")
    print("=" * 60)

    scales = np.array([0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0])
    methods = ['MNE', 'wMNE', 'sLORETA', 'eLORETA']
    operators = {
        'MNE': mne_operator(L_template, snr=3.0),
        'wMNE': mne_operator(L_template, snr=3.0, depth_weighting=0.8),
        'sLORETA': sloreta_operator(L_template, snr=3.0),
        'eLORETA': eloreta_operator(L_template, snr=3.0, max_iter=50),
    }

    # Two test sources: shallow (where forward-model error is small) and
    # deep (where it's pronounced). Deep is the diagnostic case — that's
    # where conductivity uncertainty really hurts.
    test_positions = {
        'shallow (r=50mm)': np.array([50.0, 0.0, 0.0]),
        'deep (r=20mm)': np.array([20.0, 0.0, 0.0]),
    }
    test_pos = test_positions['deep (r=20mm)']  # primary plot uses deep
    print(f"  Test source: deep, at {test_pos}, depth = {np.linalg.norm(test_pos)}mm")

    # Use a SINGLE noise realization across all skull scales — we want
    # to isolate the effect of conductivity perturbation, not stack it
    # with stochastic noise variation per condition.
    rng = np.random.default_rng(42)
    moment_base = (test_pos / np.linalg.norm(test_pos)) * 1e-8
    V_base = forward_potential(test_pos, moment_base, electrodes, model=template_model)
    sig_var = np.var(V_base)
    noise_vec = rng.normal(0, np.sqrt(sig_var / 10.0), size=V_base.shape)

    loc_errors = {m: [] for m in methods}

    for s in scales:
        true_model = scaled_skull_model(template_model, s)
        moment = (test_pos / np.linalg.norm(test_pos)) * 1e-8
        V_clean = forward_potential(test_pos, moment, electrodes, model=true_model)
        V = V_clean + noise_vec  # SAME noise realization across scales

        for m in methods:
            J_hat = operators[m] @ V
            metrics_d = summary(J_hat, sources, test_pos)
            loc_errors[m].append(metrics_d['localization_error_mm'])

        line = f"  skull scale = {s:.2f}: "
        line += ", ".join(f"{m}={loc_errors[m][-1]:5.1f}mm" for m in methods)
        print(line)

    return {'scales': scales, 'loc_errors': loc_errors, 'methods': methods}


# ============================================================
# Experiment 2
# ============================================================

def run_experiment_2(electrodes: np.ndarray,
                      sources: np.ndarray,
                      L_template: np.ndarray,
                      template_model: SphericalHeadModel) -> dict:
    """Active leadfield estimation. The headline experiment."""
    print("\n" + "=" * 60)
    print("Experiment 2: active leadfield estimation")
    print("=" * 60)

    # Ground truth: skull conductivity 0.7× nominal (within inter-subject range)
    true_scale = 0.7
    true_model = scaled_skull_model(template_model, true_scale)
    L_true = build_leadfield(sources, electrodes, true_model)
    baseline_error = leadfield_error(L_template, L_true)
    print(f"\n  True skull scale: {true_scale}")
    print(f"  Baseline ||L_template - L_true||/||L_true|| = {baseline_error:.4f}")

    K_values = [5, 10, 20, 40, 80]
    param_modes = ['global_skull', 'compartmental']
    results = {p: [] for p in param_modes}
    recovered_scales = {p: [] for p in param_modes}

    for K in K_values:
        print(f"\n  K = {K} calibration measurements:")
        cal_data = simulate_calibration_data(
            true_model=true_model,
            electrode_pos_mm=electrodes,
            n_measurements=K,
            noise_snr_power=100.0,
            seed=42,
        )
        for pmode in param_modes:
            theta_hat, fitted_model, info = fit_conductivity(
                cal_data, template_model, electrodes, parameterization=pmode
            )
            L_hat = build_leadfield(sources, electrodes, fitted_model)
            err = leadfield_error(L_hat, L_true)
            err_reduction = err / baseline_error
            results[pmode].append(err_reduction)
            if pmode == 'global_skull':
                recovered_scales[pmode].append(float(theta_hat[0]))
                print(f"    {pmode:14s}: skull_scale_hat = {theta_hat[0]:.3f} "
                      f"(true={true_scale}); rel_err = {err_reduction:.3f}")
            else:
                print(f"    {pmode:14s}: theta_hat = {np.round(theta_hat, 3).tolist()}; "
                      f"rel_err = {err_reduction:.3f}")

    return {
        'K_values': K_values,
        'param_modes': param_modes,
        'results': results,
        'baseline_error': baseline_error,
        'true_scale': true_scale,
        'recovered_scales': recovered_scales,
    }


# ============================================================
# Experiment 3
# ============================================================

def run_experiment_3(electrodes: np.ndarray,
                      sources: np.ndarray,
                      L_template: np.ndarray,
                      template_model: SphericalHeadModel) -> dict:
    """Closed-loop targeting robustness — cheating vs honest feedback.

    Setup: pick a shallow target voxel. Choose E_target as a magnitude
    we know is achievable with the safety budget. Then compare THREE
    targeting protocols, evaluated against the *true* field at target:

      - open-loop:       I = pinv(L_template^T) @ E_target, applied via L_true
      - closed (cheat):  iterate, observe through L_true directly (oracle)
      - closed (honest): iterate, observe through L_template-based inverse
                         applied to simulated EEG: V = L_true @ (L_true^T @ I).
                         Same wrong L appears in both observer and controller.

    The scientific question: does reciprocity make the read-error and
    write-error cancel (honest ≈ cheat ≈ converges) or compound (honest
    diverges / converges to a biased fixed point)?

    We track both the controller's observed error trajectory (what it
    uses to update) AND the true error trajectory (||E_target - L_true^T@I_k||)
    so we can tell when the controller *thinks* it converged vs. when it
    *actually* hit the target.
    """
    print("\n" + "=" * 60)
    print("Experiment 3: closed-loop targeting robustness")
    print("=" * 60)

    target_pos = np.array([50.0, 0.0, 30.0])
    target_idx = int(np.argmin(np.linalg.norm(sources - target_pos, axis=1)))
    target_pos_actual = sources[target_idx]
    print(f"  Target voxel: {target_pos_actual} mm (idx {target_idx})")

    # Calibrate target magnitude to what's actually achievable under
    # L_template with safety constraints. Procedure: solve unconstrained
    # for E_target = 1.0, see what current that demands, scale to fit
    # within budget. The resulting feasible target is below.
    L_target_col = L_template[:, target_idx]
    # Unconstrained: I_unc = L * 1.0 / (||L||^2 + reg). Achieved field = ||L||^2 / (||L||^2 + reg)
    norm_sq = float(np.dot(L_target_col, L_target_col))
    I_unc = L_target_col / (norm_sq + 1e-3 ** 2)
    # Now scale so total |I| fits in 2mA budget with margin
    budget = 2e-3
    headroom = 0.5  # use 50% of budget so closed-loop has room to adjust
    scale = headroom * budget / np.sum(np.abs(I_unc))
    feasible_target = norm_sq * scale / (norm_sq + 1e-3 ** 2)
    print(f"  Calibrated feasible target field: {feasible_target:.4e} V/m")
    print(f"  (this is what L_template alone, with safety margin, can deliver)")

    # Delta target — used by open / cheat / honest_raw / honest_cal
    E_target = np.zeros(sources.shape[0])
    E_target[target_idx] = feasible_target

    target_mask = np.zeros(sources.shape[0], dtype=bool)
    target_mask[target_idx] = True

    # Inverse operator for the honest observer. Built ONCE from L_template
    # so it is the "wrong" model in exactly the same way the controller's
    # gradient is. sLORETA is a natural choice (unbiased single-dipole
    # resolution), and is cheap enough to run alongside Exps 1-2.
    K_observer = sloreta_operator(L_template, snr=3.0)

    # Per-target diagonal of the template resolution matrix R = K @ L_template.
    # This is the gain the observer would apply to a delta-source at target
    # under the template model. We need it to undo the smoothing for the
    # *calibrated* honest variant (otherwise the loop's divergence is dominated
    # by R[t,t] << 1 even when L_template = L_true).
    R_template = K_observer @ L_template
    cal_factors = np.diag(R_template)[target_mask]
    print(f"  Resolution-matrix diagonal at target voxel: "
          f"R_template[t,t] = {cal_factors[0]:.4f}")
    print(f"  (raw observer underestimates field by ~{1.0/cal_factors[0]:.1f}x; "
          f"calibration divides J_hat[t] by this to undo the operator gain)")

    # Kernel-matched target — used by honest_smooth.
    #
    # The honest_raw/cal variants ask the controller to drive a *point* read
    # (J_hat at one voxel) to a *point* target. That's hopeless when the
    # inverse operator's resolution kernel doesn't peak at the target voxel.
    # Here we instead ask it to drive the full source-space J_hat to the
    # shape it WOULD have if a delta source of magnitude `feasible_target`
    # sat at the target voxel and was viewed through the template:
    #   E_target_smooth = R_template[:, target_idx] * feasible_target
    # Then at L_template = L_true the fixed point is reached when the
    # true field looks like delta_target * feasible_target — exactly the
    # same physical goal as the delta-target experiment, but expressed in
    # observer-coordinates so the controller can actually measure progress.
    kernel = R_template[:, target_idx]
    E_target_smooth = kernel * feasible_target
    target_mask_smooth = np.ones(sources.shape[0], dtype=bool)
    print(f"  Kernel-matched target: ||E_target_smooth|| = {np.linalg.norm(E_target_smooth):.4e}, "
          f"peak away from target_idx = {np.linalg.norm(sources[int(np.argmax(np.abs(kernel)))] - sources[target_idx]):.1f}mm")

    scales = [0.7, 1.0, 1.3]
    trajectories = {}

    for s in scales:
        true_model = scaled_skull_model(template_model, s)
        L_true = build_leadfield(sources, electrodes, true_model)

        # Open-loop: use template, accept the resulting error
        I_open = open_loop_target(L_template, E_target,
                                    target_mask=target_mask,
                                    total_budget_A=2e-3,
                                    per_elec_max_A=1e-3,
                                    regularization=1e-3)
        E_actual_open = L_true.T @ I_open
        open_err = abs(E_target[target_idx] - E_actual_open[target_idx])

        # Open-loop through an actively-calibrated leadfield. Active stim
        # calibration (Exp 2) at K=80, global_skull recovers L to ~2% error.
        # Test: does open-loop through L_hat deliver close to feasible at
        # the target, with cross-skull-scale variance collapsed because
        # L_hat now tracks L_true regardless of s?
        cal_data_s = simulate_calibration_data(
            true_model=true_model,
            electrode_pos_mm=electrodes,
            n_measurements=80,
            noise_snr_power=100.0,
            seed=42,
        )
        theta_hat_s, fitted_model_s, _ = fit_conductivity(
            cal_data_s, template_model, electrodes,
            parameterization='global_skull',
        )
        L_hat = build_leadfield(sources, electrodes, fitted_model_s)
        I_open_cal = open_loop_target(L_hat, E_target,
                                       target_mask=target_mask,
                                       total_budget_A=2e-3,
                                       per_elec_max_A=1e-3,
                                       regularization=1e-3)
        E_actual_open_cal = L_true.T @ I_open_cal
        open_cal_err = abs(E_target[target_idx] - E_actual_open_cal[target_idx])

        # Closed-loop (cheat): observer reads true field directly
        cheat = closed_loop_target(
            L_template=L_template,
            L_true=L_true,
            E_target=E_target,
            target_mask=target_mask,
            total_budget_A=2e-3,
            per_elec_max_A=1e-3,
            regularization=1e-3,
            gain=0.7,
            max_iter=30,
            convergence_tol=1e-7,
            observer_K=None,
        )

        # Closed-loop (honest, raw): EEG → K_template @ V_scalp, no calibration.
        # Dominated by R_template[t,t] underestimating the field; shows the
        # practical hazard of naive source-localization-based feedback.
        honest_raw = closed_loop_target(
            L_template=L_template,
            L_true=L_true,
            E_target=E_target,
            target_mask=target_mask,
            total_budget_A=2e-3,
            per_elec_max_A=1e-3,
            regularization=1e-3,
            gain=0.7,
            max_iter=30,
            convergence_tol=1e-7,
            observer_K=K_observer,
            observer_calibration=None,
        )

        # Closed-loop (honest, calibrated): J_hat[t] / R_template[t,t].
        # Removes the operator-gain bias so residual deviation = the
        # reciprocity-coupled L_template/L_true mismatch effect.
        honest_cal = closed_loop_target(
            L_template=L_template,
            L_true=L_true,
            E_target=E_target,
            target_mask=target_mask,
            total_budget_A=2e-3,
            per_elec_max_A=1e-3,
            regularization=1e-3,
            gain=0.7,
            max_iter=30,
            convergence_tol=1e-7,
            observer_K=K_observer,
            observer_calibration=cal_factors,
        )

        # Closed-loop (honest, kernel-matched target): drive J_hat → R[:, t] * fs.
        # Observation and target both live in the resolution-matrix's column
        # space, but the controller still uses pinv(L_template^T) for the
        # update — gradient direction is inconsistent with the observed cost,
        # so the iteration lands at a wrong (but skull-scale-invariant) fixed
        # point.
        honest_smooth = closed_loop_target(
            L_template=L_template,
            L_true=L_true,
            E_target=E_target_smooth,
            target_mask=target_mask_smooth,
            total_budget_A=2e-3,
            per_elec_max_A=1e-3,
            regularization=1e-3,
            gain=0.7,
            max_iter=30,
            convergence_tol=1e-7,
            observer_K=K_observer,
            observer_calibration=None,
        )

        # Closed-loop (honest, kernel-matched target, gradient-consistent
        # controller): same observation as honest_smooth, but the update uses
        # the regularized Gauss-Newton step for the controller's *observed*
        # cost ||E_target_smooth - G I||² where G = K_template @ L L^T.
        # The iteration now actually descends the observed cost; the only
        # remaining mismatch is L_template ≠ L_true in G itself — i.e. the
        # reciprocity-coupling effect, isolated.
        honest_consistent = closed_loop_target(
            L_template=L_template,
            L_true=L_true,
            E_target=E_target_smooth,
            target_mask=target_mask_smooth,
            total_budget_A=2e-3,
            per_elec_max_A=1e-3,
            regularization=1e-3,
            gain=0.7,
            max_iter=30,
            convergence_tol=1e-7,
            observer_K=K_observer,
            observer_calibration=None,
            controller_consistent=True,
        )

        # Compute uniform metric: |achieved_true_field_at_target - feasible_target|
        # per iteration, for each closed-loop variant. This lets us compare all
        # protocols on the same physical scale regardless of how their controller
        # framed the optimization.
        def at_target_traj(history: np.ndarray) -> np.ndarray:
            achieved = history @ L_true[:, target_idx]
            return np.abs(achieved - feasible_target)
        cheat_at_target_traj = at_target_traj(cheat.currents_history)
        honest_raw_at_target_traj = at_target_traj(honest_raw.currents_history)
        honest_cal_at_target_traj = at_target_traj(honest_cal.currents_history)
        honest_smooth_at_target_traj = at_target_traj(honest_smooth.currents_history)
        honest_consistent_at_target_traj = at_target_traj(honest_consistent.currents_history)

        # Signed delivery (achieved / feasible) at the target voxel. The
        # absolute-error metric drops the sign — but for open-loop vs.
        # calibrated-open-loop the sign tells us whether the protocol
        # under-delivers (regularization floor) or overshoots (skull
        # under-estimate inflating the current).
        open_signed = float(E_actual_open[target_idx] / feasible_target)
        cheat_signed = float((L_true[:, target_idx] @ cheat.currents) / feasible_target)
        open_cal_signed = float(E_actual_open_cal[target_idx] / feasible_target)

        trajectories[s] = {
            'open_error': open_err,
            'open_calibrated_error': open_cal_err,
            'open_signed_delivery': open_signed,
            'cheat_signed_delivery': cheat_signed,
            'open_calibrated_signed_delivery': open_cal_signed,
            'theta_hat_calibrated': float(theta_hat_s[0]),
            'cheat_at_target': cheat_at_target_traj,
            'cheat_final_at_target': float(cheat_at_target_traj[-1]),
            'cheat_n_iter': cheat.n_iterations,
            'cheat_converged': cheat.converged,
            'honest_raw_at_target': honest_raw_at_target_traj,
            'honest_raw_final_at_target': float(honest_raw_at_target_traj[-1]),
            'honest_raw_n_iter': honest_raw.n_iterations,
            'honest_raw_converged': honest_raw.converged,
            'honest_cal_at_target': honest_cal_at_target_traj,
            'honest_cal_final_at_target': float(honest_cal_at_target_traj[-1]),
            'honest_cal_n_iter': honest_cal.n_iterations,
            'honest_cal_converged': honest_cal.converged,
            'honest_smooth_at_target': honest_smooth_at_target_traj,
            'honest_smooth_final_at_target': float(honest_smooth_at_target_traj[-1]),
            'honest_smooth_n_iter': honest_smooth.n_iterations,
            'honest_smooth_converged': honest_smooth.converged,
            'honest_consistent_at_target': honest_consistent_at_target_traj,
            'honest_consistent_final_at_target': float(honest_consistent_at_target_traj[-1]),
            'honest_consistent_n_iter': honest_consistent.n_iterations,
            'honest_consistent_converged': honest_consistent.converged,
            'feasible_target': feasible_target,
        }
        # Report in % of target — uniformly using at-target true-field error
        ft = feasible_target
        print(f"  skull_scale={s}:")
        print(f"    open           true_err@target = {open_err:.3e} "
              f"({100*open_err/ft:6.1f}% of target)")
        print(f"    open cal       true_err@target = {open_cal_err:.3e} "
              f"({100*open_cal_err/ft:6.1f}% of target) "
              f"[theta_hat={theta_hat_s[0]:.3f}, true={s}]")
        print(f"    cheat          true_err@target = {cheat_at_target_traj[-1]:.3e} "
              f"({100*cheat_at_target_traj[-1]/ft:6.1f}% of target) "
              f"after {cheat.n_iterations} iters")
        print(f"    honest raw     true_err@target = {honest_raw_at_target_traj[-1]:.3e} "
              f"({100*honest_raw_at_target_traj[-1]/ft:6.1f}% of target) "
              f"after {honest_raw.n_iterations} iters")
        print(f"    honest cal     true_err@target = {honest_cal_at_target_traj[-1]:.3e} "
              f"({100*honest_cal_at_target_traj[-1]/ft:6.1f}% of target) "
              f"after {honest_cal.n_iterations} iters")
        print(f"    honest smooth  true_err@target = {honest_smooth_at_target_traj[-1]:.3e} "
              f"({100*honest_smooth_at_target_traj[-1]/ft:6.1f}% of target) "
              f"after {honest_smooth.n_iterations} iters")
        print(f"    honest cons.   true_err@target = {honest_consistent_at_target_traj[-1]:.3e} "
              f"({100*honest_consistent_at_target_traj[-1]/ft:6.1f}% of target) "
              f"after {honest_consistent.n_iterations} iters")
        print(f"    --- signed delivery (achieved / feasible) ---")
        print(f"      open           = {100*open_signed:7.1f}%")
        print(f"      open cal       = {100*open_cal_signed:7.1f}%")
        print(f"      cheat (final)  = {100*cheat_signed:7.1f}%")

    # Cross-scale spread summary for the open vs open-calibrated comparison.
    sd = [trajectories[s]['open_signed_delivery'] for s in scales]
    sdc = [trajectories[s]['open_calibrated_signed_delivery'] for s in scales]
    print(f"\n  cross-scale signed-delivery spread:")
    print(f"    open       : min={min(sd):.3f}, max={max(sd):.3f}, range={max(sd)-min(sd):.3f}")
    print(f"    open cal   : min={min(sdc):.3f}, max={max(sdc):.3f}, range={max(sdc)-min(sdc):.3f}")

    # Monte Carlo over the calibration draw. True skull scales held fixed
    # at {0.7, 1.0, 1.3}; only the calibration RNG (positions + noise)
    # varies across seeds. Isolates "robust to the calibration draw"
    # from "robust across conductivities" — they're separate claims and
    # shouldn't be entangled.
    mc_seeds = [42, 43, 44, 45, 46]
    mc_scales = scales
    print(f"\n  Monte Carlo: {len(mc_seeds)} seeds × {len(mc_scales)} scales "
          f"(calibration draw varies; skull scales fixed)")
    # Cache L_true per scale — independent of calibration seed.
    L_true_cache = {
        s: build_leadfield(sources, electrodes, scaled_skull_model(template_model, s))
        for s in mc_scales
    }
    mc_signed = {s: [] for s in mc_scales}
    mc_theta = {s: [] for s in mc_scales}
    for seed in mc_seeds:
        for s in mc_scales:
            true_model_mc = scaled_skull_model(template_model, s)
            cal_data_mc = simulate_calibration_data(
                true_model=true_model_mc,
                electrode_pos_mm=electrodes,
                n_measurements=80,
                noise_snr_power=100.0,
                seed=seed,
            )
            theta_mc, fitted_mc, _ = fit_conductivity(
                cal_data_mc, template_model, electrodes,
                parameterization='global_skull',
            )
            L_hat_mc = build_leadfield(sources, electrodes, fitted_mc)
            I_mc = open_loop_target(L_hat_mc, E_target,
                                     target_mask=target_mask,
                                     total_budget_A=2e-3,
                                     per_elec_max_A=1e-3,
                                     regularization=1e-3)
            E_actual_mc = L_true_cache[s].T @ I_mc
            mc_signed[s].append(float(E_actual_mc[target_idx] / feasible_target))
            mc_theta[s].append(float(theta_mc[0]))

    print(f"\n  Per-scale statistics across seeds:")
    print(f"    {'scale':>6s}  {'theta_hat':>18s}  {'signed delivery':>24s}  {'min..max':>14s}")
    for s in mc_scales:
        sdv = np.array(mc_signed[s])
        thv = np.array(mc_theta[s])
        print(f"    {s:>6.2f}  {thv.mean():>8.4f} ± {thv.std():>7.4f}  "
              f"{100*sdv.mean():>10.2f}% ± {100*sdv.std():>5.2f}%  "
              f"{100*sdv.min():>5.2f}..{100*sdv.max():<5.2f}%")

    print(f"\n  Per-seed cross-scale range (the 'collapse' claim, per realization):")
    per_seed_ranges = []
    for i, seed in enumerate(mc_seeds):
        deliveries = [mc_signed[s][i] for s in mc_scales]
        rng_seed = max(deliveries) - min(deliveries)
        per_seed_ranges.append(rng_seed)
        delivs_str = ", ".join(f"{100*d:.2f}%" for d in deliveries)
        print(f"    seed={seed}: signed=[{delivs_str}], range={rng_seed:.4f}")
    print(f"  Worst-case per-seed range: {max(per_seed_ranges):.4f}  "
          f"(vs. uncalibrated open-loop range across same scales: "
          f"{max(sd)-min(sd):.4f})")

    return {'scales': scales, 'trajectories': trajectories,
            'target_pos': target_pos_actual, 'target_idx': target_idx,
            'feasible_target': feasible_target,
            'monte_carlo': {
                'seeds': mc_seeds, 'scales': mc_scales,
                'signed': mc_signed, 'theta': mc_theta,
                'per_seed_ranges': per_seed_ranges,
            }}


# ============================================================
# Plotting
# ============================================================

def plot_results(res1: dict, res2: dict, res3: dict, out_dir: str):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # Panel (0,0): Experiment 1
    ax = axes[0, 0]
    for m in res1['methods']:
        ax.plot(res1['scales'], res1['loc_errors'][m], 'o-', label=m, linewidth=1.5)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Skull conductivity scaling (×nominal)')
    ax.set_ylabel('Localization error (mm)')
    ax.set_title('Exp 1: Forward-model sensitivity\n(deep source r=20mm, SNR=10)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (0,1): Experiment 2
    ax = axes[0, 1]
    for pmode in res2['param_modes']:
        label = pmode.replace('_', ' ') + f" ({1 if pmode=='global_skull' else 4} params)"
        ax.plot(res2['K_values'], res2['results'][pmode], 'o-',
                label=label, linewidth=1.5)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5,
                label='baseline (no calibration)')
    ax.set_xlabel('Number of calibration measurements (K)')
    ax.set_ylabel('||L_hat - L_true|| / ||L_template - L_true||')
    ax.set_title(f'Exp 2: Active leadfield estimation\n(true skull scale = {res2["true_scale"]})')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Panel (0,2): Exp 3 summary — final at-target true error per protocol/scale
    ax = axes[0, 2]
    scales = res3['scales']
    feasible = res3['feasible_target']
    protocols = [
        ('open',           'open_error',                        'C7'),
        ('open cal',       'open_calibrated_error',             'C9'),
        ('cheat',          'cheat_final_at_target',             'C2'),
        ('honest raw',     'honest_raw_final_at_target',        'C3'),
        ('honest cal',     'honest_cal_final_at_target',        'C1'),
        ('honest smooth',  'honest_smooth_final_at_target',     'C0'),
        ('honest consist', 'honest_consistent_final_at_target', 'C4'),
    ]
    width = 0.12
    x_pos = np.arange(len(scales))
    for j, (lbl, key, c) in enumerate(protocols):
        vals = [100 * res3['trajectories'][s][key] / feasible for s in scales]
        ax.bar(x_pos + (j - 3) * width, vals, width, label=lbl, color=c, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'skull={s}' for s in scales])
    ax.set_ylabel('Final |E_true[target] − E_target| (% of feasible)')
    ax.set_title('Exp 3 summary: final at-target true-field error\nby protocol × skull scale (symlog)')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('symlog', linthresh=1.0)

    # Panel (1,0): Exp 3a — cheating closed-loop vs open
    colors = {0.7: 'C0', 1.0: 'C1', 1.3: 'C2'}
    ax = axes[1, 0]
    for s in res3['scales']:
        traj = res3['trajectories'][s]
        ax.plot(100 * traj['cheat_at_target'] / feasible, '-',
                color=colors[s], label=f'cheat, skull={s}', linewidth=1.6)
        ax.axhline(100 * traj['open_error'] / feasible, linestyle='--',
                    color=colors[s], alpha=0.6,
                    label=f'open, skull={s}')
    ax.set_xlabel('Closed-loop iteration')
    ax.set_ylabel('|E_true[target] − E_target| (% of feasible)')
    ax.set_title('Exp 3a: Cheating loop (oracle field readout)\nvs open-loop dashed — sanity-check best case')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (1,1): Exp 3b — honest naive (raw + cal) vs open. Both broken.
    ax = axes[1, 1]
    for s in res3['scales']:
        traj = res3['trajectories'][s]
        ax.plot(100 * traj['honest_raw_at_target'] / feasible, '-',
                color=colors[s], label=f'honest raw, skull={s}', linewidth=1.4)
        ax.plot(100 * traj['honest_cal_at_target'] / feasible, ':',
                color=colors[s], label=f'honest cal, skull={s}', linewidth=2.0)
        ax.axhline(100 * traj['open_error'] / feasible, linestyle='--',
                    color=colors[s], alpha=0.6,
                    label=f'open, skull={s}')
    ax.set_xlabel('Closed-loop iteration')
    ax.set_ylabel('|E_true[target] − E_target| (% of feasible)')
    ax.set_title('Exp 3b: Naive point-target honest loop (raw + cal)\nResolution-kernel shift breaks the read at every scale')
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel (1,2): Exp 3c — kernel-matched target with smooth vs gradient-consistent
    # controllers, against cheat and open. The smooth controller has a gradient
    # mismatch and lands at a biased fixed point. The consistent controller uses
    # the right gradient for its observed cost — and now also delivers ≈ target.
    ax = axes[1, 2]
    for s in res3['scales']:
        traj = res3['trajectories'][s]
        ax.plot(100 * traj['cheat_at_target'] / feasible, '-',
                color=colors[s], alpha=0.35,
                label=f'cheat, skull={s}', linewidth=1.0)
        ax.plot(100 * traj['honest_smooth_at_target'] / feasible, ':',
                color=colors[s], alpha=0.85,
                label=f'honest smooth, skull={s}', linewidth=1.6)
        ax.plot(100 * traj['honest_consistent_at_target'] / feasible, '-',
                color=colors[s], label=f'honest consist, skull={s}', linewidth=2.2)
        ax.axhline(100 * traj['open_error'] / feasible, linestyle='--',
                    color=colors[s], alpha=0.5,
                    label=f'open, skull={s}')
    ax.set_xlabel('Closed-loop iteration')
    ax.set_ylabel('|E_true[target] − E_target| (% of feasible)')
    ax.set_title('Exp 3c: Honest, kernel-matched target.\nDotted = pinv(L^T) gradient (biased). Solid = G^T G gradient (consistent).')
    ax.legend(loc='best', fontsize=6, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'q_c_system_id.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"\nFigure saved to: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Q-C: System identification — active calibration + closed-loop")
    print("=" * 60)

    template_model = SphericalHeadModel()
    electrodes = make_standard_montage_64(template_model)
    print(f"\nTemplate model: 4-shell sphere")
    print(f"  sigmas (S/m) = {template_model.sigmas}")
    print(f"  electrodes = {electrodes.shape[0]}")

    print("\nBuilding source space (12mm spacing for Q-C speed)...")
    sources = build_source_space(template_model, spacing_mm=12.0)
    print(f"  {sources.shape[0]} candidate sources")

    print("\nBuilding template leadfield...")
    L_template = build_leadfield(sources, electrodes, template_model)
    print(f"  L_template shape: {L_template.shape}, cond = {np.linalg.cond(L_template):.2e}")

    res1 = run_experiment_1(electrodes, sources, L_template, template_model)
    res2 = run_experiment_2(electrodes, sources, L_template, template_model)
    res3 = run_experiment_3(electrodes, sources, L_template, template_model)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    plot_results(res1, res2, res3, out_dir)

    print("\n" + "=" * 60)
    print("Q-C complete.")
    print("=" * 60)
    print("Three claims tested:")
    print("  Exp 1: forward-model error degrades localization measurably")
    print("  Exp 2: active calibration reduces leadfield error")
    print("  Exp 3: closed-loop control absorbs residual error")
    print("\nIf Exp 2 shows curves dropping below the baseline=1 dashed line,")
    print("calibration is helping. If Exp 3 shows closed-loop converging below")
    print("open-loop horizontal lines, feedback is helping. Together: the")
    print("Q-C contribution argument.")


if __name__ == '__main__':
    main()
