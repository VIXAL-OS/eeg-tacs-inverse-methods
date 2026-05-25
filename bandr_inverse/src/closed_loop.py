"""
Closed-loop stimulation targeting with gray-box compensation for
forward-model error.

The targeting problem: given a desired electric-field pattern E_target
at some set of voxels, choose electrode currents I such that the actual
field L_true.T @ I matches E_target. The catch: we don't know L_true,
we only know an approximation L_template.

Two strategies compared:

  - Open-loop: compute I once using L_template's pseudoinverse. Apply.
    Whatever field results, that's what you get. Errors compound from
    forward-model mismatch.

  - Closed-loop: iterate. Apply I, measure resulting field (in
    simulation: read from L_true; in real life: estimate from EEG
    via source localization), compute error, update I. Converges to
    target even if L_template is wrong, as long as the error structure
    isn't pathologically biased.

Why this works mathematically
-----------------------------
The closed-loop update I_{k+1} = I_k + gamma * (L_template^T)^+ * (E_target - E_actual)
is exactly the gradient-descent step for the cost ||E_target - E_actual||^2
where the gradient is approximated using L_template instead of L_true.
This converges as long as L_template @ L_true.T is positive-definite
(approximately, ignoring regularization). For small perturbations of
the conductivity field this condition is easily satisfied — the
inner product structure of nearby leadfields is well-behaved.

Hard caveats:
- If L_template is *systematically biased* (e.g., consistent rotation
  of fields, anisotropy modeled isotropically), closed-loop may
  converge to a wrong fixed point.
- The "measurement" step requires either direct field readout
  (impossible in vivo) or source-localization-based estimation
  (introduces its own inverse-problem error). In real BandR this is
  the EEG read side, with the same L_template determining its
  accuracy. Bias propagates — see `closed_loop_target(..., observer_K=K)`
  for the honest version where read and write share the same wrong L.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class TargetingResult:
    """Outcome of one targeting protocol (open- or closed-loop)."""
    currents: np.ndarray              # final electrode currents (n_elec,)
    achieved_field: np.ndarray        # field actually produced (n_target_vox,)
    error_trajectory: np.ndarray      # what the controller sees per iter (drives stopping)
    true_error_trajectory: np.ndarray # ||E_target - L_true.T @ I_k|| per iter (ground truth)
    currents_history: np.ndarray      # (n_iter, n_elec) — electrode currents per iter;
                                       # lets the caller compute any per-iter metric externally
    converged: bool
    n_iterations: int


def _safety_project(I: np.ndarray, total_budget: float,
                     per_elec_max: float) -> np.ndarray:
    """
    Project currents onto safety constraint set:
      - sum(I) = 0 (Kirchhoff)
      - |I_k| <= per_elec_max
      - sum(|I|) <= total_budget (total injected current)

    Simple sequential projection — not jointly optimal but adequate.
    """
    # Kirchhoff: subtract mean
    I = I - np.mean(I)
    # Per-electrode bound
    I = np.clip(I, -per_elec_max, per_elec_max)
    # Total budget (L1 norm)
    total = np.sum(np.abs(I))
    if total > total_budget:
        I = I * (total_budget / total)
    # Re-enforce Kirchhoff after clipping (may slightly violate budget)
    I = I - np.mean(I)
    return I


def open_loop_target(
    L_template: np.ndarray,           # (n_elec, n_src) template leadfield
    E_target: np.ndarray,             # (n_src,) desired field at each source voxel
    target_mask: np.ndarray | None = None,  # (n_src,) bool, which voxels matter
    total_budget_A: float = 2e-3,     # 2 mA total injected current limit
    per_elec_max_A: float = 1e-3,     # 1 mA per electrode limit
    regularization: float = 1e-3,     # Tikhonov on the inverse
) -> np.ndarray:
    """
    Compute electrode currents for one-shot open-loop targeting.

    Solves the regularized least-squares problem:
        I = argmin_i || L_template^T i - E_target ||^2 + lambda^2 ||i||^2
    then projects onto the safety constraint set.

    Returns
    -------
    I : (n_elec,) electrode currents in amperes.
    """
    L = L_template
    n_elec = L.shape[0]
    if target_mask is None:
        target_mask = np.ones(L.shape[1], dtype=bool)

    A = L[:, target_mask]      # (n_elec, n_target)
    b = E_target[target_mask]  # (n_target,)

    # Regularized least squares: (A A^T + lam^2 I) I = A b
    # Note: A maps currents to fields via A^T, so we want A pseudoinverse
    AtA_plus_reg = A @ A.T + regularization ** 2 * np.eye(n_elec)
    I = np.linalg.solve(AtA_plus_reg, A @ b)
    I = _safety_project(I, total_budget_A, per_elec_max_A)
    return I


def closed_loop_target(
    L_template: np.ndarray,
    L_true: np.ndarray,               # only used for *simulating* the measurement
    E_target: np.ndarray,
    target_mask: np.ndarray | None = None,
    total_budget_A: float = 2e-3,
    per_elec_max_A: float = 1e-3,
    regularization: float = 1e-3,
    gain: float = 0.5,
    max_iter: int = 30,
    convergence_tol: float = 1e-6,
    measurement_noise: float = 0.0,
    observer_K: np.ndarray | None = None,
    observer_calibration: np.ndarray | None = None,
    controller_consistent: bool = False,
    verbose: bool = False,
) -> TargetingResult:
    """
    Iterative closed-loop targeting with template-based controller updates.

    Two observer modes (set via `observer_K`):

    A. Cheating observer (`observer_K is None`) — the historical default.
       Each iteration directly reads the true field:
           E_observed = L_true^T @ I + noise
       Useful as a best-case reference but corresponds to no real protocol:
       in vivo we have no oracle access to the brain field.

    B. Honest observer (`observer_K` provided) — what BandR actually does.
       The field is *inferred* from EEG measurements, source-localized
       through the same (wrong) template leadfield used by the controller:
           J_response = L_true^T @ I          # true field at sources (V at voxel)
           V_scalp    = L_true @ J_response + noise   # EEG signature
                                              # (linear neural-response proxy,
                                              # gain absorbed into units)
           J_hat      = observer_K @ V_scalp  # source-localized field estimate
           E_observed = J_hat[target_mask]    # (optionally calibrated, see below)

       Optional calibration via `observer_calibration` (per-target divisor):
       regularized inverses have a non-identity resolution matrix
       R_template = observer_K @ L_template. At a target voxel t,
       J_hat[t] ≈ R_template[t,t] * E_true[t] (when L_template = L_true).
       Without calibration, the controller sees a systematically attenuated
       field (R[t,t] << 1 for sLORETA on deep-ish targets) and drives the
       current up until it saturates the safety budget — divergence
       dominated by operator gain, not by the L mismatch we care about.
       Passing `observer_calibration = diag(R_template)[target_mask]`
       removes that gain so the residual deviation isolates the
       reciprocity-coupled bias from L_template != L_true.

       Because the same wrong L appears in both the EEG forward and the
       inverse, the read-error and the write-error are coupled. They may
       cancel (reciprocity gives a well-conditioned fixed point near the
       true target) or compound (smearing + bias drives the loop to a
       biased fixed point). That is the scientific question Exp 3 tests.

    Controller update (default):
           delta_I = gain * (L_template^T)^+ @ (E_target - E_observed)
           I_{k+1} = project(I_k + delta_I)
       This is the gradient of the cost ||L_template^T I - E_target||² —
       i.e. the controller's notion of "how does my current map to the
       brain field." In honest mode it does NOT match the gradient of
       the controller's actual *observed* cost (which routes through
       K @ L @ L^T, not L^T alone), so the loop can land at a fixed
       point that is far from a minimum of the observed cost.

    Gradient-consistent controller (`controller_consistent=True`, honest only):
           Define G = observer_K @ L_template @ L_template^T   (n_src, n_elec)
           delta_I = gain * (G_masked^T G_masked + λ²I)^{-1} G_masked^T err
       where G_masked = G[target_mask, :]. This is the regularized Gauss-
       Newton step for the controller's model of the observed cost
           ||E_target - G @ I||²
       so the loop actually descends the cost it can measure. When the
       observer is built on L_template and the world runs L_true, the
       fixed point is the minimum of the *template-model* observed cost
       evaluated at the *true-world* observation — i.e. the cleanest test
       of reciprocity-coupled bias one can set up with these ingredients.

    We also track `true_error_trajectory` = ||E_target - L_true^T @ I_k||
    at every iteration so the caller can compare what the controller
    *thinks* it achieved vs. what it actually achieved.

    Returns
    -------
    TargetingResult with both observed and true error trajectories.
    """
    if target_mask is None:
        target_mask = np.ones(E_target.shape[0], dtype=bool)

    n_elec = L_template.shape[0]
    A_template = L_template[:, target_mask]
    A_true = L_true[:, target_mask]
    b = E_target[target_mask]

    # Start with the open-loop solution
    AtA_plus_reg = A_template @ A_template.T + regularization ** 2 * np.eye(n_elec)
    I = np.linalg.solve(AtA_plus_reg, A_template @ b)
    I = _safety_project(I, total_budget_A, per_elec_max_A)

    # Precompute the gradient-consistent step matrices if requested.
    # G  = K_template @ L_template @ L_template^T   (n_src, n_elec)
    # When the controller minimizes ||E_target - G I||² over target_mask, the
    # regularized Gauss-Newton step is delta_I = (G_m^T G_m + λ²I)^{-1} G_m^T err.
    use_consistent = controller_consistent and (observer_K is not None)
    if use_consistent:
        G = observer_K @ (L_template @ L_template.T)         # (n_src, n_elec)
        G_masked = G[target_mask, :]                          # (n_target, n_elec)
        GtG_plus_reg = G_masked.T @ G_masked + regularization ** 2 * np.eye(n_elec)
    else:
        G_masked = None
        GtG_plus_reg = None

    rng = np.random.default_rng(0)
    error_trajectory = []
    true_error_trajectory = []
    currents_history = []
    prev_err_norm = np.inf
    converged = False
    iter_used = 0

    for k in range(max_iter):
        currents_history.append(I.copy())
        # The TRUE field at the target voxels — used only for diagnostics,
        # never fed back into the controller in honest mode.
        E_true_at_target = A_true.T @ I
        true_err_norm = float(np.linalg.norm(b - E_true_at_target))

        if observer_K is None:
            # Cheating mode: controller reads the true field directly
            E_observed = E_true_at_target.copy()
            if measurement_noise > 0:
                E_observed = E_observed + rng.normal(
                    0, measurement_noise, size=E_observed.shape
                )
        else:
            # Honest mode: simulate EEG measurement + source-localize through
            # the (wrong) template. Same L appears in observer and controller.
            J_response = L_true.T @ I              # (n_src,) true brain field
            V_scalp = L_true @ J_response          # (n_elec,) EEG scalp signature
            if measurement_noise > 0:
                V_scalp = V_scalp + rng.normal(
                    0, measurement_noise, size=V_scalp.shape
                )
            J_hat = observer_K @ V_scalp           # (n_src,) source-localized estimate
            E_observed = J_hat[target_mask]        # (n_target,) field estimate at target
            if observer_calibration is not None:
                # Undo the template-side resolution-matrix gain so the
                # observation is unbiased at L_template = L_true.
                E_observed = E_observed / observer_calibration

        err = b - E_observed
        err_norm = float(np.linalg.norm(err))
        error_trajectory.append(err_norm)
        true_error_trajectory.append(true_err_norm)

        if verbose:
            print(f"  iter {k:3d}: ||obs_err|| = {err_norm:.4e}, "
                  f"||true_err|| = {true_err_norm:.4e}, "
                  f"||I|| = {np.linalg.norm(I):.3e}")

        if abs(prev_err_norm - err_norm) < convergence_tol * max(prev_err_norm, 1e-30):
            converged = True
            iter_used = k + 1
            break

        # Gradient step using template. Default uses pinv(L_template.T)
        # (gradient of the cost ||L_template^T I - E_target||²). If
        # controller_consistent is on, use the gradient of the *observed*
        # cost ||E_target - K L L^T I||² so the iteration actually descends
        # what the loop can measure.
        if use_consistent:
            delta_I = np.linalg.solve(GtG_plus_reg, G_masked.T @ err)
        else:
            delta_I = np.linalg.solve(AtA_plus_reg, A_template @ err)
        I = I + gain * delta_I
        I = _safety_project(I, total_budget_A, per_elec_max_A)

        prev_err_norm = err_norm
        iter_used = k + 1

    E_actual_final = A_true.T @ I
    return TargetingResult(
        currents=I,
        achieved_field=E_actual_final,
        error_trajectory=np.array(error_trajectory),
        true_error_trajectory=np.array(true_error_trajectory),
        currents_history=np.array(currents_history),
        converged=converged,
        n_iterations=iter_used,
    )
