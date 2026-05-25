"""
Active leadfield estimation via conductivity-parameter fitting.

This is the Q-C system identification module. The core claim BandR-next
makes is: stimulation hardware can be used to refine the forward model
during a session, not just to deliver therapy. Concretely, we measure
the response of the head to known calibration current patterns, fit
parametric corrections to the template forward model, and produce a
calibrated leadfield that's closer to the subject-specific truth.

Simulation simplification
-------------------------
For v1 of the simulation we model "calibration measurements" as direct
noisy observations of leadfield columns at a small set of internal
test points. This is not what real EIT-style calibration physically
measures — real measurements are scalp-to-scalp transfer impedances
(voltage at one electrode due to current injected at another) which
are bilinear in the conductivity field. The parameter estimation
structure is the same in both cases (fit conductivities to observed
data), and the simplification keeps the simulation tractable while
preserving the essential mathematical point.

A more physically accurate v2 would generate scalp-to-scalp impedance
measurements from the spherical model and fit conductivities to those.
That extension is straightforward but adds ~100 lines of physics code
without changing the conclusions. Flag in the paper limitations.

Parameter spaces
----------------
We support two parameterizations of the conductivity correction:

  - 'global_skull': single scalar scaling factor for skull conductivity.
    1 parameter. Captures the dominant inter-individual variation in
    EEG forward modeling.

  - 'compartmental': independent scaling factors for each shell.
    4 parameters (brain, CSF, skull, scalp). Captures more general
    conductivity uncertainty but harder to identify from limited data.

A spatially-varying parameterization (per-region skull conductivity)
would be next, but spherical-model geometry doesn't have natural
regional subdivisions — that's a Q-B-on-real-geometry extension.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace
from typing import Tuple, Literal, Callable
from scipy.optimize import minimize

from spherical_forward import SphericalHeadModel, forward_potential


@dataclass
class CalibrationData:
    """Container for the simulated calibration measurements."""
    cal_positions_mm: np.ndarray   # (K, 3) internal test points
    cal_responses: np.ndarray      # (K, n_elec) noisy leadfield columns
    noise_sigma: float             # std-dev of additive Gaussian noise
    true_model: SphericalHeadModel # for reference / oracle comparisons


def simulate_calibration_data(
    true_model: SphericalHeadModel,
    electrode_pos_mm: np.ndarray,
    n_measurements: int = 20,
    noise_snr_power: float = 100.0,   # higher SNR than dipole sim — calibration uses bigger currents
    seed: int | None = 42,
) -> CalibrationData:
    """
    Generate noisy leadfield-column measurements at random internal points.

    See module docstring for the simplification this entails.

    Parameters
    ----------
    true_model : the ground-truth model. Calibration will try to recover
        a leadfield close to L_true via fitting.
    electrode_pos_mm : (n_elec, 3) electrode positions.
    n_measurements : number of calibration measurements (= number of
        test points where we "evaluate" the true leadfield).
    noise_snr_power : ratio of signal power to noise power per
        measurement. Calibration is high-SNR by design (you control
        the input).
    seed : RNG seed for reproducibility.

    Returns
    -------
    CalibrationData
    """
    rng = np.random.default_rng(seed)
    R_brain = true_model.radii_mm[0]
    R_max = R_brain - 5.0

    # Pick random internal points (uniform in the ball)
    # Use rejection sampling for simplicity
    positions = []
    while len(positions) < n_measurements:
        candidate = rng.uniform(-R_max, R_max, size=3)
        if np.linalg.norm(candidate) <= R_max:
            positions.append(candidate)
    cal_positions = np.array(positions)

    # Evaluate true leadfield columns at these positions (radial orientation)
    cal_responses_clean = np.zeros((n_measurements, electrode_pos_mm.shape[0]))
    for i, pos in enumerate(cal_positions):
        r = np.linalg.norm(pos)
        moment = (pos / r) if r > 1e-9 else np.array([0.0, 0.0, 1.0])
        # Unit radial dipole
        V = forward_potential(pos, moment, electrode_pos_mm, model=true_model)
        cal_responses_clean[i] = V

    # Per-measurement noise scaled to a target SNR
    sig_power = np.mean(np.var(cal_responses_clean, axis=1))
    noise_var = sig_power / noise_snr_power
    noise_sigma = np.sqrt(noise_var)
    noise = rng.normal(0, noise_sigma, size=cal_responses_clean.shape)
    cal_responses = cal_responses_clean + noise

    return CalibrationData(
        cal_positions_mm=cal_positions,
        cal_responses=cal_responses,
        noise_sigma=noise_sigma,
        true_model=true_model,
    )


def _build_model_from_params(
    template: SphericalHeadModel,
    theta: np.ndarray,
    parameterization: str,
) -> SphericalHeadModel:
    """Map parameter vector → conductivity vector → new model."""
    sigmas_template = np.array(template.sigmas)
    if parameterization == 'global_skull':
        # theta = [skull_scaling]
        sigmas = sigmas_template.copy()
        sigmas[2] = sigmas_template[2] * theta[0]  # skull is index 2
    elif parameterization == 'compartmental':
        # theta = [scale_brain, scale_csf, scale_skull, scale_scalp]
        sigmas = sigmas_template * theta
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")
    return replace(template, sigmas=tuple(sigmas))


def _predicted_responses(
    cal_positions: np.ndarray,
    electrode_pos: np.ndarray,
    model: SphericalHeadModel,
) -> np.ndarray:
    """Compute predicted leadfield columns under a candidate model."""
    out = np.zeros((cal_positions.shape[0], electrode_pos.shape[0]))
    for i, pos in enumerate(cal_positions):
        r = np.linalg.norm(pos)
        moment = (pos / r) if r > 1e-9 else np.array([0.0, 0.0, 1.0])
        out[i] = forward_potential(pos, moment, electrode_pos, model=model)
    return out


def fit_conductivity(
    cal_data: CalibrationData,
    template: SphericalHeadModel,
    electrode_pos_mm: np.ndarray,
    parameterization: Literal['global_skull', 'compartmental'] = 'global_skull',
    verbose: bool = False,
) -> Tuple[np.ndarray, SphericalHeadModel, dict]:
    """
    Estimate conductivity-correction parameters from calibration data.

    Solves:
        theta_hat = argmin_theta || V_measured - V_predicted(theta) ||^2

    where V_predicted depends on theta through the parameterization.

    Returns
    -------
    theta_hat : the fitted parameter vector
    fitted_model : SphericalHeadModel with corrected conductivities
    info : dict with optimization diagnostics ('cost', 'n_iter', 'success')
    """
    if parameterization == 'global_skull':
        x0 = np.array([1.0])
        bounds = [(0.1, 10.0)]
    elif parameterization == 'compartmental':
        x0 = np.array([1.0, 1.0, 1.0, 1.0])
        bounds = [(0.3, 3.0), (0.3, 3.0), (0.1, 10.0), (0.3, 3.0)]
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    def cost(theta):
        model = _build_model_from_params(template, theta, parameterization)
        pred = _predicted_responses(
            cal_data.cal_positions_mm, electrode_pos_mm, model
        )
        residual = cal_data.cal_responses - pred
        return 0.5 * np.sum(residual ** 2)

    if verbose:
        print(f"  Fitting {parameterization} ({len(x0)} params)...")

    result = minimize(
        cost, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 100, 'gtol': 1e-8, 'disp': verbose},
    )

    fitted_model = _build_model_from_params(template, result.x, parameterization)
    info = {
        'cost': float(result.fun),
        'n_iter': int(result.nit),
        'success': bool(result.success),
        'message': str(result.message),
    }
    return result.x, fitted_model, info


def leadfield_error(
    L_estimate: np.ndarray,
    L_true: np.ndarray,
) -> float:
    """Relative Frobenius-norm error: ||L_hat - L_true||_F / ||L_true||_F."""
    return float(np.linalg.norm(L_estimate - L_true) / np.linalg.norm(L_true))
