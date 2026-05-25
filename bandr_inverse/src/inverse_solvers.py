"""
Linear inverse solvers for EEG source localization.

Implements the standard distributed-source linear inverse methods:
  - MNE: minimum-norm estimate (Hämäläinen & Ilmoniemi 1994)
  - wMNE: weighted minimum-norm with depth compensation
  - sLORETA: standardized LORETA (Pascual-Marqui 2002)
  - eLORETA: exact LORETA (Pascual-Marqui 2007)

All solvers return an "inverse operator" K such that source estimate
J_hat = K @ V_scalp for any new measurement V_scalp. This is the standard
factoring: solve the regularization problem once given the leadfield, then
apply the resulting operator to live data.

Why solvers as priors
---------------------
Every method here is the optimum of a regularized least-squares problem
with a different prior:
  - MNE: prior = "sources have minimum L2 energy"
  - wMNE: prior = "sources have minimum L2 energy, depth-weighted"
  - sLORETA: prior = MNE + per-voxel resolution-matrix normalization
  - eLORETA: prior = "minimum localization error across the source space"

Choosing a solver IS choosing a prior on what sources look like. None of
them is "the right answer" in some absolute sense.
"""
from __future__ import annotations
import numpy as np
from typing import Literal


def mne_operator(
    leadfield: np.ndarray,
    noise_cov: np.ndarray | None = None,
    snr: float = 3.0,
    depth_weighting: float = 0.0,
) -> np.ndarray:
    """
    Compute MNE (or weighted MNE) inverse operator.

    Solves:  min_J  ||V - L J||^2_{C^{-1}} + lambda^2 ||W J||^2

    Closed-form:
        K = W^{-2} L^T (L W^{-2} L^T + lambda^2 C)^{-1}

    Parameters
    ----------
    leadfield : (n_elec, n_sources) array
        Gain matrix L mapping source currents to scalp voltages.
    noise_cov : (n_elec, n_elec) array or None
        Measurement noise covariance. Identity if None.
    snr : float
        Signal-to-noise ratio used to set regularization:
        lambda^2 = trace(L L^T) / (snr^2 * trace(C) * n_elec)
        This is the standard MNE convention.
    depth_weighting : float in [0, 1]
        If > 0, sources weighted by column-norm of L raised to this power
        (typical value 0.8). 0 disables depth weighting (plain MNE).

    Returns
    -------
    K : (n_sources, n_elec) array
        Inverse operator. J_hat = K @ V.
    """
    L = np.asarray(leadfield, dtype=float)
    n_elec, n_src = L.shape

    if noise_cov is None:
        C = np.eye(n_elec)
    else:
        C = np.asarray(noise_cov, dtype=float)
        if C.shape != (n_elec, n_elec):
            raise ValueError("noise_cov must be (n_elec, n_elec)")

    # Source weighting (depth compensation)
    if depth_weighting > 0:
        col_norms = np.linalg.norm(L, axis=0)
        # Avoid divide-by-zero on dead columns
        col_norms = np.where(col_norms > 0, col_norms, 1.0)
        w = col_norms ** depth_weighting
        # W is diagonal: W[i,i] = w[i]
        # We need W^{-2}, which is diag(1/w^2)
        Winv2_diag = 1.0 / (w ** 2)
    else:
        Winv2_diag = np.ones(n_src)

    # Regularization parameter
    LWinvLt = (L * Winv2_diag) @ L.T   # broadcasting: scales each column of L
    trace_LWL = np.trace(LWinvLt)
    trace_C = np.trace(C)
    lam_sq = trace_LWL / (snr**2 * trace_C)

    # Solve the linear system instead of inverting explicitly
    rhs_matrix = LWinvLt + lam_sq * C
    # K = diag(Winv2) @ L^T @ inv(rhs_matrix)
    inv_term = np.linalg.solve(rhs_matrix, L)  # (n_elec, n_src) -- careful
    # Actually we want inv(rhs_matrix) @ V applied later, so:
    # K_for_application = Winv2 * L^T @ inv(rhs_matrix)
    # Use solve on the transposed system to avoid forming the inverse:
    M = np.linalg.solve(rhs_matrix, np.eye(n_elec))
    K = (L.T @ M) * Winv2_diag[:, None]
    return K


def sloreta_operator(
    leadfield: np.ndarray,
    noise_cov: np.ndarray | None = None,
    snr: float = 3.0,
) -> np.ndarray:
    """
    Compute sLORETA inverse operator (Pascual-Marqui 2002).

    sLORETA = MNE estimate, then voxel-wise standardize by the variance
    expected under the resolution matrix. Result has zero localization
    bias for single dipoles in noise-free conditions.

    Returns the operator K such that J_sLORETA = K @ V.
    """
    K_mne = mne_operator(leadfield, noise_cov=noise_cov, snr=snr,
                          depth_weighting=0.0)
    L = np.asarray(leadfield, dtype=float)
    # Resolution matrix R = K @ L
    # sLORETA scales each row of K by 1/sqrt(diag(R)_ii)
    R = K_mne @ L
    diag_R = np.diag(R)
    # Guard against negative or zero (shouldn't happen in theory)
    diag_R_safe = np.where(diag_R > 1e-30, diag_R, 1e-30)
    scaling = 1.0 / np.sqrt(diag_R_safe)
    K_sloreta = K_mne * scaling[:, None]
    return K_sloreta


def eloreta_operator(
    leadfield: np.ndarray,
    noise_cov: np.ndarray | None = None,
    snr: float = 3.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute eLORETA inverse operator (Pascual-Marqui 2007).

    eLORETA finds a diagonal weighting W that makes the resolution matrix
    have unit diagonal entries (zero localization error under ideal noise
    conditions). Solved by fixed-point iteration on:

        W_ii = sqrt( L_i^T (L W^{-1} L^T + alpha C)^{-1} L_i )

    The regularization `alpha` is adaptively rescaled each iteration so
    that the regularization-to-data ratio stays at 1/snr^2:

        alpha = trace(L W^{-1} L^T) / (snr^2 * trace(C))

    This is the same convention MNE-Python uses. A fixed alpha set from
    trace(L L^T) (the MNE-MNE convention) is several orders of magnitude
    too small once W shrinks during iteration, which leaves eLORETA
    effectively unregularized and catastrophically noise-sensitive even
    though the iteration itself converges cleanly.

    Choosing snr
    ------------
    `snr` is treated as *amplitude* SNR (matching MNE-Python). For data
    with power-SNR of K, pass snr = sqrt(K) — e.g. power-SNR=10 means
    snr=3.16 here, not snr=10. eLORETA's resolution kernel is sharper
    than sLORETA's, which makes deep-source argmax especially vulnerable
    to noise amplification at low regularization. If deep sources fail
    to localize even though the iteration converges, lower `snr` until
    the regularization-to-data ratio is large enough to bound the
    operator norm; the depth_bias number should approach zero as you do.
    """
    L = np.asarray(leadfield, dtype=float)
    n_elec, n_src = L.shape

    if noise_cov is None:
        C = np.eye(n_elec)
    else:
        C = np.asarray(noise_cov, dtype=float)

    trace_C = np.trace(C)

    # Initialize weights to column norms
    w = np.linalg.norm(L, axis=0)
    w = np.where(w > 0, w, 1.0)

    converged_at: int | None = None
    final_rel_change = float('nan')
    alpha = float('nan')
    for iteration in range(max_iter):
        Winv = 1.0 / w
        LWL = (L * Winv) @ L.T
        alpha = np.trace(LWL) / (snr ** 2 * trace_C)
        M = np.linalg.inv(LWL + alpha * C)
        new_w_sq = np.einsum('ij,ji->i', L.T @ M, L)
        new_w_sq = np.maximum(new_w_sq, 1e-30)
        new_w = np.sqrt(new_w_sq)
        rel_change = np.linalg.norm(new_w - w) / max(np.linalg.norm(w), 1e-30)
        w = new_w
        final_rel_change = rel_change
        if verbose:
            print(f"    eLORETA iter {iteration:3d}: "
                  f"rel_change = {rel_change:.3e}, "
                  f"alpha = {alpha:.3e}")
        if rel_change < tol:
            converged_at = iteration
            break

    if verbose:
        if converged_at is not None:
            print(f"    eLORETA converged at iter {converged_at} "
                  f"(rel_change = {final_rel_change:.3e}, "
                  f"alpha = {alpha:.3e})")
        else:
            print(f"    eLORETA did NOT converge in {max_iter} iters "
                  f"(final rel_change = {final_rel_change:.3e}, "
                  f"alpha = {alpha:.3e})")

    # Final operator with alpha consistent with the converged W
    Winv = 1.0 / w
    LWL = (L * Winv) @ L.T
    alpha = np.trace(LWL) / (snr ** 2 * trace_C)
    M = np.linalg.inv(LWL + alpha * C)
    K = (L.T @ M) * Winv[:, None]
    return K


def solve(
    method: Literal['mne', 'wmne', 'sloreta', 'eloreta'],
    leadfield: np.ndarray,
    data: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Convenience function: compute inverse operator and apply to data.

    For repeated application to many data vectors, compute the operator
    once and reuse it instead of calling this function repeatedly.
    """
    if method == 'mne':
        K = mne_operator(leadfield, **kwargs)
    elif method == 'wmne':
        kwargs.setdefault('depth_weighting', 0.8)
        K = mne_operator(leadfield, **kwargs)
    elif method == 'sloreta':
        K = sloreta_operator(leadfield, **kwargs)
    elif method == 'eloreta':
        K = eloreta_operator(leadfield, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        return K @ data
    elif data.ndim == 2:
        return K @ data
    else:
        raise ValueError("data must be 1D (n_elec,) or 2D (n_elec, n_times)")
