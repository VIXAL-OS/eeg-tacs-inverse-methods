"""
Metrics for evaluating inverse-method performance.

The three workhorse quantities every source-localization paper reports:
  - Localization error: distance from estimated peak to true source
  - Point spread function (PSF): how broad the reconstruction is for a
    point source. Reported as FWHM, half-energy radius, or full energy
    distribution as a function of distance from true source.
  - Depth bias: do estimates systematically pull toward (or away from)
    the surface?

These are the numbers that go into the paper's tables.
"""
from __future__ import annotations
import numpy as np


def localization_error(
    source_estimate: np.ndarray,    # (n_sources,) magnitudes
    source_positions: np.ndarray,   # (n_sources, 3)
    true_position: np.ndarray,      # (3,)
) -> float:
    """
    Euclidean distance from the peak of the estimate to the true source.

    The "estimate" is treated as a vector of source magnitudes (one per
    candidate location), and we find its argmax.
    """
    mags = np.abs(source_estimate)
    if mags.ndim == 2:
        # If estimate is (n_sources, 3) -- vector sources -- take norm
        mags = np.linalg.norm(source_estimate, axis=1)
    peak_idx = int(np.argmax(mags))
    peak_pos = source_positions[peak_idx]
    return float(np.linalg.norm(peak_pos - true_position))


def psf_fwhm(
    source_estimate: np.ndarray,
    source_positions: np.ndarray,
    true_position: np.ndarray,
) -> float:
    """
    Full-width at half-maximum of the reconstruction around the true source.

    Measured as: distance from true source at which the smoothed magnitude
    of the estimate drops to half its peak value. Approximated by sorting
    sources by distance and finding the crossing point.
    """
    mags = np.abs(source_estimate)
    if mags.ndim == 2:
        mags = np.linalg.norm(source_estimate, axis=1)

    # Distance from each source to the true position
    dists = np.linalg.norm(source_positions - true_position, axis=1)

    # Sort by distance, get cumulative max (so we have a monotonic
    # envelope of "highest magnitude within radius r")
    order = np.argsort(dists)
    sorted_dists = dists[order]
    sorted_mags = mags[order]

    peak = sorted_mags.max()
    half_peak = peak / 2.0

    # Find smallest radius at which the *maximum* magnitude within that
    # radius drops below half_peak. Equivalently: find the largest
    # contiguous-from-source neighborhood where everything is above half.
    # Simpler version: find smallest distance at which mag falls below
    # half_peak.
    below = sorted_mags < half_peak
    if not below.any():
        return float(sorted_dists[-1])  # never drops, return outer radius
    first_below_idx = int(np.argmax(below))
    return float(sorted_dists[first_below_idx])


def depth_bias(
    source_estimate: np.ndarray,
    source_positions: np.ndarray,
    true_position: np.ndarray,
) -> float:
    """
    Signed depth error in mm, with the clinical sign convention:
      positive => estimated peak is *deeper* (closer to center) than truth
      negative => estimated peak is *more superficial* (farther from center)
                  than truth. This is the classic MNE surface-bias direction.

    Internally we work with the radial coordinate (distance from the head
    center). Since "deep" in EEG/MEG means "close to the center," depth
    bias is computed as (true_radius - est_radius), not the other way.
    """
    mags = np.abs(source_estimate)
    if mags.ndim == 2:
        mags = np.linalg.norm(source_estimate, axis=1)
    peak_idx = int(np.argmax(mags))
    est_radius = np.linalg.norm(source_positions[peak_idx])
    true_radius = np.linalg.norm(true_position)
    return float(true_radius - est_radius)


def summary(
    source_estimate: np.ndarray,
    source_positions: np.ndarray,
    true_position: np.ndarray,
) -> dict:
    """Compute all standard metrics in one call. Returns a dict."""
    return {
        'localization_error_mm': localization_error(
            source_estimate, source_positions, true_position
        ),
        'psf_fwhm_mm': psf_fwhm(
            source_estimate, source_positions, true_position
        ),
        'depth_bias_mm': depth_bias(
            source_estimate, source_positions, true_position
        ),
    }
