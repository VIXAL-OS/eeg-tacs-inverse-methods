"""
Q-D: network-pattern recovery on the spherical head model.

The question this module operationalizes: can a region-grouped
("network-prior") inverse recover a distributed multi-node network of
co-active sources, where standard per-voxel methods (sLORETA, eLORETA)
smear that network into a single blob? Q-A handled single dipoles; this
asks whether a coordinated set of ~10 sources can be resolved as
distinct nodes rather than averaged into one peak.

Network-prior parameterization
------------------------------
We model the source distribution as

    J = G a

where G is an (n_sources, n_parcels) indicator matrix over a parcellation
of the source space and a is a per-parcel amplitude. The inverse problem
collapses from ~thousands of voxels to ~tens of parcels:

    a_hat = MNE_operator(L @ G, snr) @ V
    J_hat = G @ a_hat

This is solved with the existing mne_operator on the reduced leadfield
L_net = L @ G — no new linear-algebra; just a different prior.

Critical inverse-crime guard
----------------------------
The parcellation G is computed from source coordinates ONLY — never from
ground-truth source activity, and never from knowledge of where the test
network's nodes lie. Aligning parcels to the true network would make the
method look artificially good; it's the network-recovery analogue of the
classical per-voxel inverse crime (reusing the same forward solve or
exact source positions for both generation and inversion).

The notebook also runs an "oracle Voronoi" partition (each true node is
its own parcel) as a ceiling reference — NOT a real method, included only
to quantify how much the blind partition's mismatch with biology costs.
"""
from __future__ import annotations
import numpy as np
from typing import Literal
from scipy.cluster.vq import kmeans2

from spherical_forward import SphericalHeadModel, forward_potential
from inverse_solvers import mne_operator, sloreta_operator


# ---------------------------------------------------------------------------
# Ground-truth network generation
# ---------------------------------------------------------------------------

def sample_network_positions(
    model: SphericalHeadModel,
    n_nodes: int = 12,
    min_separation_mm: float = 18.0,
    margin_mm: float = 8.0,
    slab_y_mm: float | None = None,
    rng: np.random.Generator | None = None,
    max_attempts: int = 200_000,
) -> np.ndarray:
    """Rejection-sample n_nodes positions inside the brain compartment.

    Enforces a minimum pairwise separation (so nodes are spatially distinct)
    and a margin from the brain/CSF interface. If slab_y_mm is given, all
    nodes are constrained to |y| <= slab_y_mm — useful for keeping a
    distributed network visible in the xz-plane slice plot.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    R_max = model.radii_mm[0] - margin_mm
    pts: list[np.ndarray] = []
    attempts = 0
    while len(pts) < n_nodes and attempts < max_attempts:
        attempts += 1
        cand = rng.uniform(-R_max, R_max, size=3)
        if slab_y_mm is not None:
            cand[1] = rng.uniform(-slab_y_mm, slab_y_mm)
        if np.linalg.norm(cand) > R_max:
            continue
        if pts:
            d_min = np.min(np.linalg.norm(np.array(pts) - cand, axis=1))
            if d_min < min_separation_mm:
                continue
        pts.append(cand)
    if len(pts) < n_nodes:
        raise RuntimeError(
            f"Only placed {len(pts)}/{n_nodes} nodes in {max_attempts} attempts; "
            f"reduce min_separation_mm or n_nodes."
        )
    return np.array(pts)


def sample_clustered_network_positions(
    model: SphericalHeadModel,
    cluster_centers_mm: np.ndarray,        # (n_clusters, 3)
    nodes_per_cluster: int = 4,
    cluster_radius_mm: float = 12.0,
    min_separation_mm: float = 6.0,
    margin_mm: float = 8.0,
    slab_y_mm: float | None = None,
    rng: np.random.Generator | None = None,
    max_attempts: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample nodes_per_cluster positions within cluster_radius_mm of each
    cluster center, with a minimum within-cluster pairwise separation.

    Used to test whether a fine-partition network-prior can resolve nodes
    inside a tight cluster that single-blob per-voxel methods fuse into one
    peak. slab_y_mm constrains all nodes to |y| <= slab_y_mm for the
    xz-plane slice visualization.

    Returns
    -------
    positions : (n_clusters * nodes_per_cluster, 3)
    cluster_id : (n_clusters * nodes_per_cluster,) int
    """
    if rng is None:
        rng = np.random.default_rng(0)
    R_brain = model.radii_mm[0] - margin_mm
    centers = np.asarray(cluster_centers_mm, dtype=float)

    all_pts: list[np.ndarray] = []
    all_cid: list[int] = []
    for c, center in enumerate(centers):
        pts: list[np.ndarray] = []
        attempts = 0
        while len(pts) < nodes_per_cluster and attempts < max_attempts:
            attempts += 1
            offset = rng.normal(0.0, cluster_radius_mm / 2.0, size=3)
            cand = center + offset
            if slab_y_mm is not None:
                cand[1] = center[1] + rng.uniform(-slab_y_mm, slab_y_mm)
            if np.linalg.norm(cand) > R_brain:
                continue
            if np.linalg.norm(cand - center) > cluster_radius_mm:
                continue
            if pts:
                d = np.min(np.linalg.norm(np.array(pts) - cand, axis=1))
                if d < min_separation_mm:
                    continue
            pts.append(cand)
        if len(pts) < nodes_per_cluster:
            raise RuntimeError(
                f"Cluster {c} only placed {len(pts)}/{nodes_per_cluster} nodes "
                f"in {max_attempts} attempts (cluster_radius={cluster_radius_mm}, "
                f"min_separation={min_separation_mm})."
            )
        all_pts.extend(pts)
        all_cid.extend([c] * nodes_per_cluster)
    return np.array(all_pts), np.array(all_cid, dtype=int)


def make_correlated_node_timecourses(
    n_nodes: int,
    n_times: int,
    shared_fraction: float = 0.6,
    cutoff_norm_freq: float = 0.08,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-node low-pass-filtered amplitude time-courses with shared structure.

    Each node's signal is
        a_k(t) = shared_fraction * carrier(t)
                 + (1 - shared_fraction) * private_k(t)
    where carrier and private_k are independent low-pass-filtered Gaussian
    noise (same spectral bandwidth). All signals are zero-mean, unit-RMS.

    shared_fraction=1.0 gives a rank-1 (perfectly coherent) network;
    shared_fraction=0.0 gives independent nodes. Default 0.6 makes the
    per-parcel time-course correlation metric a non-trivial test of whether
    the inverse correctly assigns activity to the right parcels (vs. just
    matching the global oscillation phase).
    """
    if rng is None:
        rng = np.random.default_rng(1)

    def lp_noise(n):
        x = rng.standard_normal(n)
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n)
        H = 1.0 / (1.0 + (freqs / cutoff_norm_freq) ** 4)
        y = np.fft.irfft(X * H, n=n)
        y -= y.mean()
        rms = np.sqrt(np.mean(y ** 2))
        return y / max(rms, 1e-30)

    carrier = lp_noise(n_times)
    A = np.zeros((n_nodes, n_times))
    for k in range(n_nodes):
        priv = lp_noise(n_times)
        sig = shared_fraction * carrier + (1.0 - shared_fraction) * priv
        sig -= sig.mean()
        rms = np.sqrt(np.mean(sig ** 2))
        A[k] = sig / max(rms, 1e-30)
    return A


def simulate_network_scalp_data(
    node_positions: np.ndarray,            # (n_nodes, 3) mm
    node_amplitudes: np.ndarray,           # (n_nodes, n_times) A·m
    electrode_positions: np.ndarray,       # (n_elec, 3) mm
    model: SphericalHeadModel,
    amplitude_snr: float = 3.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum analytic-forward contributions from a network of unit-radial
    dipoles modulated by node_amplitudes, then add white sensor noise.

    amplitude_snr is RMS(V_clean) / RMS(noise) — the convention used by
    the inverse solvers (MNE-Python convention).

    Returns
    -------
    V_noisy : (n_elec, n_times)
    V_clean : (n_elec, n_times)
    """
    if rng is None:
        rng = np.random.default_rng(2)
    K = node_positions.shape[0]
    n_elec = electrode_positions.shape[0]
    H = np.zeros((n_elec, K))
    for k in range(K):
        pos = node_positions[k]
        r = np.linalg.norm(pos)
        moment = pos / max(r, 1e-30)  # unit radial dipole
        H[:, k] = forward_potential(pos, moment, electrode_positions,
                                    model=model)
    V_clean = H @ node_amplitudes
    sig_rms = np.sqrt(np.mean(V_clean ** 2))
    noise_rms = sig_rms / max(amplitude_snr, 1e-30)
    V = V_clean + rng.normal(0.0, noise_rms, size=V_clean.shape)
    return V, V_clean


# ---------------------------------------------------------------------------
# Blind parcellation of the source space
# ---------------------------------------------------------------------------

def parcellate_source_space(
    source_positions: np.ndarray,
    n_parcels: int,
    random_state: int = 0,
) -> np.ndarray:
    """K-means parcellation of source positions.

    Uses ONLY the source-space coordinates — no knowledge of ground-truth
    network positions or activity. This is the blind partition required to
    avoid the network-recovery inverse crime.
    """
    _, labels = kmeans2(
        source_positions.astype(float), n_parcels,
        minit='++', seed=random_state,
    )
    return labels.astype(int)


def parcel_indicator_matrix(
    labels: np.ndarray,
    n_parcels: int | None = None,
    normalize: Literal['binary', 'mean', 'unit_l2'] = 'unit_l2',
) -> np.ndarray:
    """Build the (n_sources, n_parcels) indicator matrix G used to express
    J = G @ a.

    normalize:
      'binary'  — G[i,k] = 1 if source i in parcel k else 0.
      'mean'    — G[i,k] = 1/|parcel_k|; a_k is the total parcel current.
      'unit_l2' — column k has unit L2 norm; MNE regularizes parcels evenly
                  regardless of their size. This is the default.
    """
    n_src = len(labels)
    if n_parcels is None:
        n_parcels = int(labels.max()) + 1
    G = np.zeros((n_src, n_parcels))
    for k in range(n_parcels):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        if normalize == 'binary':
            G[idx, k] = 1.0
        elif normalize == 'mean':
            G[idx, k] = 1.0 / len(idx)
        elif normalize == 'unit_l2':
            G[idx, k] = 1.0 / np.sqrt(len(idx))
        else:
            raise ValueError(f"unknown normalize={normalize!r}")
    return G


# ---------------------------------------------------------------------------
# Network-prior inverse
# ---------------------------------------------------------------------------

def network_prior_operator(
    leadfield: np.ndarray,                 # (n_elec, n_src)
    parcel_labels: np.ndarray,             # (n_src,)
    snr: float = 3.0,
    n_parcels: int | None = None,
    normalize: Literal['binary', 'mean', 'unit_l2'] = 'unit_l2',
    method: Literal['mne', 'sloreta'] = 'mne',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Network-prior inverse: assume J = G a, solve for a on the reduced
    leadfield L_net = L @ G.

    method='mne'     — plain minimum-norm on L_net. Has a winner-take-all
                       failure mode when several parcels in the same region
                       have correlated leadfield columns: the regularized
                       least-squares solution concentrates amplitude on the
                       column best aligned with the scalp signal and shrinks
                       the rest toward zero, even when the true source mass
                       should be spread across all of them.

    method='sloreta' — sLORETA-style standardization on L_net (per-parcel
                       resolution-matrix normalization). Designed to reduce
                       that bias by dividing each row of the MNE operator
                       by sqrt(diag(R)) where R = K_mne @ L_net.

    Reuses inverse_solvers.{mne,sloreta}_operator on L_net, so the snr
    convention is identical to the per-voxel solvers in this codebase.

    Returns
    -------
    K_voxel : (n_src, n_elec) — J_hat = K_voxel @ V on the original source
              space, with all voxels in a parcel sharing one time course.
    K_parcel : (n_parcels, n_elec) — a_hat = K_parcel @ V directly.
    G : (n_src, n_parcels) — the indicator matrix used.
    """
    L = np.asarray(leadfield, dtype=float)
    G = parcel_indicator_matrix(parcel_labels, n_parcels=n_parcels,
                                normalize=normalize)
    L_net = L @ G
    if method == 'mne':
        K_parcel = mne_operator(L_net, snr=snr, depth_weighting=0.0)
    elif method == 'sloreta':
        K_parcel = sloreta_operator(L_net, snr=snr)
    else:
        raise ValueError(f"unknown method={method!r}")
    K_voxel = G @ K_parcel
    return K_voxel, K_parcel, G


# ---------------------------------------------------------------------------
# Evaluation utilities (these may use ground truth — NOT used in the inverse)
# ---------------------------------------------------------------------------

def per_source_amplitude(J_hat: np.ndarray) -> np.ndarray:
    """RMS over time for (n_src, T), absolute value for (n_src,)."""
    if J_hat.ndim == 2:
        return np.sqrt(np.mean(J_hat ** 2, axis=1))
    return np.abs(J_hat)


def parcel_mean_amplitude(
    per_source: np.ndarray,
    labels: np.ndarray,
    n_parcels: int | None = None,
) -> np.ndarray:
    """Mean per-source amplitude within each parcel (0 for empty parcels)."""
    if n_parcels is None:
        n_parcels = int(labels.max()) + 1
    out = np.zeros(n_parcels)
    for k in range(n_parcels):
        idx = np.where(labels == k)[0]
        if len(idx) > 0:
            out[k] = per_source[idx].mean()
    return out


def parcel_time_course(
    J_hat: np.ndarray,                     # (n_src, T)
    labels: np.ndarray,
    n_parcels: int | None = None,
) -> np.ndarray:
    """Mean signed time-course per parcel: shape (n_parcels, T)."""
    if n_parcels is None:
        n_parcels = int(labels.max()) + 1
    out = np.zeros((n_parcels, J_hat.shape[1]))
    for k in range(n_parcels):
        idx = np.where(labels == k)[0]
        if len(idx) > 0:
            out[k] = J_hat[idx].mean(axis=0)
    return out


def true_parcel_assignment(
    node_positions: np.ndarray,
    source_positions: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """For each true node, the parcel index of its nearest source-grid point.

    Used ONLY for evaluation; never passed to the inverse method.
    """
    out = np.empty(len(node_positions), dtype=int)
    for i, pos in enumerate(node_positions):
        d = np.linalg.norm(source_positions - pos, axis=1)
        out[i] = int(labels[int(np.argmin(d))])
    return out


def parcel_detection_pr(
    recovered_parcel_amp: np.ndarray,
    true_active_parcels: np.ndarray,
    top_k: int | None = None,
) -> tuple[float, float, float, np.ndarray]:
    """Precision/recall/F1 for selecting the top-K parcels by recovered
    amplitude vs the ground-truth active parcel set.

    Default top_k = number of true active parcels (fair test that uses no
    information beyond the *count* of active parcels).
    """
    truth_set = set(int(x) for x in true_active_parcels)
    if top_k is None:
        top_k = len(truth_set)
    order = np.argsort(recovered_parcel_amp)[::-1]
    selected = order[:top_k]
    sel_set = set(int(x) for x in selected)
    tp = len(sel_set & truth_set)
    prec = tp / max(len(sel_set), 1)
    rec = tp / max(len(truth_set), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return prec, rec, f1, selected


def per_node_within_cluster_recovery(
    per_source: np.ndarray,                # (n_src,)
    node_positions: np.ndarray,            # (n_nodes, 3)
    cluster_ids: np.ndarray,               # (n_nodes,)
    source_positions: np.ndarray,
    cluster_search_radius_mm: float = 25.0,
) -> np.ndarray:
    """For each true node, recovered amplitude at the nearest grid point
    normalized by the MAX recovered amplitude within the node's cluster
    region (sources within cluster_search_radius_mm of any same-cluster node).

    This metric is the within-cluster analogue of per_node_recovery_scores
    in the notebook: R = 1 means the node is the dominant spot inside its
    cluster's neighborhood; R << 1 means the cluster has a peak that's not
    at this node. A per-voxel method that fuses each 4-node cluster into a
    single blob will have R ~ 1 only for the one node nearest the blob —
    one node "resolved" per cluster. A finer-partition method that splits
    amplitude across multiple parcels per cluster can have R ~ 1 at
    several nodes per cluster.
    """
    n_nodes = len(node_positions)
    out = np.full(n_nodes, np.nan)
    for i in range(n_nodes):
        same_cluster = node_positions[cluster_ids == cluster_ids[i]]
        in_cluster = np.zeros(len(source_positions), dtype=bool)
        for p in same_cluster:
            d = np.linalg.norm(source_positions - p, axis=1)
            in_cluster |= (d <= cluster_search_radius_mm)
        if not in_cluster.any():
            continue
        cluster_max = per_source[in_cluster].max()
        nearest = int(np.argmin(
            np.linalg.norm(source_positions - node_positions[i], axis=1)
        ))
        out[i] = per_source[nearest] / max(cluster_max, 1e-30)
    return out


def parcel_timecourse_correlation(
    recovered_tc: np.ndarray,              # (n_parcels, T)
    true_tc: np.ndarray,                   # (n_parcels, T)
    active_parcels: np.ndarray,
) -> float:
    """Mean Pearson correlation over the active parcels between recovered
    and true per-parcel time courses. Skips parcels with zero variance
    (e.g. empty true parcels)."""
    corrs = []
    for k in active_parcels:
        a, b = recovered_tc[k], true_tc[k]
        if a.std() < 1e-30 or b.std() < 1e-30:
            continue
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(corrs)) if corrs else float('nan')
