"""
Realistic head geometry via MNE-Python BEM forward modeling.

This module is the Q-B counterpart to spherical_forward.py. Same role:
provide a leadfield L mapping cortical dipole moments to scalp EEG
measurements, plus accompanying source positions/normals. But instead
of using the analytic spherical solution, we use a 3-layer BEM solution
on triangulated boundary surfaces extracted from a subject's MRI.

The downstream solvers in inverse_solvers.py don't care which forward
model produced L. That's the whole point of Q-B: confirm the methods
generalize from sphere to real geometry without code changes.

Coordinate systems
------------------
MNE-Python uses several coordinate frames. We work primarily in HEAD
coordinates: a right-handed frame with origin midway between the
auricular points, +X toward right preauricular, +Y toward nasion,
+Z toward vertex. Units are *meters* in MNE (unlike our mm convention
in the spherical code). All conversions are done in this module so
callers can keep using mm if they want.

Why MNE-Python over rolling our own
-----------------------------------
BEM forward modeling on real geometry requires:
  - MRI segmentation into scalp/skull/brain compartments
  - Surface triangulation
  - Linear collocation BEM with isolated-skull approach
  - Source space construction on the cortical surface
  - Coregistration between MRI and electrode coordinates

Each of these is a paper unto itself. MNE-Python provides
well-validated implementations that match FieldTrip and Brainstorm
results. Using them is the right move; rolling our own would be
reinventing a wheel that has 20 years of bug fixes.
"""
from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class HeadGeometry:
    """
    Bundle of everything we need from a subject's MRI + electrode setup
    to do source localization.

    Attributes
    ----------
    leadfield : (n_elec, n_sources) array
        Gain matrix L mapping cortical dipole moments (oriented normal
        to cortex) to scalp EEG potentials. Units: V / (A·m).
    source_pos_mm : (n_sources, 3) array
        Position of each cortical vertex in HEAD coordinates, mm.
    source_normals : (n_sources, 3) array
        Outward cortical-surface normal at each vertex.
    electrode_pos_mm : (n_elec, 3) array
        Electrode positions in HEAD coordinates, mm.
    electrode_names : list of str
    subject_id : str
        Subject identifier (e.g., 'sample', 'sub-010001').
    """
    leadfield: np.ndarray
    source_pos_mm: np.ndarray
    source_normals: np.ndarray
    electrode_pos_mm: np.ndarray
    electrode_names: list
    subject_id: str

    @property
    def n_sources(self) -> int:
        return self.source_pos_mm.shape[0]

    @property
    def n_electrodes(self) -> int:
        return self.electrode_pos_mm.shape[0]

    def __repr__(self) -> str:
        return (f"HeadGeometry(subject={self.subject_id}, "
                f"n_elec={self.n_electrodes}, n_sources={self.n_sources}, "
                f"L.shape={self.leadfield.shape}, "
                f"cond(L)={np.linalg.cond(self.leadfield):.2e})")


def load_mne_sample(
    spacing: str = 'oct6',
    use_eeg_only: bool = True,
    fixed_orientation: bool = True,
    verbose: bool = False,
) -> HeadGeometry:
    """
    Load MNE-Python's built-in sample subject and build a forward solution.

    The sample dataset is ~1.5 GB and downloads on first call to
    `mne.datasets.sample.data_path()`. After that it's cached.

    Parameters
    ----------
    spacing : str
        Source-space spacing. 'oct6' = octahedral subdivision level 6,
        gives ~4098 sources per hemisphere (~8200 total) at ~4.9mm
        spacing. 'oct5' gives ~1026/hemisphere. 'ico4' is similar to
        oct6. Coarser spacing = faster but lower spatial resolution.
    use_eeg_only : bool
        If True, drop MEG channels and use only EEG sensors.
    fixed_orientation : bool
        If True, constrain dipoles to be normal to cortex (one scalar
        per vertex). Matches our spherical-code convention. If False,
        free orientation (3 dipole components per vertex).
    verbose : bool
        Pass through to MNE-Python.

    Returns
    -------
    geom : HeadGeometry
    """
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "MNE-Python not installed. Run `pip install mne` "
            "(or activate the .venv that has it)."
        ) from e

    data_path = mne.datasets.sample.data_path()
    subjects_dir = os.path.join(data_path, 'subjects')
    subject = 'sample'

    # Use the prebuilt raw to get the info (channel locations, etc.)
    raw_fname = os.path.join(
        data_path, 'MEG', subject, 'sample_audvis_raw.fif'
    )
    raw = mne.io.read_raw_fif(raw_fname, verbose=verbose)
    info = raw.info

    if use_eeg_only:
        # Keep only EEG channels
        eeg_picks = mne.pick_types(info, eeg=True, meg=False, exclude=[])
        info = mne.pick_info(info, eeg_picks)

    # Source space: tessellation of the cortical surface
    src = mne.setup_source_space(
        subject, spacing=spacing, subjects_dir=subjects_dir,
        add_dist=False, verbose=verbose,
    )

    # BEM: 3-layer (scalp, outer skull, inner skull) for EEG
    conductivity = (0.3, 0.006, 0.3)  # S/m -- standard MNE defaults
    bem_model = mne.make_bem_model(
        subject=subject, ico=4, conductivity=conductivity,
        subjects_dir=subjects_dir, verbose=verbose,
    )
    bem = mne.make_bem_solution(bem_model, verbose=verbose)

    # Co-registration: sample data ships with a precomputed trans file
    trans_fname = os.path.join(
        data_path, 'MEG', subject, 'sample_audvis_raw-trans.fif'
    )

    # Forward solution
    fwd = mne.make_forward_solution(
        info, trans=trans_fname, src=src, bem=bem,
        meg=False, eeg=True, mindist=5.0, n_jobs=1, verbose=verbose,
    )

    if fixed_orientation:
        fwd = mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=True,
            use_cps=True, verbose=verbose,
        )

    return _forward_to_geometry(fwd, info, subject_id=subject)


def _forward_to_geometry(fwd, info, subject_id: str) -> HeadGeometry:
    """Extract a HeadGeometry bundle from an MNE Forward object."""
    import mne

    # Leadfield: shape (n_chan, n_sources) when fixed orientation
    L = fwd['sol']['data'].copy()

    # Source positions in HEAD coordinates, in meters → convert to mm
    src = fwd['src']
    positions_m = np.concatenate([s['rr'][s['vertno']] for s in src])
    positions_mm = positions_m * 1000.0

    # Source normals (cortical surface normal at each vertex)
    normals = np.concatenate([s['nn'][s['vertno']] for s in src])

    # Electrode positions: pull from info
    elec_pos_m = np.array([
        ch['loc'][:3] for ch in info['chs']
        if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH
    ])
    elec_pos_mm = elec_pos_m * 1000.0
    elec_names = [
        ch['ch_name'] for ch in info['chs']
        if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH
    ]

    # Sanity: shapes should match
    n_elec = elec_pos_mm.shape[0]
    n_src = positions_mm.shape[0]
    if L.shape != (n_elec, n_src):
        raise RuntimeError(
            f"Leadfield shape mismatch: L is {L.shape}, "
            f"expected ({n_elec}, {n_src}). Check fixed_orientation flag."
        )

    return HeadGeometry(
        leadfield=L,
        source_pos_mm=positions_mm,
        source_normals=normals,
        electrode_pos_mm=elec_pos_mm,
        electrode_names=elec_names,
        subject_id=subject_id,
    )


def find_nearest_source(
    target_pos_mm: np.ndarray,
    geom: HeadGeometry,
) -> Tuple[int, float]:
    """
    Find the cortical vertex closest to a target position.

    Returns
    -------
    idx : int
        Index into geom.source_pos_mm.
    dist_mm : float
        Distance from target to nearest vertex, in mm.

    Use this to pick "test sources" at anatomically interesting
    locations specified by approximate HEAD coordinates. The nearest
    cortical vertex is what we actually use, since the source space
    is constrained to the cortical surface.
    """
    target = np.asarray(target_pos_mm, dtype=float).reshape(3)
    dists = np.linalg.norm(geom.source_pos_mm - target, axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])


def simulate_scalp_data(
    geom: HeadGeometry,
    source_idx: int,
    moment_amplitude: float = 10e-9,
    noise_snr_power: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic scalp data from a known cortical source.

    Mirrors the Q-A simulation: pick a vertex, give it a dipole moment
    normal to cortex with the specified amplitude, forward-project
    through the leadfield, add Gaussian noise.

    Parameters
    ----------
    geom : HeadGeometry
    source_idx : int
        Which vertex to activate.
    moment_amplitude : float
        Dipole moment magnitude in A·m. Typical neural dipole: 1e-8.
    noise_snr_power : float
        Power-SNR of signal to noise. SNR_power = 10 means signal
        variance is 10× noise variance.
    rng : np.random.Generator or None
        For reproducibility. None → fresh random state.

    Returns
    -------
    V_noisy : (n_elec,) array
        Synthetic scalp data with noise.
    V_clean : (n_elec,) array
        Same data without noise (for SNR verification).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Since the leadfield is in fixed-orientation form, each column
    # is the scalp response to a unit dipole oriented along the
    # cortical normal at that vertex. So we just scale the column:
    V_clean = geom.leadfield[:, source_idx] * moment_amplitude

    sig_var = np.var(V_clean)
    if sig_var <= 0:
        raise ValueError(
            f"Source at idx {source_idx} produces zero scalp signal. "
            f"This vertex might be on the medial wall or in a "
            f"poorly-sampled region. Pick another."
        )
    noise_var = sig_var / noise_snr_power
    V_noisy = V_clean + rng.normal(0, np.sqrt(noise_var), size=V_clean.shape)
    return V_noisy, V_clean
