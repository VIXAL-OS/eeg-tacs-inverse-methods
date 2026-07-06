#!/usr/bin/env python3
"""
Q-E — observability ceiling on the REAL anatomical head. SVD the realistic
75-ch DWI-anisotropic FEM leadfield (free orientation, 492k central-GM sources)
to get its SNR-limited observable rank r_L and the observable modes ON the
cortex — the anatomical analogue of the spherical r_L=5 / rank-cliff / unifier.

Saves the spectrum + per-source observable-mode magnitudes + node coords for
plotting in the main venv. Run: cd /root/qe && simnibs_python realistic_rank.py
"""
import h5py, numpy as np, os, glob, json
SUBJ = os.environ.get("QE_SUBJ", "sub-010004")
QE = os.path.join("/root/qe", SUBJ)
OUT = ("/mnt/d/Documents/Mycelial Institute/time-shenanigans/"
       "bandr_inverse/figures/viz_data")
os.makedirs(OUT, exist_ok=True)

p = glob.glob(os.path.join(QE, "leadfield_aniso", "*.hdf5"))[0]
with h5py.File(p, "r") as f:
    L = f["mesh_leadfield"]["leadfields"]["tdcs_leadfield"][:]   # (n_e, N, 3)
    nodes = f["mesh_leadfield"]["nodes"]["node_coord"][:]        # (N, 3)
n_e, N, _ = L.shape
print(f"leadfield {L.shape} -> free-orientation ({n_e} x {3*N})")

Lf = L.reshape(n_e, 3 * N)                     # free-orientation operator
# per-electrode centering is not needed; take the raw operator SVD
U, S, Vt = np.linalg.svd(Lf, full_matrices=False)   # S: (n_e,)
s0 = S[0]
sn = S / s0
summary = {"n_elec": int(n_e), "n_src": int(N)}
for snr in (3, 10, 30):
    hard = int((S > s0 / snr).sum())
    s2 = sn ** 2
    soft = float((s2 / (s2 + (1.0 / snr) ** 2)).sum())
    summary[f"r_L_hard_snr{snr}"] = hard
    summary[f"r_L_soft_snr{snr}"] = round(soft, 2)
    print(f"  snr={snr:2d}: r_L hard={hard:2d}  soft={soft:.2f}  of {n_e}")

# observable modes on the cortex: per-source magnitude of the top modes
K = 6
V = Vt[:K].T                                   # (3N, K)
mode_mag = np.linalg.norm(V.reshape(N, 3, K), axis=1)   # (N, K) orientation-free
sensitivity = np.linalg.norm(Lf, axis=0).reshape(N, 3)
sens = np.linalg.norm(sensitivity, axis=1)     # per-source ||L[:,i,:]||

np.savez(os.path.join(OUT, f"realistic_rank_{SUBJ}.npz"),
         S=S.astype(np.float32), sn=sn.astype(np.float32),
         nodes=nodes.astype(np.float32),
         mode_mag=mode_mag.astype(np.float32),
         sigma=sn[:K].astype(np.float32),
         sensitivity=sens.astype(np.float32))
json.dump(summary, open(os.path.join(OUT, f"realistic_rank_{SUBJ}_summary.json"), "w"),
          indent=2)
print("spectrum sn[:8] =", np.round(sn[:8], 4).tolist())
print(f"saved realistic_rank.npz + summary -> {OUT}")
