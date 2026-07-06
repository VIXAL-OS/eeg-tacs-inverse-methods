#!/usr/bin/env python3
r"""Q-E — the observability ceiling on the REAL head: rank-cliff + observable
modes rendered on the actual cortex (the anatomical upgrade of the spherical
qd_rank_cliff / unifier). Run in the main venv from the realistic_rank arrays."""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VD = os.path.join(ROOT, "bandr_inverse", "figures", "viz_data")
FIG = os.path.join(ROOT, "bandr_inverse", "figures")

d = np.load(os.path.join(VD, "realistic_rank.npz"))
s = json.load(open(os.path.join(VD, "realistic_rank_summary.json")))
S, sn, nodes = d["S"], d["sn"], d["nodes"]
mode_mag, sens = d["mode_mag"], d["sensitivity"]
n_e = s["n_elec"]

# ---- rank cliff ----
fig, ax = plt.subplots(figsize=(7.4, 4.8))
idx = np.arange(1, S.size + 1)
for snr, col in [(3, "#e76f51"), (10, "#e9c46a"), (30, "#9aa6b6")]:
    rL = s[f"r_L_hard_snr{snr}"]
    ax.axhline(1.0 / snr, color=col, ls="--", lw=1.3, alpha=0.9,
               label=f"SNR {snr} floor → r_L={rL}")
ax.axvspan(0.5, s["r_L_hard_snr3"] + 0.5, color="#2A9D8F", alpha=0.12, zorder=0)
ax.plot(idx, sn, "o-", color="#264653", ms=5, lw=1.2)
ax.set_yscale("log"); ax.set_xlim(0, n_e + 1); ax.set_ylim(sn.min() * 0.6, 1.4)
ax.set_xlabel("singular-value index"); ax.set_ylabel(r"$\sigma_k/\sigma_1$")
ax.set_title("Observability ceiling on the REAL head\n"
             f"75-ch DWI-anisotropic FEM leadfield (492k cortical sources): "
             f"r_L = {s['r_L_hard_snr3']} hard / {s['r_L_soft_snr3']} soft at SNR 3")
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.grid(True, which="both", axis="y", alpha=0.2)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(FIG, f"realistic_rank_cliff.{ext}"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- observable modes on the cortex (lateral view: world y-z = A-P vs S-I) ----
rng = np.random.default_rng(0)
sub = rng.choice(nodes.shape[0], min(60000, nodes.shape[0]), replace=False)
Y, Z = nodes[sub, 1], nodes[sub, 2]
def signed(v):
    return v / (np.percentile(np.abs(v), 99) + 1e-12)
panels = [("sensitivity ‖L‖", np.log10(sens[sub] + 1e-12), "magma", None),
          ("observable mode 1", signed(mode_mag[sub, 0]), "viridis", None),
          ("observable mode 2", signed(mode_mag[sub, 1]), "viridis", None),
          ("observable mode 3", signed(mode_mag[sub, 2]), "viridis", None)]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
for ax, (title, val, cmap, _) in zip(axes, panels):
    sc = ax.scatter(Y, Z, c=val, s=2, cmap=cmap, alpha=0.8, edgecolors="none",
                    vmin=np.percentile(val, 2), vmax=np.percentile(val, 98))
    ax.set_title(title, fontsize=10); ax.set_aspect("equal"); ax.axis("off")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
fig.suptitle("Observable subspace on the real cortex (charm central-GM, lateral view). "
             "Only ~3 modes clear the SNR floor — the ceiling, made anatomical.",
             fontsize=11, y=1.03)
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(FIG, f"realistic_cortex_modes.{ext}"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("r_L(snr3) =", s["r_L_hard_snr3"], "hard /", s["r_L_soft_snr3"], "soft")
print("saved realistic_rank_cliff + realistic_cortex_modes")
