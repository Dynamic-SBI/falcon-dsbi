import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# ---- Colors consistent with the big figure ----
COLOR_REF = "#bc5090"   # Validation
COLOR_NET = "#8FBFE8"   # Training

# ---- Style consistent with the big figure ----
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.linewidth": 1.1,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.formatter.useoffset": False, "axes.formatter.use_mathtext": True,
})

# ---- Load and organize data ----
x = torch.load("outputs/2025-10-12/07-12-36/graph_dir/z/train_id_history.pth")
y = torch.load("outputs/2025-10-12/07-12-36/graph_dir/z/validation_id_history.pth")
x = np.asarray(x)
y = np.asarray(y)

# Align time so that it starts from 0 (use the minimum timestamp of both)
t0 = min(x[:, 0].min(), y[:, 0].min())
tx = (x[:, 0] - t0) / 60.0  # minutes
ty = (y[:, 0] - t0) / 60.0

# ---- Plot ----
fig, ax = plt.subplots(figsize=(6.2, 3.8))

# Training (blue) and Validation (pink); adjust marker and size as needed
ax.scatter(tx, x[:, 1], marker='.', s=8, color=COLOR_NET, alpha=0.9, label="Training")
ax.scatter(ty, y[:, 1], marker='.', s=8, color=COLOR_REF, alpha=0.9, label="Validation")

# Axes and grid style consistent with the big figure
ax.set_xlabel("Time [min]")
ax.set_ylabel("Simulation ID")
ax.tick_params(which="both", direction="in")
for side in ("left", "right", "top", "bottom"):
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(1.1)

# Legend style consistent with the big figure
ax.legend(loc="best", frameon=False, fontsize=12, handlelength=2.0, markerscale=3.0)

fig.tight_layout()
out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "sample_lifespan.png")
fig.savefig(out_path, bbox_inches="tight")
plt.show()
