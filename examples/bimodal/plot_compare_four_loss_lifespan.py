# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------- style ----------------
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

# ---------------- colors ----------------
COLOR_MAP = {
    "Dynamic SBI (our method)":      "#4C92C3",
    "Amortized":                     "#bc5090",
    "Round-based (keep old data)":   "#ffa600",
    "Round-based (only keep new data)": "#2f9e44",
}
COLOR_REF = "#bc5090"   # Validation (for right panels)
COLOR_NET = "#8FBFE8"   # Training  (for right panels)

# ---------------- Root dirs for each method (we will append graph_dir/z) ----------------
METHOD_DIRS = {
    "regular":      "outputs/2025-10-12/07-12-36",
    "amortized":    "outputs/2025-10-12/07-12-36",
    "rounds_fill":  "outputs/2025-10-12/07-12-36", 
    "rounds_renew": "outputs/2025-10-12/07-12-36",
}

# Left column: display name -> node_dir (= each method's graph_dir/z)
LEFT_METHOD_DIRS = {
    "Dynamic SBI (our method)":           os.path.join(METHOD_DIRS["regular"],      "graph_dir", "z"),
    "Amortized":                           os.path.join(METHOD_DIRS["amortized"],    "graph_dir", "z"),
    "Round-based (keep old data)":         os.path.join(METHOD_DIRS["rounds_fill"],  "graph_dir", "z"),
    "Round-based (only keep new data)":    os.path.join(METHOD_DIRS["rounds_renew"], "graph_dir", "z"),
}

# ---------------- plot styles ----------------
LINE_KW_VAL   = dict(lw=1.8, linestyle="-")
LINE_KW_TRAIN = dict(lw=1.8, linestyle="--")

# Set to infinity to show all time; e.g. 150.0 means only plot the first 150 minutes
T_MAX_MINUTES = 150

def beautify_axes(ax):
    ax.tick_params(which="both", direction="in")
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.1)

# ---------- Left column: load *.pth time series from node_dir ----------
def load_curves_from_pth(node_dir):
    """
    Read:
      - elapsed_minutes.pth          -> t (E,)
      - loss_train_posterior.pth     -> lt (E,)
      - loss_val_posterior.pth       -> lv (E,)
      - n_samples_total.pth          -> ntot (E,)
    """
    def _ld(name):
        path = os.path.join(node_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return torch.load(path, map_location="cpu")
        # If you are on PyTorch >= 2.6 and hit safe unpickling issues, add: weights_only=False

    t    = np.asarray(_ld("elapsed_minutes.pth"), dtype=float)
    lt   = np.asarray(_ld("loss_train_posterior.pth"), dtype=float)
    lv   = np.asarray(_ld("loss_val_posterior.pth"), dtype=float)
    ntot = np.asarray(_ld("n_samples_total.pth"), dtype=float)
    return t, lt, lv, ntot

# ---------- Right column: read scatter histories from pth ----------
def load_pair(dir_path):
    """Load training/validation histories for a method and rebase time so min time = 0."""
    x = torch.load(os.path.join(dir_path, "graph_dir/z/train_id_history.pth"), map_location="cpu")
    y = torch.load(os.path.join(dir_path, "graph_dir/z/validation_id_history.pth"), map_location="cpu")
    x = np.asarray(x)
    y = np.asarray(y)
    t0 = min(x[:, 0].min(), y[:, 0].min())
    tx = (x[:, 0] - t0) / 60.0  # minutes
    ty = (y[:, 0] - t0) / 60.0
    return (tx, x[:, 1]), (ty, y[:, 1])

def plot_one_scatter(ax, pair, title, tmax=np.inf):
    """Right-column subplot: Training/Validation scatter; filter by tmax before plotting."""
    (tx, xid), (ty, yid) = pair
    if np.isfinite(tmax):
        m_tr = (tx <= tmax)
        m_va = (ty <= tmax)
        tx, xid = tx[m_tr], xid[m_tr]
        ty, yid = ty[m_va], yid[m_va]

    ax.scatter(tx, xid, marker='.', s=12, color=COLOR_NET, alpha=0.9, label="Training")
    ax.scatter(ty, yid, marker='.', s=12, color=COLOR_REF, alpha=0.9, label="Validation")
    ax.set_title(title)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Simulation ID")
    beautify_axes(ax)
    ax.legend(loc="best", frameon=False, fontsize=11, handlelength=2.0, markerscale=3.0 )

def main():
    # ---- Create a 2×3 figure: left column (2 subplots), right grid (2×2) ----
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 6.8))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.10, top=0.92, wspace=0.30, hspace=0.32)

    # -------------------------------------------------
    # Upper-left: Train/Validation Loss (from *.pth time series)
    # -------------------------------------------------
    ax_l_top = axes[0, 0]
    handles_left = []
    for display_name, node_dir in LEFT_METHOD_DIRS.items():
        t, lt, lv, ntot = load_curves_from_pth(node_dir)
        m = (t <= T_MAX_MINUTES)
        t_sel, lt_sel, lv_sel = t[m], lt[m], lv[m]

        h_val, = ax_l_top.plot(t_sel, lv_sel, color=COLOR_MAP[display_name], label=f"{display_name}", **LINE_KW_VAL)
        _      = ax_l_top.plot(t_sel, lt_sel, color=COLOR_MAP[display_name], **LINE_KW_TRAIN)
        handles_left.append(h_val)

    ax_l_top.set_xlabel("Time [min]")
    ax_l_top.set_ylabel("Train/Validation Loss")
    beautify_axes(ax_l_top)

    # Main legend (method names)
    leg1 = ax_l_top.legend(
        handles=handles_left,
        loc="upper left",
        bbox_to_anchor=(0.20, 1.00),
        frameon=False,
        fontsize=10,
        handlelength=1.2,
        labelspacing=0.25,
        borderpad=0.2,
    )
    ax_l_top.add_artist(leg1)

    # Line-style legend (Validation / Training)
    proxy_val, = ax_l_top.plot([], [], **LINE_KW_VAL,   color="k", label="Val")
    proxy_trn, = ax_l_top.plot([], [], **LINE_KW_TRAIN, color="k", label="Train")
    ax_l_top.legend(
        handles=[proxy_val, proxy_trn],
        loc="lower left",
        bbox_to_anchor=(0.7, 0.18),
        frameon=False,
        fontsize=10,
        handlelength=1.2,
        labelspacing=0.2,
        borderpad=0.2,
    )

    # -------------------------------------------------
    # Lower-left: Total Samples (from *.pth time series)
    # -------------------------------------------------
    ax_l_bot = axes[1, 0]
    handles_right = []
    for display_name, node_dir in LEFT_METHOD_DIRS.items():
        t, lt, lv, ntot = load_curves_from_pth(node_dir)
        m = (t <= T_MAX_MINUTES)
        t_sel, ntot_sel = t[m], ntot[m]
        h, = ax_l_bot.plot(t_sel, ntot_sel, color=COLOR_MAP[display_name], label=display_name, lw=1.8)
        handles_right.append(h)

    ax_l_bot.set_xlabel("Time [min]")
    ax_l_bot.set_ylabel("Total Samples")
    beautify_axes(ax_l_bot)
    ax_l_bot.legend(handles=handles_right, loc="best", frameon=False, fontsize=8.5, handlelength=2.0)

    # -------------------------------------------------
    # Right grid (2×2): Simulation ID vs Time scatter (apply T_MAX_MINUTES filter)
    # -------------------------------------------------
    pairs = {name: load_pair(path) for name, path in METHOD_DIRS.items()}

    order = [
        ("regular",      "Dynamic SBI (our method)",           axes[0, 1]),
        ("amortized",    "Amortized",                          axes[0, 2]),
        ("rounds_fill",  "Round-based (keep old data)",        axes[1, 1]),
        ("rounds_renew", "Round-based (only keep new data)",   axes[1, 2]),
    ]
    for key, title, ax in order:
        plot_one_scatter(ax, pairs[key], title, tmax=T_MAX_MINUTES)

    # ---- Save ----
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "comparison_os_a_round_amortized.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")

    plt.show()

if __name__ == "__main__":
    main()
