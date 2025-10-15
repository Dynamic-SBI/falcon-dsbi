# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.patches import Patch
from JSD import compute_js_divergence

# ---------------- paths & constants ----------------
# 1) Your posterior samples (unchanged)
PATH_POSTERIOR   = "samples_posterior.joblib"

# 2) Training stats are now saved as separate files under this directory (modify to your actual path)
NODE_DIR = "outputs/2025-10-12/07-12-36/graph_dir/z"

# 3) Other data (unchanged)
PATH_X_OBS       = "data/x_obs_10dim.npy"
PATH_TRUE_Z      = "data/true_z_10dim.npy"
PATH_SHIFT       = "data/shift_10dim.npy"

THETA_TRUE_MODE  = "true_z"     # "mu1" / "mu2" / "midpoint" / "true_z"
SMOOTH_SIGMA_ERR = 0            # 0 means no smoothing
LIVE_N           = 128

DIR_OUT    = os.path.dirname(PATH_POSTERIOR) or "."
DIM        = 10
SIGMA      = 1e-2
MEAN_SHIFT = 3.0 * SIGMA
N_THEORY   = 1000
N_PLOT_MAX = 4000

# Colors (you may change only the colors; keep other linestyles)
COLOR_REF = "#bc5090"   # pink: parameter range “max”, Validation loss, err_min of extreme error
COLOR_NET = "#8FBFE8"   # light blue: parameter range “min”, Training loss, err_max of extreme error
LOSS_VAL_WARM   = "#e77081"  # red — validation loss
LOSS_TRAIN_COOL = "#00857b"  # blue — training loss

RANGE_MAX_WARM  = "#fe9a81"  # coral orange — parameter range: max
RANGE_MIN_COOL  = "#817DEE"  # deep purple  — parameter range: min

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

# ---------------- helpers ----------------
def load_posterior(path, dim):
    data = joblib.load(path)
    if isinstance(data, list):
        assert len(data) and isinstance(data[0], dict) and "z" in data[0]
        Z = np.stack([d["z"] for d in data], axis=0)
    elif isinstance(data, dict):
        assert "z" in data
        Z = np.asarray(data["z"])
    else:
        raise RuntimeError("posterior joblib 格式未知")
    assert Z.ndim == 2 and Z.shape[1] == dim
    return torch.as_tensor(Z, dtype=torch.float64)

def choose_global_scale(arrays, target_max=12.0):
    vmax = max(float(np.nanmax(np.abs(a))) for a in arrays)
    if vmax <= 0:
        return 0, 1.0
    k = int(np.floor(np.log10(vmax / target_max)))
    return k, 10.0**k

def set_two_ticks_4dp(ax, positions, axis="x"):
    labels = [f"{p:.4f}" for p in positions]
    if axis == "x":
        ax.xaxis.set_major_locator(FixedLocator(positions))
        ax.xaxis.set_major_formatter(FixedFormatter(labels))
    else:
        ax.yaxis.set_major_locator(FixedLocator(positions))
        ax.yaxis.set_major_formatter(FixedFormatter(labels))

def _safe_load_pth(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)

def load_training_stats(node_dir):
    """Read training metrics and parameter ranges saved as separate files under node_dir.
       Returns dict: epochs, loss_train, loss_val, theta_mins, theta_maxs
       where theta_* have shape (B, D)
    """
    node_dir = os.path.abspath(node_dir)
    if not os.path.isdir(node_dir):
        raise NotADirectoryError(node_dir)

    epochs_path      = os.path.join(node_dir, "epochs.pth")
    loss_tr_path     = os.path.join(node_dir, "loss_train_posterior.pth")
    loss_val_path    = os.path.join(node_dir, "loss_val_posterior.pth")
    th_mins_b_path   = os.path.join(node_dir, "theta_mins_batches.pth")
    th_maxs_b_path   = os.path.join(node_dir, "theta_maxs_batches.pth")

    epochs      = np.asarray(_safe_load_pth(epochs_path), dtype=float)
    loss_train  = np.asarray(_safe_load_pth(loss_tr_path), dtype=float)
    loss_val    = np.asarray(_safe_load_pth(loss_val_path), dtype=float)

    # theta_*_batches are saved as list[np.ndarray(D,)]; stack them to (B, D)
    theta_mins_batches = _safe_load_pth(th_mins_b_path)  # list of (D,)
    theta_maxs_batches = _safe_load_pth(th_maxs_b_path)  # list of (D,)
    if len(theta_mins_batches) == 0 or len(theta_maxs_batches) == 0:
        raise RuntimeError("theta_mins_batches/theta_maxs_batches are empty; cannot plot.")
    theta_mins = np.vstack([np.asarray(x) for x in theta_mins_batches])  # (B, D)
    theta_maxs = np.vstack([np.asarray(x) for x in theta_maxs_batches])  # (B, D)

    return dict(
        epochs=epochs,
        loss_train=loss_train,
        loss_val=loss_val,
        theta_mins=theta_mins,
        theta_maxs=theta_maxs,
    )

def compute_extreme_error_series(node_dir, path_x_obs, path_true_z, path_shift):
    """Return (x, err_min[B,D], err_max[B,D], noise_only, B).
       Training stats are read from *.pth files in node_dir.
    """
    stats = load_training_stats(node_dir)
    theta_mins = stats["theta_mins"]  # (B, D)
    theta_maxs = stats["theta_maxs"]  # (B, D)
    B, D = theta_mins.shape

    x_obs  = np.load(path_x_obs).astype(float).squeeze()
    assert x_obs.ndim == 1 and x_obs.size == D, "x_obs dimension mismatches theta_mins/theta_maxs."
    true_z = np.load(path_true_z).astype(float).squeeze()
    shift  = np.load(path_shift).astype(float).squeeze()

    mu1 = x_obs - MEAN_SHIFT
    mu2 = x_obs + MEAN_SHIFT
    if THETA_TRUE_MODE == "mu1":
        theta_true = mu1
    elif THETA_TRUE_MODE == "mu2":
        theta_true = mu2
    elif THETA_TRUE_MODE == "midpoint":
        theta_true = 0.5 * (mu1 + mu2)
    elif THETA_TRUE_MODE == "true_z":
        theta_true = true_z  # + shift if you need bias, enable here
    else:
        raise ValueError("THETA_TRUE_MODE must be one of 'mu1' / 'mu2' / 'midpoint' / 'true_z'.")

    err_min = np.abs(theta_mins - theta_true[None, :])   # (B, D)
    err_max = np.abs(theta_maxs - theta_true[None, :])   # (B, D)

    if SMOOTH_SIGMA_ERR and SMOOTH_SIGMA_ERR > 0:
        err_min = gaussian_filter1d(err_min, sigma=SMOOTH_SIGMA_ERR, axis=0)
        err_max = gaussian_filter1d(err_max, sigma=SMOOTH_SIGMA_ERR, axis=0)

    noise_only = SIGMA
    x = np.arange(B)
    return x, err_min, err_max, noise_only, B

def add_side_panels_and_corner_legend(fig, axes_grid, node_dir, corner_handles):
    """
    Layout:
      - top-left: Train/Validation loss
      - top-right: Evolution of z min/max (smoothed)   [min=blue, max=pink]
      - bottom-right: Dataset convergence ... (log y)
      - bottom-left: blank
    And place corner plot legend in the “blank space to the left of the loss plot”.
    """
    # Get bounds of corner subplots
    poss = [ax.get_position() for row in axes_grid for ax in row if ax.get_visible()]
    xmax = max(p.x1 for p in poss)
    ymax = max(p.y1 for p in poss)

    # —— geometry params ——
    gap_x_side = 0.012
    width_col  = 0.21
    height     = 0.21
    gap_col    = 0.090
    gap_row    = 0.090

    # column positions
    x0_left  = min(0.98 - (2*width_col + gap_col), xmax + gap_x_side)   # loss goes here
    x0_right = x0_left + width_col + gap_col                            # parameter range & extremes

    # row positions
    y0_bottom = max(0.12, ymax - (2*height + gap_row))
    y0_top    = y0_bottom + height + gap_row

    # Read training stats
    try:
        stats = load_training_stats(node_dir)
        epochs     = stats["epochs"]
        loss_train = stats["loss_train"]
        loss_val   = stats["loss_val"]
        theta_mins = stats["theta_mins"]
        theta_maxs = stats["theta_maxs"]
        B_common   = int(theta_mins.shape[0])
    except Exception as e:
        # If reading fails, annotate the error in the right panel and return
        ax_dummy = fig.add_axes([x0_left, y0_top, width_col, height])
        ax_dummy.axis("off")
        ax_dummy.text(0.5, 0.5, f"load_training_stats failed:\n{e}",
                      ha="center", va="center", transform=ax_dummy.transAxes)
        return

    # ---------- top-left: loss ----------
    ax_top_left = fig.add_axes([x0_left, y0_top, width_col, height])
    ax_top_left.plot(epochs, loss_train, color=LOSS_TRAIN_COOL, lw=1.8, label="Training", alpha=0.9)
    ax_top_left.plot(epochs, loss_val,   color=LOSS_VAL_WARM,  lw=1.0, label="Validation", alpha=0.9)
    ax_top_left.set_xlabel("Epoch")
    ax_top_left.set_ylabel("Train/Validation Loss")
    ax_top_left.tick_params(direction="in")
    for s in ("left", "right", "top", "bottom"):
        ax_top_left.spines[s].set_visible(True)
        ax_top_left.spines[s].set_linewidth(1.1)
    ax_top_left.legend(loc="upper right", frameon=False, fontsize=12, handlelength=2.0)

    # ---------- top-right: parameter ranges (min solid = blue, max dashed = pink) ----------
    ax_top_right = fig.add_axes([x0_right, y0_top, width_col, height])
    num_batches, _ = theta_mins.shape
    x = np.arange(num_batches)

    for s in ("left", "right", "top", "bottom"):
        ax_top_right.spines[s].set_visible(True)
        ax_top_right.spines[s].set_linewidth(1.1)
    ax_top_right.tick_params(which="both", direction="in", pad=2)

    # raw (light)
    ax_top_right.plot(x, theta_mins, linestyle='-',  color=RANGE_MIN_COOL, alpha=0.10)
    ax_top_right.plot(x, theta_maxs, linestyle='--', color=RANGE_MAX_WARM, alpha=0.10)
    # smoothed (single color for each)
    sm_min = gaussian_filter1d(theta_mins, sigma=200, axis=0)
    sm_max = gaussian_filter1d(theta_maxs, sigma=200, axis=0)
    ax_top_right.plot(x, sm_min, linestyle='-',  linewidth=1.6, color=RANGE_MIN_COOL)
    ax_top_right.plot(x, sm_max, linestyle='--', linewidth=1.6, color=RANGE_MAX_WARM)

    # unify x-axis
    ax_top_right.set_xlim(0, B_common - 1)
    ax_top_right.set_xlabel("Batch Index", labelpad=4)
    ax_top_right.set_ylabel("Parameter Value")
    ax_top_right.set_title(r"Evolution of $z$ min/max (smoothed)", fontsize=12)

    # ---------- bottom-right: extreme error (log y; x-axis aligns with parameter range) ----------
    ax_bottom_right = fig.add_axes([x0_right, y0_bottom, width_col, height])
    try:
        x_err, err_min, err_max, noise_only, B = compute_extreme_error_series(
            NODE_DIR, PATH_X_OBS, PATH_TRUE_Z, PATH_SHIFT
        )
        D_local = err_min.shape[1]
        ax_bottom_right.set_yscale("log")
        for i in range(D_local):
            ax_bottom_right.plot(x_err, err_max[:, i], "-",  lw=1.5, alpha=0.90, color=RANGE_MAX_WARM)
            ax_bottom_right.plot(x_err, err_min[:, i], "--", lw=1.2, alpha=0.90, color=RANGE_MIN_COOL)

        ax_bottom_right.set_xlabel("Batch Index")
        ax_bottom_right.set_ylabel(r"$|z_{\mathrm{extreme}} - z_{\mathrm{true}}|$")
        ax_bottom_right.set_title("Dataset convergence during training (log)", fontsize=12)

        ax_bottom_right.axhline(noise_only, ls=":", lw=1.4, color="k", alpha=0.85, label=r"Noise level")
        hloc, lloc = ax_bottom_right.get_legend_handles_labels()
        if hloc:
            ax_bottom_right.legend(hloc[-2:], lloc[-2:], loc="upper right", frameon=False)

        ax_bottom_right.set_xlim(0, B_common - 1)
        ymin_pos = min(np.nanmin(err_min[err_min > 0]), np.nanmin(err_max[err_max > 0]))
        ax_bottom_right.set_ylim(max(ymin_pos * 0.5, 1e-8), None)

        for s in ("left", "right", "top", "bottom"):
            ax_bottom_right.spines[s].set_visible(True)
            ax_bottom_right.spines[s].set_linewidth(1.1)
    except Exception as e:
        ax_bottom_right.text(0.5, 0.5, f"extreme-error plot failed:\n{e}",
                             ha="center", va="center", transform=ax_bottom_right.transAxes)

    # ---------- place corner legend in the blank space to the left of the loss plot ----------
    legend_x = x0_left - 0.17
    legend_y = y0_top + height * 0.80
    fig.legend(
        handles=corner_handles,
        loc="center",
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(legend_x, legend_y)
    )

# ---------------- main ----------------
def main():
    torch.set_default_dtype(torch.float64)

    # Network samples
    Z_net_full = load_posterior(PATH_POSTERIOR, DIM)
    if len(Z_net_full) > N_PLOT_MAX:
        idx = torch.randperm(len(Z_net_full))[:N_PLOT_MAX]
        Z_net = Z_net_full[idx]
    else:
        Z_net = Z_net_full
    Z_net_np = Z_net.numpy()

    # Observations and theoretical bimodal reference
    x_obs = torch.from_numpy(np.load(PATH_X_OBS)).to(torch.float64).squeeze()
    assert x_obs.ndim == 1 and x_obs.numel() == DIM
    mu1 = (x_obs - MEAN_SHIFT)
    mu2 = (x_obs + MEAN_SHIFT)
    true_z = np.load(PATH_TRUE_Z).astype(float).squeeze()
    shift  = np.load(PATH_SHIFT).astype(float).squeeze()
    theta_true = true_z  # + shift

    modes  = torch.randint(0, 2, (N_THEORY, DIM))
    mu_mat = torch.where(modes == 0, mu1, mu2).to(torch.float64)
    Z_theory = (mu_mat + torch.randn(N_THEORY, DIM) * SIGMA).numpy()

    # Global scaling
    k, scale = choose_global_scale([Z_net_np, Z_theory, mu1.numpy(), mu2.numpy()])
    s = 1.0 / scale
    Z_net_s    = Z_net_np * s
    Z_theory_s = Z_theory * s
    mu1_s, mu2_s = mu1.numpy() * s, mu2.numpy() * s
    theta_true_s = theta_true * s

    # Corner: Reference → Network
    fig = corner.corner(
        Z_theory_s,
        color=COLOR_REF,
        plot_datapoints=True, markersize=1.8, alpha=0.22,
        fill_contours=False, plot_contours=True, levels=[0.5, 0.8, 0.95],
        contour_kwargs=dict(linewidths=1.2),
        bins=35, smooth=0.9,
        hist_kwargs=dict(density=True, histtype="step", linewidth=1.5),
        labels=None, max_n_ticks=2, use_math_text=True, quiet=True,
    )
    corner.corner(
        Z_net_s,
        color=COLOR_NET,
        plot_datapoints=True, markersize=1.8, alpha=0.22,
        fill_contours=False, plot_contours=True, levels=[0.5, 0.8, 0.95],
        contour_kwargs=dict(linewidths=1.2),
        bins=35, smooth=0.9,
        hist_kwargs=dict(density=True, histtype="step", linewidth=1.5),
        labels=None, max_n_ticks=2, use_math_text=True, fig=fig, quiet=True,
    )

    # Corner decorations
    axes = np.array(fig.axes).reshape((DIM, DIM))
    for i in range(DIM):
        for j in range(DIM):
            ax = axes[i, j]
            if i < j:
                ax.set_visible(False)
                continue
            for sname in ("left", "right", "top", "bottom"):
                ax.spines[sname].set_visible(True)
                ax.spines[sname].set_linewidth(1.1)
            ax.tick_params(which="both", direction="in", pad=2)
            show_x = (i == DIM - 1)
            show_y = (j == 0)
            if not show_x:
                ax.set_xticks([]); ax.set_xticklabels([])
            if not (show_y and (i != j)):
                ax.set_yticks([]); ax.set_yticklabels([])
            if i == j:
                ax.axvline(theta_true_s[i], ls="--", lw=1.0, c="black", alpha=0.95, zorder=5)
            elif i > j:
                ax.axvline(theta_true_s[j], ls="--", lw=1.0, c="black", alpha=0.9, zorder=5)
                ax.axhline(theta_true_s[i], ls="--", lw=1.0, c="black", alpha=0.9, zorder=5)
                ax.plot(theta_true_s[j], theta_true_s[i],
                        "o", ms=3.2, mec="black", mfc="black", alpha=0.95, zorder=6)

    # Legend for the corner plot (to be placed in the blank space left of the loss panel)
    corner_handles = [
        Patch(edgecolor=COLOR_REF, facecolor="none", label="Reference Samples", lw=1.8),
        Patch(edgecolor=COLOR_NET, facecolor="none", label="Learned Posterior", lw=1.8),
    ]

    # Scaling annotation
    if k != 0:
        fig.text(0.03, 0.965, rf"$\times 10^{{-{k}}}$",
                 ha="left", va="top", fontsize=12)

    # Corner layout
    fig.subplots_adjust(top=0.94, left=0.08, right=0.98, bottom=0.07,
                        hspace=0.09, wspace=0.09)
    base = 1.9
    fig.set_size_inches(base*DIM*0.6, base*DIM*0.6)

    # Three panels on the right + place the corner legend in the blank space left of the loss panel
    add_side_panels_and_corner_legend(fig, axes, NODE_DIR, corner_handles)

    # JSD (print)
    jsd = compute_js_divergence(Z_net_s, Z_theory_s, 10)
    print(f"JS Divergence (Network vs Theory): {jsd}")

    # Save
    out_dir = os.path.join(DIR_OUT, "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "posterior_corner_with_loss_and_theta_new.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
