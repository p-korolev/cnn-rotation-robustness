import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

DARK_BG   = "#ffffff"
PANEL_BG  = "#ffffff"
BORDER    = "#a5a5a5"
TEXT_COL  = "#1f1f1f"
GRID_COL  = "#dadada"
SO2_PAL = ["#312e81", "#6d28d9", "#a855f7", "#e879f9", "#fce7f3"]
C4_PAL  = ["#064e3b", "#047857", "#10b981", "#34d399", "#bbf7d0"]
C8_PAL  = ["#1e3a5f", "#1d4ed8", "#3b82f6", "#7dd3fc", "#e0f2fe"]
BASE_COL = "#f59e0b"
M_VALUES  = [1, 2, 4, 8, 16]
CK_ORDERS = [4, 8] 
GROUP_STYLES = {
    "so2": ("SO(2)",  SO2_PAL, "-"),
    "c4":  ("$C_4$",  C4_PAL,  "-."),
    "c8":  ("$C_8$",  C8_PAL,  ":"),
}

def _apply_dark_theme() -> None:
    mpl.rcParams.update({
        "figure.facecolor"    : DARK_BG,
        "axes.facecolor"      : PANEL_BG,
        "axes.edgecolor"      : BORDER,
        "axes.labelcolor"     : TEXT_COL,
        "xtick.color"         : TEXT_COL,
        "ytick.color"         : TEXT_COL,
        "text.color"          : TEXT_COL,
        "grid.color"          : GRID_COL,
        "grid.linestyle"      : "--",
        "grid.alpha"          : 0.35,
        "axes.grid"           : True,
        "font.size"           : 11,
        "axes.titlesize"      : 13,
        "axes.labelsize"      : 12,
        "legend.framealpha"   : 0.18,
        "legend.edgecolor"    : BORDER,
        "legend.facecolor"    : PANEL_BG,
        "axes.spines.top"     : False,
        "axes.spines.right"   : False,
        "axes.spines.left"    : True,
        "axes.spines.bottom"  : True,
    })

def _spine_colour(ax, colour: str = BORDER) -> None:
    for sp in ax.spines.values():
        sp.set_edgecolor(colour)

def _log2_xticks(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(M_VALUES)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Sampling density  $m$")


def plot_accuracy_vs_m(results: dict, save_path: str = "plot_acc_vs_m.pdf"):
    _apply_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor(DARK_BG)

    markers = {"so2": "o", "c4": "D", "c8": "s"}

    for col, (metric, ylabel, panel) in enumerate([
        ("final_acc", "Test accuracy (randomly rotated)", "(a)"),
        ("mean_robustness", "Mean rotation robustness", "(b)"),
    ]):
        ax = axes[col]

        # baseline 
        if "baseline" in results:
            base_val = results["baseline"][metric]
            ax.axhline(base_val, color=BASE_COL, lw=1.8, ls="--",
                       alpha=0.85, label="No augmentation")
        else:
            base_val = 0.0

        anchor_vals = None

        # each curve per group
        for prefix, (gname, palette, ls) in GROUP_STYLES.items():
            vals = []
            for i, m in enumerate(M_VALUES):
                key = f"{prefix}_m{m}"
                if key not in results:
                    continue
                v = results[key][metric]
                vals.append((m, v, i))

            if not vals:
                continue

            ms_list = [x[0] for x in vals]
            vs_list = [x[1] for x in vals]
            ax.plot(ms_list, vs_list, color=palette[2], lw=2.0,
                    ls=ls, zorder=3, label=f"{gname}")
            for m, v, i in vals:
                ax.scatter(m, v, color=palette[i], s=85, zorder=5,
                           marker=markers[prefix],
                           edgecolors=TEXT_COL, linewidths=0.5)

            if prefix == "so2":
                anchor_vals = vals

        # constant anchor guide
        if anchor_vals and len(anchor_vals) > 1:
            ms_arr = np.array([x[0] for x in anchor_vals], dtype=float)
            a0     = anchor_vals[0][1] - base_val
            theory = base_val + a0 / np.sqrt(ms_arr / ms_arr[0])
            ax.plot(ms_arr, theory, color="#6b7280", lw=1.1, ls=":",
                    alpha=0.6, label="$\\mathcal{O}(1/\\sqrt{m})$ guide", zorder=2)

        _log2_xticks(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel} {ylabel} vs. Sampling Density")
        ax.legend(fontsize=9)
        _spine_colour(ax)

    fig.tight_layout(pad=2.5)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f" ✓ {save_path}")
    return fig, axes

def plot_polar_robustness(results: dict, save_path: str = "plot_polar_robustness.pdf"):
    _apply_dark_theme()

    conditions = [
        ("baseline", "No augmentation",  BASE_COL,   "-",   2.2),
        ("so2_m1",   "SO(2)  $m=1$",    SO2_PAL[0], "-",   1.5),
        ("so2_m4",   "SO(2)  $m=4$",    SO2_PAL[2], "-",   1.8),
        ("so2_m16",  "SO(2)  $m=16$",   SO2_PAL[4], "-",   2.2),
        ("c4_m4",    "$C_4$  $m=4$",    C4_PAL[2],  "-.",  1.8),
        ("c8_m4",    "$C_8$  $m=4$",    C8_PAL[2],  ":",   1.8),
        ("c8_m8",    "$C_8$  $m=8$",    C8_PAL[3],  ":",   2.0),
    ]

    fig  = plt.figure(figsize=(8, 8), facecolor=DARK_BG)
    ax   = fig.add_subplot(111, projection="polar", facecolor=PANEL_BG)

    legend_handles = []
    for key, label, colour, ls, lw in conditions:
        if key not in results:
            continue
        deg = results[key]["angles_deg"]
        rob = results[key]["rot_robustness"]
        rad = np.deg2rad(deg)
        rad_c = np.append(rad, rad[0])
        rob_c = np.append(rob, rob[0])
        ax.plot(rad_c, rob_c, color=colour, lw=lw, ls=ls, alpha=0.90, zorder=3)
        ax.fill(rad_c, rob_c, color=colour, alpha=0.06)
        legend_handles.append(
            Line2D([0], [0], color=colour, lw=lw, ls=ls, label=label)
        )

    # polar axes style
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(40)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x*100)}%")
    )
    ax.set_thetagrids(
        np.arange(0, 360, 30),
        [f"{d}°" for d in range(0, 360, 30)],
        color=TEXT_COL, fontsize=9,
    )
    ax.tick_params(colors=TEXT_COL)
    ax.spines["polar"].set_color(BORDER)
    ax.grid(color=GRID_COL, linestyle="--", alpha=0.45)

    ax.set_title("Rotation Robustness Across $[0°, 360°)$", color=TEXT_COL, fontsize=14, pad=22)
    ax.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.42, -0.08), fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"✓ {save_path}")
    return fig, ax

def plot_orbit_variance(results: dict, save_path: str = "plot_orbit_variance.pdf"):
    _apply_dark_theme()
    groups   = list(GROUP_STYLES.items())
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5.2), sharey=False)
    fig.patch.set_facecolor(DARK_BG)
    if n_groups == 1:
        axes = [axes]

    for col, (prefix, (gname, palette, _)) in enumerate(groups):
        ax = axes[col]

        # anchor var at m=1
        anchor_var: list = []
        anchor_key = f"{prefix}_m1"
        if anchor_key in results:
            anchor_var = results[anchor_key]["history"]["orbit_var"]

        for i, m in enumerate(M_VALUES):
            key = f"{prefix}_m{m}"
            if key not in results:
                continue
            ovar   = np.array(results[key]["history"]["orbit_var"])
            epochs = np.arange(1, len(ovar) + 1)

            ax.plot(epochs, ovar, color=palette[i], lw=2.0, alpha=0.9,
                    label=f"$m = {m}$", zorder=3)
            ax.fill_between(epochs, 0, ovar, color=palette[i], alpha=0.07)

            # theoretical guide
            if anchor_var and m > 1:
                theory_level = anchor_var[0] / m
                ax.axhline(theory_level, color=palette[i], lw=0.9, ls=":", alpha=0.45, zorder=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Orbit-loss variance")
        panel_ch = chr(ord('a') + col)
        ax.set_title(f"({panel_ch}) {gname} — Orbit-Loss Variance During Training")
        ax.set_xlim(1, None)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        _spine_colour(ax)

    fig.text(0.5, -0.02,
             "Dotted lines: theoretical $\\sigma^2_f / m$ level (anchored at $m=1$)",
             ha="center", fontsize=9, color="#6b7280")

    fig.tight_layout(pad=2.5)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"✓ {save_path}")
    return fig, axes

def plot_ece(results: dict, save_path: str = "plot_ece.pdf"):
    """
    Grouped horizontal bar chart comparing Expected Calibration Error (ECE)
    across all augmentation conditions.
    """
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor(DARK_BG)

    entries  = []
    sep_rows = []

    if "baseline" in results:
        entries.append(("No augmentation", results["baseline"]["ece"], BASE_COL, "base"))

    for prefix, (gname, palette, _) in GROUP_STYLES.items():
        sep_rows.append(len(entries) - 0.5)
        for i, m in enumerate(M_VALUES):
            key = f"{prefix}_m{m}"
            if key in results:
                entries.append((f"{gname}  $m={m}$", results[key]["ece"], palette[i], prefix))

    labels = [e[0] for e in entries]
    eces   = [e[1] for e in entries]
    colors = [e[2] for e in entries]
    y_pos  = np.arange(len(entries))
    fig.set_size_inches(10, max(6.5, len(entries) * 0.45))

    bars = ax.barh(y_pos, eces, color=colors, edgecolor=BORDER, linewidth=0.6, height=0.65, zorder=3)

    for bar, v in zip(bars, eces):
        ax.text(v + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9, color=TEXT_COL)

    for sp in sep_rows:
        ax.axhline(sp, color=BORDER, lw=1.0, ls="--", zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Expected Calibration Error (ECE)")
    ax.set_title("Model Calibration Across Haar Sampling Conditions")
    ax.set_xlim(0, max(eces) * 1.35)
    ax.invert_yaxis()
    _spine_colour(ax)

    # group annotations
    cursor = 1
    for prefix, (gname, _, _) in GROUP_STYLES.items():
        n = sum(1 for e in entries if e[3] == prefix)
        if n:
            mid = cursor + n / 2 - 0.5
            ax.annotate(gname, xy=(1.01, mid / len(entries)),
                        xycoords="axes fraction", fontsize=9, color="#9ca3af",
                        va="center", rotation=270, ha="left")
            cursor += n

    fig.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"✓ {save_path}")
    return fig, ax

def plot_training_curves(results: dict, save_path: str = "plot_training_curves.pdf"):
    _apply_dark_theme()
    groups    = list(GROUP_STYLES.items())   #so2, c4, c8
    n_cols    = len(groups)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 9))
    fig.patch.set_facecolor(DARK_BG)
    base_hist = results.get("baseline", {}).get("history", {})
    n_epochs  = len(base_hist.get("train_loss", []))
    ep        = np.arange(1, n_epochs + 1) if n_epochs else np.arange(1, 16)

    METRICS = [
        ("train_loss", "Training loss"),
        ("test_acc_rotated", "Test accuracy"),
    ]

    for col, (prefix, (gname, palette, _)) in enumerate(groups):
        for row, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row][col]

            # Baseline
            if base_hist:
                bvals_key = metric if metric in base_hist else (
                    "test_acc" if metric == "test_acc_rotated" and "test_acc" in base_hist
                    else None
                )
                if bvals_key:
                    bvals = np.array(base_hist[bvals_key])
                    ax.plot(ep[:len(bvals)], bvals, color=BASE_COL, lw=1.5,
                            ls="--", alpha=0.75, label="No augmentation")

            for i, m in enumerate(M_VALUES):
                key = f"{prefix}_m{m}"
                if key not in results:
                    continue
                hist = results[key]["history"]
                mkey = metric if metric in hist else (
                    "test_acc" if metric == "test_acc_rotated" and "test_acc" in hist
                    else None
                )
                if mkey is None:
                    continue
                vals = np.array(hist[mkey])
                ep_i = np.arange(1, len(vals) + 1)
                ax.plot(ep_i, vals, color=palette[i], lw=1.9, alpha=0.9, label=f"$m={m}$")
                ax.fill_between(ep_i, vals, vals.mean(), color=palette[i], alpha=0.04)

            panel_label = chr(ord("a") + col * 2 + row)
            ax.set_title(f"({panel_label})  {gname}  —  {ylabel}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8, ncol=2)
            ax.set_xlim(1, None)
            _spine_colour(ax)

    fig.suptitle("Training Dynamics — Test accuracy evaluated on randomly rotated images", fontsize=14, color=TEXT_COL, y=1.01)
    fig.tight_layout(pad=2.5)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"✓ {save_path}")
    return fig, axes


def plot_compute_efficiency(results: dict, save_path: str = "plot_compute_efficiency.pdf"):
    _apply_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    markers = {"so2": "o", "c4": "D", "c8": "s"}

    # epoch computation time vs m
    ax = axes[0]
    for prefix, (gname, palette, ls) in GROUP_STYLES.items():
        ms_list, t_list = [], []
        for i, m in enumerate(M_VALUES):
            key = f"{prefix}_m{m}"
            if key not in results:
                continue
            t = results[key].get("mean_epoch_time_s", None)
            if t is None:
                et = results[key]["history"].get("epoch_time_s", [])
                t  = float(np.mean(et)) if et else None
            if t is not None:
                ms_list.append(m)
                t_list.append(t)

        if not ms_list:
            continue

        ax.plot(ms_list, t_list, color=palette[2], lw=2.0, ls=ls,
                zorder=3, label=gname)
        for i, (m, t) in enumerate(zip(ms_list, t_list)):
            ax.scatter(m, t, color=palette[i], s=90, zorder=5,
                       marker=markers[prefix], edgecolors=TEXT_COL, linewidths=0.5)

    # linear reference O(m)
    first_times = []
    for prefix in GROUP_STYLES:
        key = f"{prefix}_m1"
        if key in results:
            t1 = results[key].get("mean_epoch_time_s") or 0
            if t1:
                first_times.append(t1)
    if first_times:
        t1_ref = min(first_times)
        ms_arr = np.array(M_VALUES, dtype=float)
        ax.plot(ms_arr, t1_ref * ms_arr, color="#6b7280", lw=1.0, ls=":", alpha=0.55, label="Linear $O(m)$ guide", zorder=2)

    _log2_xticks(ax)
    ax.set_ylabel("Mean epoch training time (s)")
    ax.set_title("(a)  Epoch Training Time vs. Sampling Density")
    ax.legend(fontsize=9)
    _spine_colour(ax)

    # second panel
    ax2 = axes[1]

    if "baseline" in results:
        b  = results["baseline"]
        bt = b.get("total_train_time_s", sum(b["history"].get("epoch_time_s", [0])))
        ba = b["final_acc"]
        ax2.scatter(bt, ba, color=BASE_COL, s=160, zorder=6,
                    marker="*", edgecolors=TEXT_COL, linewidths=0.7,
                    label="No augmentation")
        ax2.annotate("Base", (bt, ba), textcoords="offset points",
                     xytext=(6, 4), fontsize=8, color=BASE_COL)

    for prefix, (gname, palette, _) in GROUP_STYLES.items():
        for i, m in enumerate(M_VALUES):
            key = f"{prefix}_m{m}"
            if key not in results:
                continue
            r   = results[key]
            tot = r.get("total_train_time_s", sum(r["history"].get("epoch_time_s", [0])))
            acc = r["final_acc"]
            ax2.scatter(tot, acc, color=palette[i], s=120, zorder=5, marker=markers[prefix], edgecolors=TEXT_COL, linewidths=0.5)
            ax2.annotate(
                f"$m={m}$", (tot, acc),
                textcoords="offset points", xytext=(5, 3),
                fontsize=7.5, color=palette[i]
            )

    # legends
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0],[0], marker="*", color="w", markerfacecolor=BASE_COL,
               markersize=10, label="No augmentation"),
    ]
    for prefix, (gname, palette, _) in GROUP_STYLES.items():
        legend_handles.append(
            Line2D([0],[0], marker=markers[prefix], color="w",
                   markerfacecolor=palette[2], markersize=9, label=gname)
        )
    ax2.legend(handles=legend_handles, fontsize=9)

    ax2.set_xlabel("Total training time (s)")
    ax2.set_ylabel("Test accuracy (randomly rotated)")
    ax2.set_title("(b)  Accuracy vs. Total Training Cost")
    _spine_colour(ax2)

    fig.suptitle("Computational Efficiency of Haar Sampling",
                 color=TEXT_COL, fontsize=13, y=1.01)
    fig.tight_layout(pad=2.5)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"✓  {save_path}")
    return fig, axes
