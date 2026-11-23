"""Static figures for the slides: test forecasts vs realised, and feature importance."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COLORS = {"LSTM": "#2a78d6", "HAR": "#eb6834", "realized": "#9b9a94", "crisis": "#f0efec"}

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})


def _shade(ax, crisis):
    """Shade contiguous runs of crisis days."""
    start = None
    for d, flag in crisis.items():
        if flag and start is None:
            start = d
        elif not flag and start is not None:
            ax.axvspan(start, d, color=COLORS["crisis"], lw=0, zorder=0)
            start = None
    if start is not None:
        ax.axvspan(start, crisis.index[-1], color=COLORS["crisis"], lw=0, zorder=0)


def forecasts_figure(results, path):
    hs = list(results)
    fig, axes = plt.subplots(len(hs), 1, figsize=(10, 2.8 * len(hs)), sharex=True, squeeze=False)
    for ax, h in zip(axes[:, 0], hs):
        d = results[h]["detail"]
        _shade(ax, d["crisis"])
        ax.plot(d.index, d["realized"], color=COLORS["realized"], lw=0.8, label="Realised RV")
        ax.plot(d.index, d["HAR"], color=COLORS["HAR"], lw=1.5, label="HAR")
        ax.plot(d.index, d["LSTM"], color=COLORS["LSTM"], lw=1.5, label="LSTM")
        ax.set_yscale("log")
        ax.set_title(f"h = {h} day{'s' if h > 1 else ''}: average daily realized variance, test origins",
                     loc="left", color=INK, fontsize=10)
        ax.set_ylabel("variance (log scale)")
    axes[0, 0].legend(loc="upper right", frameon=False, ncol=3)
    fig.text(0.01, 0.005, "Shaded: ex-ante high-volatility regime (22-session RV at the origin "
             "above its 80th training percentile)", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def importance_figure(results, path):
    hs = [h for h in results if results[h]["importance"] is not None]
    if not hs:
        return
    fig, axes = plt.subplots(1, len(hs), figsize=(4 * len(hs), 3.2), squeeze=False)
    for ax, h in zip(axes[0], hs):
        imp = results[h]["importance"].sort_values("relative_increase")
        ax.barh(imp["feature"], 100 * imp["relative_increase"], color=COLORS["LSTM"], height=0.6)
        ax.set_title(f"h = {h}", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("validation MSE increase when shuffled (%)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("LSTM permutation feature importance (validation set)", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(results, out_dir):
    forecasts_figure(results, out_dir / "test_forecasts.png")
    importance_figure(results, out_dir / "feature_importance.png")
