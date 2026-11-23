"""Visualize spike detection and magnitude errors for the selected Timestral run."""

import argparse
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator, PercentFormatter
import pandas as pd

from Mistral.plot_results import BLUE, GRID, INK, MODEL_ID, MUTED, REPORT_DIR, VARIANCE_SCALE
from pipeline.csv_io import ACTUALS, HORIZONS, load
from pipeline.report import join_forecasts

PERSISTENCE = "#65727c"


def precision_recall(events: pd.Series, forecasts: pd.Series) -> tuple[list[float], list[float], float]:
    """Evaluate precision and recall at every distinct forecast threshold."""
    groups = events.groupby(forecasts).agg(["sum", "size"]).sort_index(ascending=False)
    hits = groups["sum"].cumsum()
    precision = hits / groups["size"].cumsum()
    recall = hits / events.sum()
    average_precision = ((groups["sum"] / events.sum()) * precision).sum()
    return [0.0, *recall], [1.0, *precision], average_precision


def make_figure(data: pd.DataFrame, prediction: str, thresholds: pd.Series, percentile: int) -> Figure:
    methods = (
        (prediction, "Timestral-209M", BLUE, "-", "o"),
        ("baseline", "Persistence", PERSISTENCE, "--", "x"),
    )
    columns = ["actual", prediction, "baseline"]
    data = data.dropna(subset=columns)
    events = data["actual"].gt(data["horizon"].map(thresholds))
    spike_values = data.loc[events, columns] * VARIANCE_SCALE
    limits = (spike_values.min().min() * 0.8, spike_values.max().max() * 1.25)

    figure = Figure(figsize=(10.7, 7.3), facecolor="white")
    axes = figure.subplots(2, len(HORIZONS))
    figure.subplots_adjust(left=0.103, right=0.977, top=0.765, bottom=0.144, hspace=0.56, wspace=0.24)
    left = axes[0, 0].get_position().x0
    figure.text(left, 0.949, "Forecasting high SPY variance", fontsize=21, weight="bold", color=INK)
    figure.text(left, 0.908, f"An event exceeds the {percentile}th percentile of validation actuals for its horizon.",
                fontsize=10, color=MUTED)
    figure.legend(
        handles=[Line2D([], [], label=label, color=color, linestyle=style, marker=marker,
                        linewidth=1, markersize=4) for _, label, color, style, marker in methods],
        loc="upper left", bbox_to_anchor=(left, 0.885), ncols=2, frameon=False, borderaxespad=0,
        fontsize=9, labelcolor=INK,
    )
    figure.text(left, 0.817, "Event detection on the test set", fontsize=11, weight="bold", color=INK)
    figure.text(left, axes[1, 0].get_position().y1 + 0.052, "Actual vs. forecast on high-variance windows",
                fontsize=11, weight="bold", color=INK)

    for column_index, horizon in enumerate(HORIZONS):
        detection, magnitude = axes[:, column_index]
        panel = data["horizon"].eq(horizon)
        spikes = events & panel
        count = spikes.sum()
        session_label = "session" if horizon == 1 else "sessions"
        detection.set_title(f"{horizon} {session_label}  ·  {count} events", loc="left",
                            fontsize=10, weight="bold", color=INK, pad=7)
        average_precisions = []
        for method, _, color, style, marker in methods:
            recall, precision, ap = precision_recall(events.loc[panel], data.loc[panel, method])
            average_precisions.append(ap)
            detection.step(recall, precision, where="pre", color=color, linestyle=style, linewidth=1.15)
            magnitude.scatter(spike_values.loc[spikes, "actual"], spike_values.loc[spikes, method],
                              color=color, marker=marker, s=19, linewidths=0.85, alpha=0.86)
        detection.text(0.96, 0.94, f"AP {average_precisions[0]:.2f} vs. {average_precisions[1]:.2f}",
                       transform=detection.transAxes, ha="right", va="top", color=INK, fontsize=8.5,
                       bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
        detection.set(xlim=(0, 1), ylim=(0, 1.02), xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[0, 0.5, 1])
        detection.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        detection.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        detection.set_xlabel("Recall", fontsize=8.5, color=MUTED, labelpad=3)

        magnitude.plot(limits, limits, color="#9aa8b2", linestyle=":", linewidth=1, zorder=0)
        magnitude.set(xscale="log", yscale="log", xlim=limits, ylim=limits)
        for axis in (magnitude.xaxis, magnitude.yaxis):
            axis.set_ticks([0.5, 1, 2, 5, 10, 20, 50])
            axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.set_minor_locator(NullLocator())
        errors = spike_values.loc[spikes, [prediction, "baseline"]].sub(spike_values.loc[spikes, "actual"], axis=0)
        rmse = errors.pow(2).mean().pow(0.5)
        change = 1 - rmse[prediction] / rmse["baseline"]
        direction = "lower" if change >= 0 else "higher"
        magnitude.set_title(f"Event RMSE {abs(change):.1%} {direction}", loc="left", fontsize=9,
                            weight="bold", color=BLUE if change >= 0 else INK, pad=7)
        if horizon == 1:
            peak = spike_values.loc[spikes].sort_values("actual").iloc[-1]
            magnitude.annotate("Largest spike", xy=(peak["actual"], peak[prediction]), xytext=(-12, -10),
                               textcoords="offset points", ha="right", va="top", fontsize=8, color=INK,
                               arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.7})

    for axis in axes.flat:
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#c2cdd4")
        axis.grid(color=GRID, linewidth=0.55)
        axis.tick_params(axis="both", colors=MUTED, labelsize=8, length=0, pad=4)
    axes[0, 0].set_ylabel("Precision", fontsize=9, color=MUTED)
    axes[1, 0].set_ylabel("Forecast variance\n(×10⁻⁴, log scale)", fontsize=9, color=MUTED)
    figure.text(0.54, 0.087, "Actual variance (×10⁻⁴, log scale)", ha="center", fontsize=9, color=MUTED)
    figure.text(left, 0.028,
                "AP means average precision, with Timestral listed first. Event RMSE changes are relative to persistence.\n"
                "Points below the diagonal show forecasts that underestimated realized variance.",
                fontsize=8, color=MUTED)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--percentile", type=int, choices=(85, 90, 95), default=95)
    percentile = parser.parse_args().percentile
    data_dir = Path("data/alpaca")
    validation = load(data_dir / "validation" / "actuals.csv", ACTUALS)
    thresholds = validation.groupby("horizon")["actual"].quantile(percentile / 100)
    test_dir = data_dir / "test"
    forecast = REPORT_DIR / "forecasts" / "test" / f"{MODEL_ID}.csv"
    data, columns = join_forecasts(test_dir / "forecast_schedule.csv", test_dir / "actuals.csv", [forecast])
    filename = "spike_diagnostics" if percentile == 95 else f"spike_diagnostics_p{percentile}"
    output = REPORT_DIR / "test" / filename
    with mpl.rc_context({"svg.hashsalt": f"{MODEL_ID}-p{percentile}"}):
        figure = make_figure(data, columns[MODEL_ID], thresholds, percentile)
        figure.savefig(output.with_suffix(".png"), dpi=260)
        figure.savefig(output.with_suffix(".svg"), metadata={
            "Date": None,
            "Title": f"Forecasting high SPY variance at the {percentile}th percentile cutoff",
            "Description": "Precision-recall curves and actual-versus-forecast scatter plots for three forecast horizons.",
        })


if __name__ == "__main__":
    main()
