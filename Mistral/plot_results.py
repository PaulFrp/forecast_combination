"""Plot the selected Timestral test run for the project report."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
import pandas as pd

from pipeline.csv_io import HORIZONS
from pipeline.report import join_forecasts, score

MODEL_ID = "timestral_209m_rv_vix_ohlcv_lb_256"
REPORT_DIR = Path("reports/timestral_209m_lookback")
VARIANCE_SCALE = 10_000
INK = "#17252f"
MUTED = "#52636f"
BLUE = "#1668b1"
GRID = "#e2e8ec"


def make_figure(data: pd.DataFrame, prediction: str, rmse: pd.DataFrame, pooled_rmse: pd.Series) -> Figure:
    dates = pd.to_datetime(data["origin_date"])
    one_session = data.loc[data["horizon"].eq(1), "actual"]
    peak_date = dates.loc[one_session.idxmax()]
    detail_start = peak_date - pd.Timedelta(weeks=4)
    detail_end = peak_date + pd.Timedelta(weeks=4)
    in_detail = dates.between(detail_start, detail_end)
    plot_margin = pd.Timedelta(days=10)
    full_period = (dates.min() - plot_margin, dates.max() + plot_margin)
    series = (
        ("actual", "Actual", INK, "-", 0.63, 3),
        (prediction, "Timestral-209M", BLUE, "-", 0.68, 4),
        ("baseline", "Persistence", "#707b84", "--", 0.58, 2),
    )
    values = data[[column for column, *_ in series]] * VARIANCE_SCALE
    positive = values.where(values.gt(0))
    log_limits = (positive.min().min() * 0.75, positive.max().max() * 1.25)

    figure = Figure(figsize=(10.2, 7), facecolor="white")
    sections = figure.add_gridspec(2, 1, left=0.102, right=0.975, top=0.788, bottom=0.126,
                                  height_ratios=[3, 1], hspace=0.39)
    overview_grid = sections[0].subgridspec(len(HORIZONS), 1, hspace=0.39)
    detail_grid = sections[1].subgridspec(1, len(HORIZONS), wspace=0.28)
    overview_axes = [figure.add_subplot(panel) for panel in overview_grid]
    detail_axes = [figure.add_subplot(panel) for panel in detail_grid]
    left = overview_axes[0].get_position().x0
    right = overview_axes[0].get_position().x1
    figure.text(left, 0.947, "Forecasting SPY realized variance", fontsize=21, weight="bold", color=INK)
    figure.text(left, 0.905, "Test set  ·  256 trading sessions of history  ·  OHLCV and lagged VIX",
                fontsize=10, color=MUTED)
    for first_axis, heading in (
        (overview_axes[0], "Full test period (log scale)"),
        (detail_axes[0], f"{peak_date:%B %Y} zoom (linear scale)"),
    ):
        figure.text(left, first_axis.get_position().y1 + 0.042, heading, fontsize=11, weight="bold", color=INK)
    pooled = pooled_rmse * VARIANCE_SCALE
    figure.text(right, overview_axes[0].get_position().y1 + 0.042,
                f"Pooled RMSE (×10⁻⁴)  ·  Timestral {pooled[MODEL_ID]:.3f}  ·  Persistence {pooled['Baseline']:.3f}",
                ha="right", fontsize=9, weight="bold", color=INK)

    for horizon, overview, detail in zip(HORIZONS, overview_axes, detail_axes):
        panel = data["horizon"].eq(horizon)
        detail_panel = panel & in_detail
        for column, label, color, style, width, order in series:
            overview.plot(dates[panel], positive.loc[panel, column], label=label, color=color,
                          linestyle=style, linewidth=width, zorder=order)
            detail.plot(dates[detail_panel], values.loc[detail_panel, column], color=color,
                        linestyle=style, linewidth=width, zorder=order)
        for axis in (overview, detail):
            axis.set_title(f"{horizon}-session horizon", loc="left", fontsize=10, weight="bold", color=INK, pad=5)
        overview.axvspan(detail_start, detail_end, color="#e7f0f9", linewidth=0, zorder=0)
        overview.set(yscale="log", ylim=log_limits, xlim=full_period)
        overview.yaxis.set_minor_locator(NullLocator())
        metric = f"rmse_{horizon}"
        model_rmse = rmse.loc[MODEL_ID, metric]
        change = 1 - model_rmse / rmse.loc["Baseline", metric]
        direction = "lower" if change >= 0 else "higher"
        overview.set_title(
            f"RMSE {model_rmse * VARIANCE_SCALE:.3f} × 10⁻⁴  ·  {abs(change):.1%} {direction} than persistence",
            loc="right", fontsize=8.5, color=BLUE, pad=5,
        )
        detail.set(xlim=(detail_start, detail_end), ylim=(0, values.loc[detail_panel].max().max() * 1.10))
        detail.yaxis.set_major_locator(MaxNLocator(nbins=3))
        detail.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        detail.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    for axis in (*overview_axes, *detail_axes):
        axis.set_axisbelow(True)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set_color("#c2cdd4")
        axis.grid(axis="y", color=GRID, linewidth=0.55)
        axis.tick_params(axis="both", colors=MUTED, labelsize=8, length=0, pad=4)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    for axis in overview_axes[:-1]:
        axis.tick_params(axis="x", labelbottom=False)

    overview_axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    overview_axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    figure.legend(*overview_axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(left, 0.884),
                  ncols=3, frameon=False, borderaxespad=0, fontsize=9, labelcolor=INK)
    figure.supylabel("Average daily realized variance (×10⁻⁴)", x=0.025, y=0.458, fontsize=10, color=INK)
    figure.supxlabel("Forecast origin date", y=0.073, fontsize=10, color=INK)
    figure.text(left, 0.018,
                "Shaded dates appear in the zoom. An axis value of 1 represents variance of 0.0001.\n"
                "RMSE uses variance before display scaling. Pooled RMSE combines errors across all three horizons.",
                fontsize=8, color=MUTED)
    return figure


def main() -> None:
    data_dir = Path("data/alpaca/test")
    forecast = REPORT_DIR / "forecasts" / "test" / f"{MODEL_ID}.csv"
    data, columns = join_forecasts(data_dir / "forecast_schedule.csv", data_dir / "actuals.csv", [forecast])
    scores, coverage = score(data, columns)
    rmse = scores.set_index("method")[[f"rmse_{horizon}" for horizon in HORIZONS]]
    scored = coverage.loc[coverage["method"].eq(MODEL_ID)].set_index("horizon")["scored"]
    counts = scored.reindex(HORIZONS).to_numpy()
    pooled_rmse = rmse.pow(2).mul(counts).sum(axis=1).div(counts.sum()).pow(0.5)
    output = REPORT_DIR / "test" / "forecasts_report"
    with mpl.rc_context({"svg.hashsalt": MODEL_ID}):
        figure = make_figure(data, columns[MODEL_ID], rmse, pooled_rmse)
        figure.savefig(output.with_suffix(".png"), dpi=260)
        figure.savefig(output.with_suffix(".svg"), metadata={
            "Date": None,
            "Title": "Forecasting SPY realized variance",
            "Description": "Three full-period log-scale test panels above one row of three linear-scale April 2025 zooms.",
        })


if __name__ == "__main__":
    main()
