"""Score model CSVs on common origins and plot a selected forecast."""

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import pandas as pd

from .csv_io import ACTUALS, FORECAST, HORIZONS, KEY, load


def join_forecasts(schedule_path: Path, actuals_path: Path,
                   forecast_paths: list[Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    schedule = load(schedule_path, KEY)
    if schedule.empty:
        raise ValueError(f"{schedule_path}: forecast schedule is empty")
    actuals = load(actuals_path, ACTUALS, schedule=schedule)
    data = schedule.merge(actuals, on=list(KEY), validate="one_to_one")
    method_columns = {}
    seen = {}
    for path in forecast_paths:
        forecast = load(path, FORECAST, schedule=schedule)
        model_id = forecast["model_id"].iloc[0]
        if model_id in seen:
            raise ValueError(f"{path}: model_id {model_id!r} already used in {seen[model_id]}")
        seen[model_id] = path
        column = f"prediction_{len(method_columns)}"
        method_columns[model_id] = column
        data = data.merge(forecast.drop(columns="model_id").rename(columns={"prediction": column}),
                          on=list(KEY), validate="one_to_one")
    return data, method_columns


def score(data: pd.DataFrame, method_columns: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = [(model_id, "forecast", column) for model_id, column in sorted(method_columns.items())]
    methods.append(("Baseline", "baseline", "baseline"))
    scores = [{"method": name, "kind": kind} for name, kind, _ in methods]
    coverage = []
    required = ["actual", *(column for _, _, column in methods)]
    for horizon in HORIZONS:
        scheduled = data.loc[data["horizon"].eq(horizon)]
        common = scheduled.dropna(subset=required)
        for row, (name, kind, column) in zip(scores, methods):
            errors = (common[column] - common["actual"]).abs()
            rmse = math.nan
            if not errors.empty:
                scale = errors.max()
                rmse = scale * math.sqrt(((errors / scale) ** 2).mean()) if scale else 0.0
            row[f"rmse_{horizon}"] = rmse
            coverage.append({"method": name, "kind": kind, "horizon": horizon, "scheduled": len(scheduled),
                             "scored": len(common), "missing_actuals": scheduled["actual"].isna().sum(),
                             "missing_predictions": scheduled[column].isna().sum()})
    return pd.DataFrame(scores), pd.DataFrame(coverage)


def write_summary(scores: pd.DataFrame, coverage: pd.DataFrame, path: Path) -> None:
    rmse_columns = [f"rmse_{horizon}" for horizon in HORIZONS]
    best = scores[rmse_columns].min()
    lines = ["# Forecast RMSE", "", "RMSE in unannualized variance units on common origins.", "",
             "| Method | 1 session | 5 sessions | 22 sessions |", "|---|---:|---:|---:|"]
    for _, row in scores.iterrows():
        cells = []
        for column in rmse_columns:
            value = row[column]
            rendered = f"{value:.8g}" if pd.notna(value) else "—"
            cells.append(f"**{rendered}**" if value == best[column] else rendered)
        label = str(row["method"]).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        if row["kind"] == "baseline":
            label = "Persistence baseline"
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    counts = coverage.drop_duplicates("horizon")
    lines.extend(["", "## Coverage", "", "| Horizon | Scheduled | Scored | Missing actuals |",
                  "|---:|---:|---:|---:|"])
    for row in counts.itertuples(index=False):
        scored = str(row.scored) if row.scored else "0 (no common origins)"
        lines.append(f"| {row.horizon} | {row.scheduled} | {scored} | {row.missing_actuals} |")
    lines.extend(["", "Missing predictions, including blanks and missing baseline values, are listed in `coverage.csv`."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_forecast(data: pd.DataFrame, selected_id: str, prediction_column: str, output: Path) -> None:
    figure = Figure(figsize=(9, 7), layout="constrained")
    axes = figure.subplots(len(HORIZONS), 1, sharex=True)
    dates = pd.to_datetime(data["origin_date"])
    for axis, horizon in zip(axes, HORIZONS):
        mask = data["horizon"].eq(horizon)
        panel = data.loc[mask]
        panel_dates = dates.loc[mask]
        axis.plot(panel_dates, panel["actual"], color="#272a2e", linewidth=1.5, label="Actual")
        axis.plot(panel_dates, panel[prediction_column], color="#1864ab", linewidth=1.45, label=selected_id)
        axis.plot(panel_dates, panel["baseline"], color="#70757c", linestyle="--", label="Baseline")
        axis.set_title(f"{horizon}-session horizon", loc="left", fontsize=10, fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel("Forecast origin date")
    figure.supylabel("Average daily variance (decimal returns²)")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=3, frameon=False)
    figure.savefig(output / "forecasts.png", dpi=180)
    with mpl.rc_context({"svg.hashsalt": "forecast-combination"}):
        figure.savefig(output / "forecasts.svg", metadata={"Date": None})


def generate_report(schedule: Path, actuals: Path, forecasts: list[Path], selected_id: str,
                    output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data, methods = join_forecasts(schedule, actuals, forecasts)
    if selected_id not in methods:
        raise ValueError(f"selected model_id {selected_id!r} was not supplied; available IDs: {list(methods)}")
    scores, coverage = score(data, methods)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "scores.csv", index=False, encoding="utf-8", lineterminator="\n")
    coverage.to_csv(output / "coverage.csv", index=False, encoding="utf-8", lineterminator="\n")
    write_summary(scores, coverage, output / "report.md")
    plot_forecast(data, selected_id, methods[selected_id], output)
    return scores, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--actuals", type=Path, required=True)
    parser.add_argument("--forecasts", type=Path, nargs="+", required=True)
    parser.add_argument("--selected-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        generate_report(args.schedule, args.actuals, args.forecasts, args.selected_id, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote score table and figures to {args.output}")


if __name__ == "__main__":
    main()
