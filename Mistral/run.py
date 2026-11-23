"""Generate zero-shot Timestral forecasts for the shared SPY backtest."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.csv_io import DAILY, FORECAST, HORIZONS, KEY, export, load

VARIANTS = {name: name.replace("-", "_") for name in ("rv", "rv-vix", "rv-vix-ohlcv")}
MODEL_REVISIONS = {
    "28m": "49b26409ac87b5935539a5d8ffe7f397b720db47",
    "209m": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
}
LOOKBACK = 512
BATCH_SIZE = 16
PHASES = ("train", "validation", "test")
OHLCV = ("spy_open", "spy_high", "spy_low", "spy_close", "spy_volume")


def quantile_mean(quantiles: np.ndarray, levels: tuple[float, ...]) -> np.ndarray:
    """Integrate a piecewise-linear quantile curve with flat endpoint tails."""
    probabilities = np.concatenate(([0.0], levels, [1.0]))
    values = np.concatenate((quantiles[..., :1], quantiles, quantiles[..., -1:]), axis=-1)
    with np.errstate(invalid="ignore"):
        return (0.5 * (values[..., :-1] + values[..., 1:])) @ np.diff(probabilities)


def parse_lookback(value: str) -> int | None:
    if value == "all":
        return None
    if value.isdecimal() and int(value) > 0:
        return int(value)
    raise argparse.ArgumentTypeError("lookback must be a positive number of sessions or 'all'")


def generate(
    phase: str,
    variant: str,
    checkpoint: Path | None,
    data_root: Path,
    output_root: Path,
    *,
    model_size: str = "28m",
    lookback: int | None = LOOKBACK,
) -> None:
    import torch
    from timestral import TimestralForecaster

    phase_dir = data_root / phase
    schedule = load(phase_dir / "forecast_schedule.csv", KEY)
    if schedule.empty:
        raise ValueError(f"{phase_dir}: forecast schedule is empty")

    past_phases = PHASES[: PHASES.index(phase) + 1]
    daily = pd.concat(
        [load(data_root / split / "daily_inputs.csv", DAILY) for split in past_phases],
        ignore_index=True,
    ).set_index("date").sort_index()
    if not daily.index.is_unique:
        raise ValueError(f"{data_root}: daily inputs contain duplicate dates across phases")
    origins = schedule["origin_date"].drop_duplicates().to_numpy()
    positions = daily.index.get_indexer(origins)
    if (positions < 0).any():
        missing = origins[positions < 0][0]
        raise ValueError(f"{phase_dir}: forecast origin {missing} is missing from daily inputs")

    rv = daily["spy_rv"].to_numpy(dtype=np.float32)
    covariates = None
    if variant != "rv":
        past_inputs = daily[["vix_close"]].shift(1)
        if variant == "rv-vix-ohlcv":
            past_inputs = past_inputs.join(daily[list(OHLCV)])
        covariates = past_inputs.to_numpy(dtype=np.float32)
    torch.set_num_threads(4)
    model_repo = f"mistralai/Timestral-{model_size.upper()}"
    forecaster = TimestralForecaster.from_pretrained(
        checkpoint or model_repo, device="cpu", revision=MODEL_REVISIONS[model_size], token=False
    )
    context_limit = forecaster.max_context
    if lookback is not None:
        context_limit = min(lookback, context_limit)
    daily_means = np.full((len(origins), max(HORIZONS)), np.nan)
    for start in range(0, len(origins), BATCH_SIZE):
        batch = slice(start, start + BATCH_SIZE)
        window = min(context_limit, positions[batch].max() + 1)
        offsets = np.arange(window - 1, -1, -1)
        indices = positions[batch, None] - offsets
        valid = indices >= 0
        safe_indices = indices.clip(min=0)
        context = np.where(valid, rv[safe_indices], np.nan)
        observed = np.isfinite(context).any(axis=1)
        if observed.any():
            past_covariates = None
            if covariates is not None:
                covariate_context = np.where(valid[..., None], covariates[safe_indices], np.nan)
                past_covariates = covariate_context[observed].transpose(0, 2, 1)
            forecast = forecaster.forecast(
                context[observed], prediction_length=max(HORIZONS), past_covariates=past_covariates
            )
            quantiles = forecast.quantiles[:, 0].numpy().astype(np.float64)
            means = quantile_mean(quantiles, forecast.quantile_levels)
            daily_means[np.flatnonzero(observed) + start] = means

    by_horizon = pd.DataFrame(
        {horizon: daily_means[:, :horizon].mean(axis=1) for horizon in HORIZONS},
        index=pd.Index(origins, name="origin_date"),
    )
    predictions = by_horizon.reset_index().melt(
        id_vars="origin_date", var_name="horizon", value_name="prediction"
    )
    model_id = f"timestral_{model_size}_{VARIANTS[variant]}"
    if lookback != LOOKBACK:
        model_id += f"_lb_{lookback if lookback is not None else 'all'}"
    submission = schedule.merge(predictions, on=list(KEY), how="left", validate="one_to_one")
    submission.insert(0, "model_id", model_id)
    failed = ~np.isfinite(submission["prediction"]) | submission["prediction"].le(0)
    submission.loc[failed, "prediction"] = np.nan
    for origin, horizon in submission.loc[failed, list(KEY)].itertuples(index=False, name=None):
        print(f"{phase} {origin} h={horizon}: missing context or invalid forecast", file=sys.stderr)

    output = output_root / phase / f"{model_id}.csv"
    export(submission[list(FORECAST)], output, FORECAST, schedule=schedule)
    print(f"Wrote {len(submission)} forecasts to {output} ({failed.sum()} blank)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES[1:], required=True)
    parser.add_argument("--model", choices=MODEL_REVISIONS, default="28m", help="Timestral checkpoint size")
    parser.add_argument("--variant", choices=VARIANTS, default="rv")
    parser.add_argument("--lookback", type=parse_lookback, default=LOOKBACK,
                        help="History in trading sessions, or 'all' through each origin (default: 512)")
    parser.add_argument("--checkpoint", type=Path, help="Local copy of the checkpoint selected by --model")
    parser.add_argument("--data-root", type=Path, default=Path("data/alpaca"))
    parser.add_argument("--output-root", type=Path, default=Path("forecasts"))
    args = parser.parse_args()
    generate(
        args.phase, args.variant, args.checkpoint, args.data_root, args.output_root,
        model_size=args.model, lookback=args.lookback,
    )


if __name__ == "__main__":
    main()
