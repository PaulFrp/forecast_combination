# Timestral

Zero-shot SPY realized variance forecasts from `mistralai/Timestral-28M`, checkpoint
`49b26409ac87b5935539a5d8ffe7f397b720db47`. The model-specific requirements pin
the inference code and PyTorch version. The run used Python 3.14.5 and CPU inference
with four PyTorch threads and batches of 16 independent forecast origins.

## Method

- Use the prepared Alpaca snapshot in `data/alpaca/`, with one step per SPY trading
  session. Take the last 512 sessions through each origin at the 17:00 New York
  cutoff, carrying history across phase boundaries. `rv` uses `spy_rv` alone.
  `rv-vix` adds the previous trading session's `vix_close` as a past-only covariate.
- The checkpoint stays fixed. Only the input window advances. Missing input values
  stay masked and the trading calendar stays intact. The runner reserves a blank
  prediction for a missing target history or a nonfinite or nonpositive forecast
  and prints the affected origin and horizon.
- Predict 22 daily variances in one call, with Timestral's default symmetric
  averaging and nonnegative clipping. Integrate the daily quantile curve using
  linear interpolation between its 21 quantiles and constant tails below 0.01 and
  above 0.99. This approximates the mean and caps the modeled upper tail at the
  99th percentile. Average the first 1, 5, and 22 daily means. Inputs and outputs
  use unannualized variance units throughout.
- Compare both variants by the equally weighted mean of the three validation
  RMSE ratios to persistence, using common dates for both variants and persistence.
  Use the lower ratio to select one variant for the final test.

This is a retrospective checkpoint evaluation. The final scheduled test origin
was October 14, 2026. Mistral released Timestral on October 15, 2026 and describes
real-data post-training. Exposure to this SPY series and its held-out outcomes is
unverified. [Timestral technical report](https://arxiv.org/abs/2610.37585)

## Reproduce validation

Run from `forecast_combination/` in a project virtual environment. The first run
downloads the pinned checkpoint to the Hugging Face cache. Use `--checkpoint`
to point to a local copy of that checkpoint.

```sh
python -m pip install -r requirements.txt -r Mistral/requirements.txt
python -m Mistral.run --phase validation --variant rv
python -m Mistral.run --phase validation --variant rv-vix
python -m pipeline.report --schedule data/alpaca/validation/forecast_schedule.csv \
  --actuals data/alpaca/validation/actuals.csv \
  --forecasts forecasts/validation/timestral_28m_rv.csv \
    forecasts/validation/timestral_28m_rv_vix.csv \
  --selected-id timestral_28m_rv_vix --output reports/timestral_validation
```

Lagged VIX reduced validation RMSE at all three horizons. Its mean RMSE ratio
to persistence was 0.9430 across horizons, versus 0.9551 for realized variance alone.

## Reproduce test

Use the selected `timestral_28m_rv_vix` submission ID for both phases when
estimating combination weights and evaluating the combined forecasts.

```sh
python -m Mistral.run --phase test --variant rv-vix
python -m pipeline.report --schedule data/alpaca/test/forecast_schedule.csv \
  --actuals data/alpaca/test/actuals.csv \
  --forecasts forecasts/test/timestral_28m_rv_vix.csv \
  --selected-id timestral_28m_rv_vix --output reports/timestral_test
```

Both validation variants returned predictions for all 481 origins per horizon.
The selected test variant returned predictions for all 949 origins per horizon.
See the [validation report](../reports/timestral_validation/report.md) and
[test report](../reports/timestral_test/report.md) for RMSE and coverage.

## Compare Timestral-209M with Timestral-28M

Use `mistralai/Timestral-209M` at checkpoint
`9874d252ab0aae16f7f7c999f98f98d6ad7d8365`. Both sizes use the same lagged VIX
inputs, 512-session windows, quantile-to-mean conversion, and CPU settings. Score
both sizes and persistence on the same origins per horizon. This follow-up uses
the existing test period, which was already scored for the 28M model.

With the 28M submissions above in place, run:

```sh
for phase in validation test; do
  python -m Mistral.run --phase "$phase" --model 209m --variant rv-vix
  python -m pipeline.report --schedule "data/alpaca/$phase/forecast_schedule.csv" \
    --actuals "data/alpaca/$phase/actuals.csv" \
    --forecasts "forecasts/$phase/timestral_28m_rv_vix.csv" \
      "forecasts/$phase/timestral_209m_rv_vix.csv" \
    --selected-id timestral_209m_rv_vix --output "reports/timestral_209m_$phase"
done
```

Both sizes produced forecasts for all 481 validation origins and 949 test
origins per horizon. Relative to 28M, 209M increased validation RMSE by 1.0%,
1.3%, and 2.4% at 1, 5, and 22 sessions. On test, 209M lowered 1-session RMSE
by 1.7% and increased 5- and 22-session RMSE by 1.2% and 10.7%. The validation
criterion favors keeping 28M as the main Timestral submission.
[Validation comparison](../reports/timestral_209m_validation/report.md)
· [Test comparison](../reports/timestral_209m_test/report.md)

## Compare lookback lengths for Timestral-209M

Keep the 209M checkpoint, lagged VIX inputs, and mean conversion fixed. Compare
256, 512, and 1,024 trading sessions and expanding history on the validation
origins. Expanding history uses every available session through the origin,
up to the checkpoint context limit. In this split it grows from 1,260 to 1,740
sessions. Rank lookbacks by the equally weighted mean of their three validation
RMSE ratios to persistence. Keep the existing 28M run as a comparison.

The benchmark uses the original 512-session CSVs and writes the exploratory
submissions to the report directory.

```sh
for lookback in 256 1024 all; do
  python -m Mistral.run --phase validation --model 209m --variant rv-vix \
    --lookback "$lookback" --output-root reports/timestral_209m_lookback/forecasts
done
```

### Add SPY OHLCV

`rv-vix-ohlcv` adds five raw numeric past covariates to lagged VIX: daily SPY
open, high, low, close, and volume. They come from regular-session Alpaca bars
through the origin and are available at the 17:00 New York cutoff. The target
remains daily `spy_rv`. VIX stays lagged by one trading session. The model masks
blank values and receives covariates only through the origin.

Use the 256-session configuration for a matched comparison with the original
RV and VIX inputs.

```sh
python -m Mistral.run --phase validation --model 209m --variant rv-vix-ohlcv \
  --lookback 256 --output-root reports/timestral_209m_lookback/forecasts
```

On validation, 256 sessions had the lowest mean RMSE ratio among the lookbacks
with RV and lagged VIX (0.9350). Adding OHLCV lowered this ratio to 0.9079.
The two 256-session setups were fixed before scoring them on the existing test
dates. Reproduce the matched comparison with:

```sh
for variant in rv-vix rv-vix-ohlcv; do
  python -m Mistral.run --phase test --model 209m --variant "$variant" \
    --lookback 256 --output-root reports/timestral_209m_lookback/forecasts
done
for phase in validation test; do
  python -m pipeline.report --schedule "data/alpaca/$phase/forecast_schedule.csv" \
    --actuals "data/alpaca/$phase/actuals.csv" \
    --forecasts "forecasts/$phase/timestral_28m_rv_vix.csv" \
      "forecasts/$phase/timestral_209m_rv_vix.csv" \
      reports/timestral_209m_lookback/forecasts/"$phase"/*.csv \
    --selected-id timestral_209m_rv_vix_ohlcv_lb_256 \
    --output "reports/timestral_209m_lookback/$phase"
done
```

Adding OHLCV lowered test RMSE by 1.6%, 7.4%, and 22.7% at 1, 5, and
22 sessions versus the matched 256-session run. It also beat persistence
at all three horizons. Every comparison used 481 validation and 949 test
origins per horizon. [Validation results](../reports/timestral_209m_lookback/validation/report.md)
· [Test results](../reports/timestral_209m_lookback/test/report.md)
