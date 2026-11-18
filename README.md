# Forecast Combination

**DSBA Forecasting and Predictive Analytics · Topic 15**

Can combining forecasts improve predictions of SPY volatility over the next trading day, week, and month? We compare mean point forecasts from econometrics, machine learning, deep learning, and time-series foundation models. Each model owner uses the same trading calendar, target definitions, and forecast origins. The integration teammate can combine and score submissions from CSVs alone.

## Where to work

| Path | What you will find |
|---|---|
| [`LSTM/`](LSTM/) · [`Mistral/`](Mistral/) · [`model_3/`](model_3/) · [`model_4/`](model_4/) | Model-owner workspaces for code, dependencies, and method notes |
| [`weights/`](weights/) | Workspace for combination logic |
| [`pipeline/`](pipeline/) | Shared CSV validation and forecast reporting |
| [`data/alpaca/`](data/alpaca/) | SPY/VIX training, validation, and test CSVs for this project |
| [`data/yahoo/`](data/yahoo/) | Daily S&P 500 index data and scripts from the Yahoo Finance experiment |
| `forecasts/<phase>/` | Model submission CSVs |
| `combined_forecasts/<phase>/` | Combination submissions |
| `reports/` | Generated score tables, coverage, and figures |
| [`examples/synthetic/`](examples/synthetic/) | Small, reproducible example with data, two forecasts, and a report |

### Implement an assigned model

Give your coding agent the assigned folder, model family or specific method, and a unique submission ID. For example, assign `model_3/`, econometrics using HAR-RV, and `model_id=har_rv`. If the assignment specifies a family, document the method chosen within that family.

1. Put the model code and any model-specific dependencies in the assigned folder. Record the exact command to run from the directory containing this README in that folder’s `MODEL.md`.
2. Generate both `forecasts/validation/<model_id>.csv` and `forecasts/test/<model_id>.csv` by following steps 2–5.
3. Run the report command in step 6 for each phase using that phase’s schedule, actuals, and forecast CSV. Check the scored and missing-prediction counts before handoff.

## 1. Set up and run the example

Open a terminal in the directory containing this README. Use Python 3.11–3.14.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Preview the report with the included synthetic CSVs.

```sh
python -m pipeline.report --schedule examples/synthetic/forecast_schedule.csv \
  --actuals examples/synthetic/actuals.csv \
  --forecasts examples/synthetic/forecasts/synthetic_a.csv examples/synthetic/forecasts/synthetic_b.csv \
  --selected-id synthetic_a --output reports/demo
```

Open `reports/demo/report.md` and `reports/demo/forecasts.png` to see the score table and plots. [Example report](examples/synthetic/report/report.md)

## 2. Use the Alpaca splits

Use `data/alpaca/` for all SPY model fitting, forecasting, and evaluation.

| Folder | Session dates | Input sessions | Forecast origins per horizon | Use |
|---|---|---:|---:|---|
| [`train/`](data/alpaca/train/) | 2016-01-04 to 2020-12-31 | 1,259 | — | Fit the initial model |
| [`validation/`](data/alpaca/validation/) | 2021-01-04 to 2022-12-30 | 503 | 481 | Tune models and learn combination weights from walk-forward forecasts |
| [`test/`](data/alpaca/test/) | 2023-01-03 to 2026-11-13 | 971 | 949 | Evaluate the selected models and combination rules |

Each folder contains `daily_inputs.csv` and `actuals.csv`. Validation and test also contain `forecast_schedule.csv`, with one row per requested `origin_date,horizon`. Submit forecasts against the schedule for that phase.

For a model trained directly on forward labels, join `train/daily_inputs.csv` to nonblank labels for your horizon in `train/actuals.csv`, matching `date` to `origin_date`. Training labels are blank where the outcome window extends past the training cutoff. The last 22 input sessions in each evaluation phase provide outcomes for earlier forecast origins. [Data sources and gaps](data/alpaca/data_quality.md)

## 3. Know your inputs and target

Each row is one SPY trading session on the New York calendar.

| Input | Meaning |
|---|---|
| `date` | Session date in New York, in `YYYY-MM-DD` format |
| `spy_open`, `spy_close` | First five-minute bar’s opening price and last bar’s closing price, in USD |
| `spy_high`, `spy_low` | Highest and lowest traded prices across regular-session bars, in USD |
| `spy_volume` | Total regular-session volume in shares |
| `spy_rv` | Realized variance during that session, in squared decimal-return units |
| `vix_close` | Cboe VIX closing level as quoted. It measures option-implied S&P 500 volatility over roughly 30 days, expressed as an annualized percentage level |

Daily realized variance starts with the first bar’s opening price $P_{t,0}$ and uses successive five-minute closes through the actual session close. Sum the squared log returns. A normal session contributes 78 returns, including the opening-to-first-close return. Early closes use the shorter exchange schedule, and overnight changes are excluded.

An empty interior five-minute slot fully covered by a verified market-wide halt contributes zero return and volume. The next observed close still captures the price change across the halt. Daily OHLCV uses observed trades. For the two sessions with feed gaps, complete SPY OHLCV is unavailable and `spy_rv` uses the wider-return estimates described below. [NYSE halt records](https://www.nyse.com/publicdocs/nyse/markets/nyse/Report_of_the_Market-Wide_Circuit_Breaker_Working_Group.pdf) · [Data quality](data/alpaca/data_quality.md)

$$
RV_t = \sum_{j=1}^{m_t}\left[\log(P_{t,j}/P_{t,j-1})\right]^2.
$$

Forecast the **expected average daily realized variance** for the next $h$ trading sessions. The target excludes the origin session.

$$
Y_{t,h}=\frac{1}{h}\sum_{j=1}^{h}RV_{t+j}.
$$

| `horizon` | What `actual` averages |
|---:|---|
| 1 | The next SPY session |
| 5 | The first five subsequent SPY sessions |
| 22 | The first twenty-two subsequent SPY sessions |

For a Friday origin, the one-session target is the next session the exchange actually opens. The five-session target skips any holidays. The **persistence baseline** takes the average of the last $h$ values of `spy_rv` through the origin, including the origin. Keep the full calendar when calculating shifts or rolling features. A missing variance invalidates any target or baseline window that contains it. Missing SPY measures and unavailable labels are blank.

Submit predictions in **unannualized variance units**. For scale, 1% daily volatility corresponds to variance of `0.0001`. If you model log variance, document how you recover the mean in variance units. Exponentiating the predicted mean log variance generally estimates a different quantity.

## 4. Fit your model and forecast in date order

For each forecast origin, follow this sequence.

1. **Set the information cutoff.** The snapshot assumes 17:00 New York time on the origin date. Cboe’s daily history has no as-of timestamps. Confirm that your source’s observations were available by the agreed cutoff before using same-day VIX. If availability cannot be established, use an available lag or omit the feature and document that choice. [Cboe VIX dissemination notice](https://cdn.cboe.com/resources/release_notes/2021/CSMI_VIX_End_Time_Change.pdf)
2. **Build features from the trading calendar through the origin.** Carry prior input rows across split boundaries for lookback. Common starters are current `spy_rv`, a one-session lag, and trailing 5- and 22-session averages. Document the lookback and missing-value rule. Fit scalers, imputers, feature selection, and other learned preprocessing using data available at the cutoff.
3. **Restrict supervised fitting to matured targets.** A historical label with horizon $h$ becomes usable once its $h$-th subsequent SPY session has finished and the observations are available. With `pos` mapping dates to positions in the full trading calendar, require `pos[label_origin] + h <= pos[current_origin]` and a nonblank target. Use this filter on any phase’s `actuals.csv` before refitting. If your refit needs labels near a split boundary or outside a forecast schedule, apply the target formula above to the concatenated `spy_rv` series through the current origin, using complete outcome windows. Keep `actual` and `baseline` out of the feature matrix.
4. **Predict and advance.** Produce a mean forecast for each requested horizon. A direct model can fit one model per horizon or a joint model. A recursive daily variance model can fit to the observed series through the origin, forecast the next $h$ daily variances, and average those forecasts. Generate validation forecasts in origin order. Before the final test, freeze the selected hyperparameters, missing-data rules, and update policy. A rolling or expanding refit during the test can use earlier outcomes as they mature when the policy specifies it.

### Variance estimates for two training gaps

The training data includes `spy_rv` for every session. For the two dates below, the estimate sums squared log returns from the session open to the close, with one wider return spanning the feed gap. Price moves inside that wider interval can reverse or reinforce each other, so the estimate can differ from five-minute realized variance.

- **2016-02-02.** The estimate uses the 10-minute return between the observed Alpaca closes at 11:20 and 11:30 New York time. Its availability date is February 2.
- **2019-08-12.** The estimate uses the 25-minute return from the last Alpaca bar close at 15:35 to the official NYSE Arca close of $288.07 at 16:00. Its conservative availability date is August 14, following Nasdaq’s August 13 update. [NYSE closing prices](https://www.nyse.com/publicdocs/nyse/notifications/market-status/NYSE_Group_Closing_Prices_08122019.xlsx) · [Nasdaq incident update](https://www.nasdaqtrader.com/TraderNews.aspx?id=nfnnews2019-01)

Training target and baseline windows include these estimates where applicable. If you simulate an earlier cutoff within training, use each estimate and any feature or label derived from it only from its availability date onward. Validation and test scoring use measured variance.

## 5. Export and hand off your forecasts

Submit one CSV per model ID for each evaluation phase, plus a short method note. Suggested layout is `forecasts/validation/<model_id>.csv` and `forecasts/test/<model_id>.csv`. Use the same model ID across phases and a unique ID per model or combination within each report.

```csv
model_id,origin_date,horizon,prediction
example_model,2024-06-03,1,0.000100
example_model,2024-06-03,5,0.000120
example_model,2024-06-03,22,0.000140
```

- Use UTF-8, the exact header, ISO dates, integer horizons `1`, `5`, `22`, and no dataframe index.
- Include exactly one row for every `origin_date,horizon` key in your phase’s schedule. A successful prediction is finite and strictly positive. Keep the row with a blank prediction when a forecast fails.
- In your model folder’s `MODEL.md`, record the run command, model family and configuration or checkpoint, input columns and lookbacks, data snapshot and cutoff, fit and update policy, any conversion back to variance units, and reasons for failed forecasts. If prior exposure to test-period data is known for a pretrained model, record it.

Use the shared validator when writing a submission. This smoke test exports trailing means that could have been calculated at each origin. Replace the example calculation with your own forecasts.

```python
from pathlib import Path

import pandas as pd

from pipeline.csv_io import DAILY, FORECAST, HORIZONS, KEY, export, load

root = Path("data/alpaca")
daily = pd.concat(
    [load(root / phase / "daily_inputs.csv", DAILY) for phase in ("train", "validation", "test")],
    ignore_index=True,
).set_index("date")
schedule = load(root / "test/forecast_schedule.csv", KEY)
trailing = {
    h: daily["spy_rv"].rolling(h, min_periods=h).mean() for h in HORIZONS
}
submission = schedule.assign(
    model_id="example_persistence",
    prediction=[trailing[h].get(day) for day, h in schedule.itertuples(index=False, name=None)],
)
submission["prediction"] = submission["prediction"].where(submission["prediction"].gt(0))
export(submission[list(FORECAST)], Path("reports/preview/example_persistence.csv"), FORECAST, schedule=schedule)
```

## 6. Combine forecasts and report results

The integration teammate collects one matching schedule and actuals file per phase, plus the model CSVs. Join on `origin_date,horizon`. One starting combination is the equal-weight average on origins where every component forecast exists. For learned weights, fit on earlier out-of-sample validation predictions with outcomes available by the fitting cutoff. For example, minimize squared error subject to nonnegative weights summing to one, separately by horizon. Export each combination with the same four forecast columns and its own ID, such as `combined_forecasts/test/equal_weight.csv`. See Lecture Notes 8 §8.4.

**RMSE is the primary metric.** Square the forecast errors, average them, and take the square root. Lower is better. Report it separately for each horizon in the same units as realized variance. Squared loss aligns with the conditional mean target. See Lecture Notes 8 §8.2. QLIKE is an optional secondary metric.

Within each horizon, score every method and the persistence baseline on the **same origin dates**, where the actual, baseline, and all included forecasts are present. The set of dates can differ across horizons. Missing predictions can reduce the common sample for everyone, so include coverage alongside scores.

Preview the report using the example submission from step 5.

```sh
python -m pipeline.report --schedule data/alpaca/test/forecast_schedule.csv \
  --actuals data/alpaca/test/actuals.csv \
  --forecasts reports/preview/example_persistence.csv --selected-id example_persistence \
  --output reports/preview
```

For the final test report, pass the test-phase files for each included model and combination after `--forecasts`. Set `--selected-id` to the ID of the combination you want to plot.

| Output | How to read it |
|---|---|
| `report.md` and `scores.csv` | One row per method and the baseline, with RMSE columns for 1, 5, and 22 sessions. The Markdown table highlights the lowest score per horizon. |
| `coverage.csv` | Scheduled and scored origin counts, missing actuals, and missing predictions by method and horizon. The baseline row counts missing baseline values as missing predictions. Zero common origins means RMSE is undefined for that horizon. |
| `forecasts.png` and `forecasts.svg` | Three panels on a shared forecast-origin date axis. Actuals are charcoal, the selected forecast blue, and the baseline dashed gray. Missing values appear as gaps. |
