# forecast_combination

**DSBA Forecasting and Predictive Analytics · Topic 15**

Can combining forecasts improve predictions of SPY volatility over the next trading day, week, and month? We compare point forecasts from four model families: econometrics, machine learning, deep learning, and time-series foundation models. This README defines what each model predicts and what its owner submits, so the integration teammate can compare, combine, and plot forecasts from CSVs alone.

## What we forecast

We measure daily realized variance for **SPY during regular trading hours**, excluding overnight price changes. Start at the session open and follow the five-minute bar closes through the actual session close. Square each consecutive log return and sum the squares:

$$
RV_t = \sum_{j=1}^{m_t}\left[\log(P_{t,j}/P_{t,j-1})\right]^2.
$$

Here $P_{t,0}$ is the session opening price. A normal 09:30–16:00 New York session gives 78 returns, including the return from the open to the first five-minute close. Use the actual session end on early-close days. Flag incomplete sessions during data preparation. Daily realized volatility is $\sqrt{RV_t}$.

At each forecast origin $t$, predict the **expected average daily realized variance** across the next $h$ SPY trading sessions:

$$
Y_{t,h}=\frac{1}{h}\sum_{j=1}^{h}RV_{t+j}.
$$

| `horizon` | Sessions that contribute to the target |
|---:|---|
| 1 | The next trading session |
| 5 | The first through the fifth subsequent trading sessions |
| 22 | The first through the twenty-second subsequent trading sessions |

The origin session is excluded. Report **mean point forecasts** in unannualized variance based on decimal returns. For example, 1% daily volatility corresponds to variance of `0.0001`. If you fit a model on a transformed target, document how you recover the mean on the variance scale. Exponentiating a predicted mean log variance alone generally yields a different quantity.

> **Data assumption.** We expect to obtain sufficient SPY regular-session five-minute OHLCV and daily VIX history. The sources, permissions, date coverage, and data quality still need verification. If suitable intraday data are unavailable, agree on and document a revised target before modelling.

## Shared inputs and forecast timing

The data owner prepares one daily inputs file with one row per SPY session. Dates refer to the New York trading calendar.

| Columns | Meaning |
|---|---|
| `date` | SPY trading-session date |
| `spy_open`, `spy_high`, `spy_low`, `spy_close`, `spy_volume` | SPY regular-session OHLCV |
| `spy_rv` | SPY realized variance for that session |
| `vix_close` | VIX closing level in its quoted units, an option-implied measure of approximately 30-day annualized S&P 500 volatility |

Models may use different subsets, lags, and trailing features from these inputs. For each `origin_date`, forecast after the required observations for that session are available and before the next SPY session. Document the precise cutoff once data sources are confirmed.

At each origin, use only information available by the cutoff for inputs, preprocessing, model fitting, and any weight fitting. A training target can be used once its **entire** outcome window has completed and the outcome is available. For example, a 22-session label is still unavailable 10 sessions into its window. Store future targets separately from model inputs.

## What each model owner submits

Submit `forecasts/<model_id>.csv` and a short note with your model family, model or checkpoint, inputs and lookback, fitting or update procedure, and any output conversion. Include enough information for a teammate to understand how the predictions were produced.

Use exactly these columns. The numbers below are illustrative.

```csv
model_id,origin_date,horizon,prediction
example_model,2024-06-03,1,0.000100
example_model,2024-06-03,5,0.000120
example_model,2024-06-03,22,0.000140
```

- Use UTF-8 CSV, a header row, ISO dates (`YYYY-MM-DD`), and no exported dataframe index.
- Use one `model_id` per file, unique across submissions. Submit exactly one row per requested `origin_date,horizon` from the shared schedule. Use integer horizons `1`, `5`, and `22`.
- A successful prediction is strictly positive and finite. If a forecast fails, keep the row, leave `prediction` empty, and document the reason in your submission note.

## Handoff to the integration teammate

| File | Columns | Owner |
|---|---|---|
| `forecast_schedule.csv` | `origin_date,horizon` | Data and integration teammates |
| `actuals.csv` | `origin_date,horizon,actual,baseline` | Data owner |
| `forecasts/<model_id>.csv` | `model_id,origin_date,horizon,prediction` | Model owners |
| `combined_forecasts/<model_id>.csv` | Same four columns as a model file, with a unique ID for each combination | Integration teammate |

The schedule and actuals each have one row per `origin_date,horizon`. In `actuals.csv`, `actual` is the average over the next $h$ sessions and `baseline` is the average over the **last** $h$ sessions, including the origin session. Compute both against the full SPY trading calendar. An incomplete window stays blank. Join the files on `origin_date,horizon`. Model code, checkpoints, or training environments are unnecessary for scoring and plotting the exports.

Before training is complete, each owner should submit a small sample covering the same agreed origins and all three horizons to check the handoff.

**Combination suggestion.** Start with an equal-weight average on dates where every selected model provided a forecast. If comparing learned weights, one option is to fit nonnegative weights that sum to one by minimizing squared error on earlier out-of-sample forecasts, separately by horizon. Evaluate the resulting combination on later forecasts. The integration teammate decides which approaches to use. See Lecture Notes 8 §8.4 for the course discussion of equal and performance-based weights.

## Evaluation and report

Use **RMSE** in the original variance units as the primary metric, separately for each horizon. RMSE is the square root of mean squared error. Squared loss aligns with the conditional mean forecast we agreed to submit (Lecture Notes 8 §8.2). QLIKE is an optional secondary check.

For the main comparison, score the individual models, combinations, and baseline on the **same origin dates within each horizon** with all required values present. Report how many dates were scored and flag missing forecasts so readers can see any loss of coverage.

- **Score table.** One row per model, combination, and baseline. Use one RMSE column per horizon, report the common sample size, and highlight the lowest score in each column.
- **Forecast figure.** Stack three panels vertically and share the date axis. Show the realized target in charcoal, the selected combination in blue, and the baseline in dashed gray. Label the axis **Forecast origin date** and the quantity **Average daily variance**. At a given origin, the actual line shows the outcome over the subsequent target window. Use one legend and leave gaps where a forecast is missing.
