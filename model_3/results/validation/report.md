# Forecast RMSE

RMSE in unannualized variance units on common origins.

| Method | 1 session | 5 sessions | 22 sessions |
|---|---:|---:|---:|
| har_rv | **7.2971321e-05** | **6.0609435e-05** | **5.3717534e-05** |
| Persistence baseline | 8.3094634e-05 | 6.5903006e-05 | 5.6139168e-05 |

## Coverage

| Horizon | Scheduled | Scored | Missing actuals |
|---:|---:|---:|---:|
| 1 | 481 | 481 | 0 |
| 5 | 481 | 481 | 0 |
| 22 | 481 | 481 | 0 |

Missing predictions, including blanks and missing baseline values, are listed in `coverage.csv`.
