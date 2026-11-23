# Forecast RMSE

RMSE in unannualized variance units on common origins.

| Method | 1 session | 5 sessions | 22 sessions |
|---|---:|---:|---:|
| timestral_209m_rv_vix | 7.0954982e-05 | 5.9750892e-05 | 6.2584835e-05 |
| timestral_28m_rv_vix | **7.0256221e-05** | **5.8976708e-05** | 6.1106325e-05 |
| Persistence baseline | 8.3094634e-05 | 6.5903006e-05 | **5.6139168e-05** |

## Coverage

| Horizon | Scheduled | Scored | Missing actuals |
|---:|---:|---:|---:|
| 1 | 481 | 481 | 0 |
| 5 | 481 | 481 | 0 |
| 22 | 481 | 481 | 0 |

Missing predictions, including blanks and missing baseline values, are listed in `coverage.csv`.
