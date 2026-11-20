# Forecast RMSE

RMSE in unannualized variance units on common origins.

| Method | 1 session | 5 sessions | 22 sessions |
|---|---:|---:|---:|
| xgboost | **0.00011292562** | **9.9300839e-05** | **6.8024326e-05** |
| Persistence baseline | 0.00016752674 | 0.00013842873 | 8.805514e-05 |

## Coverage

| Horizon | Scheduled | Scored | Missing actuals |
|---:|---:|---:|---:|
| 1 | 949 | 949 | 0 |
| 5 | 949 | 949 | 0 |
| 22 | 949 | 949 | 0 |

Missing predictions, including blanks and missing baseline values, are listed in `coverage.csv`.
