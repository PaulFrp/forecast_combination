# Forecast RMSE

RMSE in unannualized variance units on common origins.

| Method | 1 session | 5 sessions | 22 sessions |
|---|---:|---:|---:|
| synthetic_a | 4.68639e-05 | **3.0265432e-05** | **9.2740413e-06** |
| synthetic_b | 5.6646563e-05 | 4.5591472e-05 | 3.7453106e-05 |
| Persistence baseline | **3.8901867e-05** | 5.6283267e-05 | 1.0665494e-05 |

## Coverage

| Horizon | Scheduled | Scored | Missing actuals |
|---:|---:|---:|---:|
| 1 | 53 | 50 | 2 |
| 5 | 53 | 37 | 10 |
| 22 | 53 | 8 | 44 |

Missing predictions, including blanks and missing baseline values, are listed in `coverage.csv`.
