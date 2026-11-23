# Forecast RMSE

RMSE in unannualized variance units on common origins.

| Method | 1 session | 5 sessions | 22 sessions |
|---|---:|---:|---:|
| timestral_209m_rv_vix | 7.0954982e-05 | 5.9750892e-05 | 6.2584835e-05 |
| timestral_209m_rv_vix_lb_1024 | 7.1278483e-05 | 5.9640041e-05 | 6.1478459e-05 |
| timestral_209m_rv_vix_lb_256 | 7.1426875e-05 | 5.9108612e-05 | 5.8867517e-05 |
| timestral_209m_rv_vix_lb_all | 7.1723503e-05 | 5.9926823e-05 | 6.0829976e-05 |
| timestral_209m_rv_vix_ohlcv_lb_256 | 7.0355397e-05 | **5.7923592e-05** | **5.6030149e-05** |
| timestral_28m_rv_vix | **7.0256221e-05** | 5.8976708e-05 | 6.1106325e-05 |
| Persistence baseline | 8.3094634e-05 | 6.5903006e-05 | 5.6139168e-05 |

## Coverage

| Horizon | Scheduled | Scored | Missing actuals |
|---:|---:|---:|---:|
| 1 | 481 | 481 | 0 |
| 5 | 481 | 481 | 0 |
| 22 | 481 | 481 | 0 |

Missing predictions, including blanks and missing baseline values, are listed in `coverage.csv`.
