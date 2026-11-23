# Forecast RMSE

RMSE in unannualized variance units on common origins.

| Method | 1 session | 5 sessions | 22 sessions |
|---|---:|---:|---:|
| timestral_209m_rv_vix | **0.00015046802** | 0.00012162537 | 9.7812539e-05 |
| timestral_209m_rv_vix_lb_256 | 0.0001569907 | 0.00012564061 | 0.0001057981 |
| timestral_209m_rv_vix_ohlcv_lb_256 | 0.00015448117 | **0.00011632262** | **8.1831358e-05** |
| timestral_28m_rv_vix | 0.00015307196 | 0.00012021322 | 8.8389149e-05 |
| Persistence baseline | 0.00016752674 | 0.00013842873 | 8.805514e-05 |

## Coverage

| Horizon | Scheduled | Scored | Missing actuals |
|---:|---:|---:|---:|
| 1 | 949 | 949 | 0 |
| 5 | 949 | 949 | 0 |
| 22 | 949 | 949 | 0 |

Missing predictions, including blanks and missing baseline values, are listed in `coverage.csv`.
