# Dataset quality summary
SPY source and adjustments: Alpaca consolidated SIP, raw USD trade OHLC, volume in shares
VIX source: Cboe VIX historical CSV
Raw timestamps: bar start; explicit UTC offsets; session alignment: America/New_York
Calendar: XNYS (NYSE Arca core-session hours); exchange_calendars 4.13.3
Forecast cutoff assumption: 17:00 on the origin date America/New_York. Confirm actual feed availability; the VIX history has no as-of timestamps.
SPY input: spy_5min.csv; SHA-256 f4f733fc4bfcdc2a645f832a2072e2e1bd20dc4ea8a74ae043647eed0fb3fe7a
VIX input: VIX_History.csv; SHA-256 311c6cf4c34f712b60c3604747652aa47df0ba25058d35651ac1cf8d93bf5119
Market halt input: market_halts.csv; SHA-256 2760f0d65c0440dfad0e3c4cb2257bcb82a4a1b9d6a7467ff0d9d19901590f86

Calendar sessions: 2733 (2016-01-04 through 2026-11-13)
SPY bars: 501259 input rows; 212404 checked within 2016-01-01 through 2026-11-13
Extended-hours SPY bars removed at request: 288855
VIX rows: 9317; dates 1990-01-02 through 2026-11-16
SPY variance coverage: 2733 sessions (2731 based on observed five-minute bars and verified halts, 2 estimated)
VIX close coverage: 2733 sessions
Off-calendar SPY bar dates inside data window: none

Incomplete SPY bar coverage (2)
- 2016-02-02 (1 missing bars)
- 2019-08-12 (5 missing bars)

Missing VIX session dates (0)
- none

Invalid VIX sessions (0)
- none

Documented market-wide halts with valid daily measures
Empty interior slots fully covered by a halt contribute zero return and zero volume. Variance includes the change from the pre-halt close to the next observed close. Daily OHLCV uses observed bars.
- 2020-03-09: 09:34:13–09:49:13 ET, empty bar starts 09:35, 09:40 ET. [Source](https://www.nyse.com/publicdocs/nyse/markets/nyse/Report_of_the_Market-Wide_Circuit_Breaker_Working_Group.pdf)
- 2020-03-12: 09:35:44–09:50:44 ET, empty bar starts 09:40, 09:45 ET. [Source](https://www.nyse.com/publicdocs/nyse/markets/nyse/Report_of_the_Market-Wide_Circuit_Breaker_Working_Group.pdf)
- 2020-03-16: 09:30:01–09:45:01 ET, empty bar starts 09:35, 09:40 ET. [Source](https://www.nyse.com/publicdocs/nyse/markets/nyse/Report_of_the_Market-Wide_Circuit_Breaker_Working_Group.pdf)
- 2020-03-18: 12:56:17–13:11:17 ET, empty bar starts 13:00, 13:05 ET. [Source](https://www.nyse.com/publicdocs/nyse/markets/nyse/Report_of_the_Market-Wide_Circuit_Breaker_Working_Group.pdf)

## Estimated training variances

Both dates have a variance estimate in the daily inputs. Their complete SPY OHLCV fields remain blank.
Training target and baseline windows use these estimates. Validation and test labels use observed variance.

- 2016-02-02 uses a 10-minute return from 11:20 to 11:30 ET. Availability date is 2016-02-02.
- 2019-08-12 uses a 25-minute return from 15:35 ET to the official NYSE Arca close of $288.07 at 16:00 ET. The conservative availability date is 2019-08-14, following Nasdaq’s August 13 update. [NYSE closing prices](https://www.nyse.com/publicdocs/nyse/notifications/market-status/NYSE_Group_Closing_Prices_08122019.xlsx) · [Nasdaq incident update](https://www.nasdaqtrader.com/TraderNews.aspx?id=nfnnews2019-01)

1-session horizon. Scheduled origins: 2733
- Missing actual (1): 2026-11-13
- Missing baseline (0): none

5-session horizon. Scheduled origins: 2733
- Missing actual (5): 2026-11-09, 2026-11-10, 2026-11-11, 2026-11-12, 2026-11-13
- Missing baseline (4): 2016-01-04, 2016-01-05, 2016-01-06, 2016-01-07

22-session horizon. Scheduled origins: 2733
- Missing actual (22): 2026-10-15, 2026-10-16, 2026-10-19, 2026-10-20, 2026-10-21, 2026-10-22, 2026-10-23, 2026-10-26, 2026-10-27, 2026-10-28, 2026-10-29, 2026-10-30, 2026-11-02, 2026-11-03, 2026-11-04, 2026-11-05, 2026-11-06, 2026-11-09, 2026-11-10, 2026-11-11, 2026-11-12, 2026-11-13
- Missing baseline (21): 2016-01-04, 2016-01-05, 2016-01-06, 2016-01-07, 2016-01-08, 2016-01-11, 2016-01-12, 2016-01-13, 2016-01-14, 2016-01-15, 2016-01-19, 2016-01-20, 2016-01-21, 2016-01-22, 2016-01-25, 2016-01-26, 2016-01-27, 2016-01-28, 2016-01-29, 2016-02-01, 2016-02-02
