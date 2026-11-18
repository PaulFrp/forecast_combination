# Dataset quality summary
SPY source and adjustments: SYNTHETIC FIXTURE. Deterministic prices and volumes
VIX source: SYNTHETIC FIXTURE. Deterministic quoted levels
Raw timestamps: bar start; explicit UTC offsets; session alignment: America/New_York
Calendar: XNYS (NYSE Arca core-session hours); exchange_calendars 4.13.3
Forecast cutoff assumption: 17:00 on the origin date America/New_York. Confirm actual feed availability; the VIX history has no as-of timestamps.
SPY input: synthetic_spy.csv; SHA-256 6e4aa34841297576b91e800635c9f4c46552bfb033857702da6f6d424f7af3a5
VIX input: synthetic_vix.csv; SHA-256 2bda6c9da4ed76511e4a24831eb942e9ffa4126ecf72c347d2053ed5c2b13b45

Calendar sessions: 75 (2024-10-14 through 2025-01-31)
SPY bars: 5777 input rows; 5777 checked within 2024-10-14 through 2025-01-31
Extended-hours SPY bars removed at request: 0
VIX rows: 74; dates 2024-10-14 through 2025-01-31
Usable SPY sessions: 74; VIX closes: 74; overlap: 73
Off-calendar SPY bar dates inside data window: none

Incomplete or invalid SPY sessions (1)
- 2024-12-27 (1 missing bars)

Missing VIX session dates (1)
- 2024-12-16

Invalid VIX sessions (0)
- none

1-session horizon. Scheduled origins: 53
- Missing actual (2): 2024-12-26, 2025-01-31
- Missing baseline (1): 2024-12-27

5-session horizon. Scheduled origins: 53
- Missing actual (10): 2024-12-19, 2024-12-20, 2024-12-23, 2024-12-24, 2024-12-26, 2025-01-27, 2025-01-28, 2025-01-29, 2025-01-30, 2025-01-31
- Missing baseline (5): 2024-12-27, 2024-12-30, 2024-12-31, 2025-01-02, 2025-01-03

22-session horizon. Scheduled origins: 53
- Missing actual (44): 2024-11-25, 2024-11-26, 2024-11-27, 2024-11-29, 2024-12-02, 2024-12-03, 2024-12-04, 2024-12-05, 2024-12-06, 2024-12-09, 2024-12-10, 2024-12-11, 2024-12-12, 2024-12-13, 2024-12-16, 2024-12-17, 2024-12-18, 2024-12-19, 2024-12-20, 2024-12-23, 2024-12-24, 2024-12-26, 2024-12-30, 2024-12-31, 2025-01-02, 2025-01-03, 2025-01-06, 2025-01-07, 2025-01-08, 2025-01-10, 2025-01-13, 2025-01-14, 2025-01-15, 2025-01-16, 2025-01-17, 2025-01-21, 2025-01-22, 2025-01-23, 2025-01-24, 2025-01-27, 2025-01-28, 2025-01-29, 2025-01-30, 2025-01-31
- Missing baseline (22): 2024-12-27, 2024-12-30, 2024-12-31, 2025-01-02, 2025-01-03, 2025-01-06, 2025-01-07, 2025-01-08, 2025-01-10, 2025-01-13, 2025-01-14, 2025-01-15, 2025-01-16, 2025-01-17, 2025-01-21, 2025-01-22, 2025-01-23, 2025-01-24, 2025-01-27, 2025-01-28, 2025-01-29, 2025-01-30
