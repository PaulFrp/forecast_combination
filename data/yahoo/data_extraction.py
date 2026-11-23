from pathlib import Path

import yfinance as yf

df = yf.download("^GSPC", period="20y", interval="1d",
                 auto_adjust=False, progress=False)
df = df[["Open", "High", "Low", "Close", "Volume"]]
df.columns = df.columns.get_level_values(0)  # flatten
df.to_csv(Path(__file__).resolve().with_name("sp500_20y_daily.csv"))
