# es_daily_ranges.py
# Builds a CSV of the last ~6 months of /ES daily ranges (high - low, in points).
#
# Setup (once):   pip install yfinance pandas
# Run:            python es_daily_ranges.py
# Output:         es_daily_ranges.csv in the same folder

import pandas as pd
import yfinance as yf

df = yf.download("ES=F", period="6mo", interval="1d",
                 auto_adjust=False, progress=False)

# Newer yfinance versions return multi-level column names; flatten them.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

out = pd.DataFrame({
    "date": df.index.strftime("%Y-%m-%d"),
    "day_of_week": df.index.strftime("%A"),
    "range": (df["High"] - df["Low"]).round(2),
})

# Extras that are handy for testing "if yesterday was X, skip" rules.
out["prev_day_range"] = out["range"].shift(1)
out["avg_range_5d"] = out["range"].rolling(5).mean().shift(1).round(2)

out.to_csv("es_daily_ranges.csv", index=False)
print(out.tail(10).to_string(index=False))
print(f"\nSaved {len(out)} rows to es_daily_ranges.csv")
