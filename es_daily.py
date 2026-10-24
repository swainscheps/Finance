es_daily_ranges.py
# Builds a CSV of recent /ES daily ranges, % changes, and event flags.
#
# Setup (once):   pip install yfinance pandas numpy
# Run:            python es_daily_ranges.py
# Output:         es_daily_ranges.csv in the same folder
#
# NOTE: yfinance has NO economic-calendar feed. CPI / jobs / FOMC dates are
# hardcoded below from the official BLS and Federal Reserve calendars.
# Everything else (opex, VIX expiration, ES roll window, month-end, holidays)
# is computed from rules, so it works for any date range.
#
# LOOKAHEAD RULE: columns starting with "prev_" or "vix_prev_" are known BEFORE
# the open and are safe to filter on. Same-day columns (range, pct_chg_close,
# gap_pct, intraday_pct) are outcomes -- use them as the thing you're testing.
 
import datetime as dt
 
import numpy as np
import pandas as pd
import yfinance as yf
 
PERIOD = "9mo"        # hardcoded event dates below start Dec 2025
WIDE_RATIO = 1.5      # prev-day range >= 1.5x its prior-20-day avg -> "wide"
NARROW_RATIO = 0.7    # prev-day range <= 0.7x -> "narrow"
OUT_FILE = "es_daily_ranges.csv"
 
# ---- Hardcoded release dates (BLS + Federal Reserve calendars) -------------
EVENT_COVERAGE_START = pd.Timestamp("2025-12-01")
 
# CPI release days (8:30 ET)
CPI_DATES = [
    "2025-12-18", "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10",
    "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12", "2026-09-11",
    "2026-10-14", "2026-11-10", "2026-12-10",
]
# Employment Situation / NFP release days (8:30 ET)
NFP_DATES = [
    "2025-12-16", "2026-01-09", "2026-02-11", "2026-03-06", "2026-04-03",
    "2026-05-08", "2026-06-05", "2026-07-02", "2026-08-07", "2026-09-04",
    "2026-10-02", "2026-11-06", "2026-12-04",
]
# FOMC decision days (day 2 of each meeting, 2:00 ET)
FOMC_DATES = [
    "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]
 
 
# ---- Helpers ----------------------------------------------------------------
def flatten(d):
    """Newer yfinance versions return multi-level columns; flatten them."""
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d
 
 
def clean_index(d):
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d
 
 
def third_friday(year, month):
    first = dt.date(year, month, 1)
    first_friday = first + dt.timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + dt.timedelta(days=14)
 
 
def int_flag(cond, valid=None):
    """0/1 flag; blank (NA) where `valid` is False (e.g. not enough history)."""
    arr = cond.astype(float)
    if valid is not None:
        arr = arr.where(valid)
    return arr.astype("Int64")
 
 
# ---- Download ES -------------------------------------------------------------
df = yf.download("ES=F", period=PERIOD, interval="1d",
                 auto_adjust=False, progress=False)
df = clean_index(flatten(df).dropna(subset=["Open", "High", "Low", "Close"]))
idx = df.index
 
if idx[0] < EVENT_COVERAGE_START:
    print(f"WARNING: data starts {idx[0].date()}, but CPI/NFP/FOMC dates only "
          f"cover {EVENT_COVERAGE_START.date()} onward. Those flags will be 0 "
          f"before then. Add earlier dates to the lists above.\n")
if idx[-1].date() == dt.date.today():
    print("NOTE: last row may be an incomplete session (today's range is still forming).\n")
 
o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
prev_c = c.shift(1)
 
# ---- Core columns + % changes -----------------------------------------------
out = pd.DataFrame({
    "date": idx.strftime("%Y-%m-%d").to_numpy(),
    "day_of_week": idx.strftime("%A").to_numpy(),
    "open": o.round(2),
    "high": h.round(2),
    "low": l.round(2),
    "close": c.round(2),
    "range": (h - l).round(2),
    "range_pct": ((h - l) / o * 100).round(3),            # range as % of open
    "pct_chg_close": ((c / prev_c - 1) * 100).round(3),   # close vs prior close
    "gap_pct": ((o / prev_c - 1) * 100).round(3),         # open vs prior close
    "intraday_pct": ((c / o - 1) * 100).round(3),         # close vs open
}, index=idx)
 
# ---- Previous-day context (safe to use as filters) ---------------------------
out["prev_day_pct_chg"] = out["pct_chg_close"].shift(1)
out["prev_day_intraday_pct"] = out["intraday_pct"].shift(1)
out["prev_day_dir"] = np.where(out["prev_day_pct_chg"].isna(), "",
                               np.where(out["prev_day_pct_chg"] >= 0, "up", "down"))
out["prev_day_range"] = out["range"].shift(1)
out["prev_day_range_pct"] = out["range_pct"].shift(1)
out["avg_range_5d"] = out["range"].rolling(5).mean().shift(1).round(2)
out["avg_range_20d"] = out["range"].rolling(20).mean().shift(1).round(2)
out["range_vs_20d"] = (out["range"] / out["avg_range_20d"]).round(2)   # outcome
 
# prev-day range relative to the 20 days BEFORE it (no self-inclusion)
ratio = out["prev_day_range"] / out["range"].rolling(20).mean().shift(2)
out["prev_range_vs_20d"] = ratio.round(2)
out["prev_range_wide"] = int_flag(ratio >= WIDE_RATIO, ratio.notna())
out["prev_range_narrow"] = int_flag(ratio <= NARROW_RATIO, ratio.notna())
 
# ---- Event / calendar flags ---------------------------------------------------
FLAG_COLS = [
    "flag_cpi", "flag_nfp", "flag_fomc", "flag_major_event",
    "flag_day_after_major_event", "flag_monthly_opex", "flag_quad_witching",
    "flag_opex_week", "flag_vix_expiry", "flag_es_roll_window",
    "flag_first_td_month", "flag_last_td_month", "flag_last_td_quarter",
    "flag_post_holiday", "flag_pre_holiday",
]
flags = pd.DataFrame(0, index=idx, columns=FLAG_COLS)
 
 
def on_or_before(d):
    """Calendar date -> trading day on/before it (handles holiday-shifted opex).
    Returns None if the date is past the end of the data."""
    ts = pd.Timestamp(d)
    if ts > idx[-1]:
        return None
    prior = idx[idx <= ts]
    return prior[-1] if len(prior) else None
 
 
def mark_exact(dates, col):
    for d in dates:
        ts = pd.Timestamp(d)
        if ts in idx:
            flags.loc[ts, col] = 1
 
 
mark_exact(CPI_DATES, "flag_cpi")
mark_exact(NFP_DATES, "flag_nfp")
mark_exact(FOMC_DATES, "flag_fomc")
flags["flag_major_event"] = flags[["flag_cpi", "flag_nfp", "flag_fomc"]].max(axis=1)
flags["flag_day_after_major_event"] = (
    flags["flag_major_event"].shift(1).fillna(0).astype(int))
 
# Rule-based: opex, quad witching, VIX expiry, ES quarterly roll window
opex_weeks = set()
for p in pd.period_range(idx[0].to_period("M") - 1, idx[-1].to_period("M") + 1, freq="M"):
    tf = third_friday(p.year, p.month)
    opex_weeks.add(tuple(int(x) for x in pd.Timestamp(tf).isocalendar()[:2]))
 
    td = on_or_before(tf)                       # Thursday if Friday is a holiday
    if td is not None:
        flags.loc[td, "flag_monthly_opex"] = 1
        if p.month in (3, 6, 9, 12):
            flags.loc[td, "flag_quad_witching"] = 1
 
    # VIX expiry: the Wednesday 30 days before NEXT month's third Friday
    nxt = p + 1
    vx = on_or_before(third_friday(nxt.year, nxt.month) - dt.timedelta(days=30))
    if vx is not None:
        flags.loc[vx, "flag_vix_expiry"] = 1
 
    # ES rolls ~8 days before quarterly expiry. Close-to-close % changes in this
    # window can include fake "gaps" from the contract switch. High-low range
    # is within one bar, so it is mostly unaffected.
    if p.month in (3, 6, 9, 12):
        start = pd.Timestamp(tf - dt.timedelta(days=8))
        flags.loc[(idx >= start) & (idx <= pd.Timestamp(tf)), "flag_es_roll_window"] = 1
 
iso = idx.isocalendar()
flags["flag_opex_week"] = [
    int((int(y), int(w)) in opex_weeks) for y, w in zip(iso["year"], iso["week"])
]
 
# First / last trading day of month and quarter
per = pd.Series(idx.to_period("M"), index=idx)
first_td = per != per.shift(1)
last_td = per != per.shift(-1)
if (idx[0] - pd.offsets.BDay(1)).month == idx[0].month:      # data starts mid-month
    first_td.iloc[0] = False
if (idx[-1] + pd.offsets.BDay(1)).month == idx[-1].month:    # data ends mid-month
    last_td.iloc[-1] = False
flags["flag_first_td_month"] = first_td.astype(int)
flags["flag_last_td_month"] = last_td.astype(int)
flags["flag_last_td_quarter"] = (last_td & idx.month.isin([3, 6, 9, 12])).astype(int)
 
# Day after / before a market holiday (inferred from gaps in the trading dates)
dates = pd.Series(idx, index=idx)
gap_prev = (dates - dates.shift(1)).dt.days
gap_next = (dates.shift(-1) - dates).dt.days
wd = pd.Series(idx.weekday, index=idx)
flags["flag_post_holiday"] = ((gap_prev > 3) | ((gap_prev > 1) & (wd != 0))).astype(int)
flags["flag_pre_holiday"] = ((gap_next > 3) | ((gap_next > 1) & (wd != 4))).astype(int)
 
out = out.join(flags)
 
labels = {"flag_cpi": "CPI", "flag_nfp": "NFP", "flag_fomc": "FOMC",
          "flag_monthly_opex": "OPEX", "flag_quad_witching": "QUAD_WITCH",
          "flag_vix_expiry": "VIX_EXP", "flag_post_holiday": "POST_HOL",
          "flag_pre_holiday": "PRE_HOL"}
out["events"] = flags[list(labels)].apply(
    lambda r: ";".join(labels[k] for k in labels if r[k]), axis=1)
 
# ---- VIX (prior-day level and change) ------------------------------------------
try:
    vix = yf.download("^VIX", period=PERIOD, interval="1d",
                      auto_adjust=False, progress=False)
    vc = clean_index(flatten(vix))["Close"]
    out["vix_prev_close"] = vc.shift(1).reindex(idx).round(2)
    out["vix_prev_day_pct_chg"] = (vc.pct_change().shift(1) * 100).reindex(idx).round(2)
except Exception as e:
    print(f"WARNING: could not download VIX ({e}). Skipping VIX columns.\n")
 
# ---- Save + quick look -----------------------------------------------------------
out.to_csv(OUT_FILE, index=False)
 
print(out.tail(10).to_string(index=False))
print(f"\nSaved {len(out)} rows, {out.shape[1]} columns to {OUT_FILE}\n")
 
print("Average range by day of week:")
print(out.groupby("day_of_week")["range"].agg(["count", "mean", "median"])
      .round(1).to_string())
 
rows = []
for col in [c for c in out.columns if c.startswith("flag_")] + ["prev_range_wide", "prev_range_narrow"]:
    on, off = out.loc[out[col] == 1, "range"], out.loc[out[col] == 0, "range"]
    rows.append((col, len(on),
                 round(on.mean(), 1) if len(on) else np.nan,
                 round(off.mean(), 1) if len(off) else np.nan))
print("\nAverage range when flag is ON vs OFF:")
print(pd.DataFrame(rows, columns=["flag", "days_on", "avg_range_on", "avg_range_off"])
      .to_string(index=False))
 
