"""
fetch_data.py — Fetch today's Nifty 2-min OHLCV bars
=====================================================

Called by the GitHub Actions live_signal workflow before signal generation.
Outputs: data/today_2min.csv  (DatetimeIndex, columns: open high low close volume)

Data sources tried in order:
  1. Broker API  (Zerodha or Upstox)  — requires BROKER_ACCESS_TOKEN secret
  2. yfinance    (^NSEI, free)         — always available, no credentials needed
                                         Note: index has no volume; F8 (volume
                                         ratio) gracefully uses 0 as neutral.

Usage:
    python fetch_data.py [--bar-minutes 2] [--output data/today_2min.csv]
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("fetch_data")

IST              = ZoneInfo("Asia/Kolkata")
BROKER_API_KEY   = os.getenv("BROKER_API_KEY", "")
BROKER_TOKEN     = os.getenv("BROKER_ACCESS_TOKEN", "")
BROKER_TYPE      = os.getenv("BROKER_TYPE", "zerodha").lower()


# ─── Source 1: Zerodha Kite ───────────────────────────────────────────────────

def fetch_zerodha(bar_minutes: int) -> pd.DataFrame:
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=BROKER_API_KEY)
    kite.set_access_token(BROKER_TOKEN)

    now   = datetime.now(IST)
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)

    # Nifty 50 index instrument token = 256265
    data  = kite.historical_data(256265, start, now,
                                  f"{bar_minutes}minute")
    df    = pd.DataFrame(data).rename(columns={"date": "datetime"})
    df    = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(IST) \
                if df.index.tzinfo is None else df.index
    log.info(f"Zerodha: fetched {len(df)} bars")
    return df


# ─── Source 2: Upstox ─────────────────────────────────────────────────────────

def fetch_upstox(bar_minutes: int) -> pd.DataFrame:
    import upstox_client
    cfg                  = upstox_client.Configuration()
    cfg.access_token     = BROKER_TOKEN
    api                  = upstox_client.HistoryApi(upstox_client.ApiClient(cfg))

    today  = date.today().isoformat()
    resp   = api.get_historical_candle_data1(
        "NSE_INDEX|Nifty 50", f"{bar_minutes}minute", today, today, "v2")

    candles = resp.data.candles
    df = pd.DataFrame(candles,
                      columns=["datetime","open","high","low","close","volume","oi"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")[["open","high","low","close","volume"]]
    log.info(f"Upstox: fetched {len(df)} bars")
    return df


# ─── Source 3: yfinance (free fallback) ───────────────────────────────────────

def fetch_yfinance(bar_minutes: int) -> pd.DataFrame:
    import yfinance as yf

    # yfinance supports 1m interval for the last 7 days
    raw = yf.download("^NSEI", period="1d", interval="1m",
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned empty dataframe for ^NSEI")

    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                   for c in raw.columns]
    raw = raw[["open", "high", "low", "close", "volume"]]

    # Resample to bar_minutes
    if bar_minutes > 1:
        raw = raw.resample(f"{bar_minutes}min").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna(subset=["open", "close"])

    # Convert index to IST and filter to today's session
    if raw.index.tzinfo is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert(IST)

    today_str   = date.today().isoformat()
    session_start = pd.Timestamp(f"{today_str} 09:15:00", tz=IST)
    session_end   = pd.Timestamp(f"{today_str} 15:35:00", tz=IST)
    raw = raw[(raw.index >= session_start) & (raw.index <= session_end)]

    # yfinance index data usually has zero volume — fill with 100k placeholder
    # so F8 (volume ratio) is neutral rather than NaN
    if raw["volume"].sum() == 0:
        log.warning("Volume is zero (index data) — using placeholder volume 100000")
        raw["volume"] = 100_000.0

    log.info(f"yfinance: fetched {len(raw)} × {bar_minutes}-min bars")
    return raw


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def fetch(bar_minutes: int) -> pd.DataFrame:
    """Try broker API first, fall back to yfinance."""
    if BROKER_API_KEY and BROKER_TOKEN:
        try:
            if BROKER_TYPE == "zerodha":
                return fetch_zerodha(bar_minutes)
            elif BROKER_TYPE == "upstox":
                return fetch_upstox(bar_minutes)
        except Exception as e:
            log.warning(f"Broker fetch failed ({e}), falling back to yfinance")

    return fetch_yfinance(bar_minutes)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bar-minutes", type=int, default=2)
    p.add_argument("--output",      type=str, default="data/today_2min.csv")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    df = fetch(args.bar_minutes)

    if df.empty:
        log.error("No data fetched — aborting")
        sys.exit(1)

    # Strip timezone from index so CSV is clean and portable
    if df.index.tzinfo is not None:
        df.index = df.index.tz_localize(None)

    df.to_csv(args.output)
    log.info(f"Saved {len(df)} bars → {args.output}")
    log.info(f"  First bar : {df.index[0]}   close={df['close'].iloc[0]:.1f}")
    log.info(f"  Last bar  : {df.index[-1]}   close={df['close'].iloc[-1]:.1f}")
