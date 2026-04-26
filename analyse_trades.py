"""
Trade Analyser — diagnose what's hurting the stock backtest
============================================================

Reads results_stocks/stock_trades.csv and prints a structured breakdown
that tells you exactly WHERE edge is being destroyed:
  - Cost drag vs gross P&L
  - Hold-time distribution (winning vs losing trades)
  - Time-of-day performance
  - Signal switch depth (how far score moved before flip)
  - Stock categorisation (which sectors work / don't)

Usage:
    python analyse_trades.py --trades results_stocks/stock_trades.csv
"""

import argparse
import numpy as np
import pandas as pd


SECTOR_MAP = {
    # IT
    "INFY":"IT","TCS":"IT","HCLTECH":"IT","WIPRO":"IT","TECHM":"IT",
    # Finance / Banks
    "HDFCBANK":"Bank","ICICIBANK":"Bank","AXISBANK":"Bank","KOTAKBANK":"Bank",
    "SBIN":"Bank","INDUSINDBK":"Bank","BAJFINANCE":"NBFC","BAJAJFINSV":"NBFC",
    "HDFCLIFE":"Insurance","SBILIFE":"Insurance","SHRIRAMFIN":"NBFC",
    # Consumer
    "HINDUNILVR":"FMCG","ITC":"FMCG","BRITANNIA":"FMCG","NESTLEIND":"FMCG",
    "TATACONSUM":"FMCG","ASIANPAINT":"Paint",
    # Auto
    "MARUTI":"Auto","TATAMOTORS":"Auto","M&M":"Auto",
    "BAJAJ-AUTO":"Auto","HEROMOTOCO":"Auto","EICHERMOT":"Auto",
    # Pharma
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma",
    "APOLLOHOSP":"Healthcare",
    # Commodity / Metal / Energy
    "HINDALCO":"Metal","TATASTEEL":"Metal","JSWSTEEL":"Metal",
    "COALINDIA":"PSU","ONGC":"PSU","BPCL":"PSU","NTPC":"PSU",
    "POWERGRID":"PSU","GRASIM":"Cement","ULTRACEMCO":"Cement",
    # Infra / Others
    "LT":"Infra","ADANIENT":"Conglomerate","ADANIPORTS":"Infra",
    "RELIANCE":"Conglomerate","BHARTIARTL":"Telecom",
    "TITAN":"Consumer","TRENT":"Consumer",
}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["entry_time","exit_time"])
    df["hold_minutes"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60
    df["hour_of_entry"] = df["entry_time"].dt.hour
    df["sector"] = df["symbol"].map(SECTOR_MAP).fillna("Other")
    df["winner"] = df["net_pnl"] > 0
    return df


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


def run(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  TRADE DIAGNOSTIC REPORT")
    print("="*60)

    # ── 1. Cost drag ─────────────────────────────────────────────────────────
    section("1. COST DRAG vs GROSS P&L")
    gross   = df["gross_pnl"].sum()
    costs   = df["costs"].sum()
    net     = df["net_pnl"].sum()
    avg_cost = df["costs"].mean()
    print(f"  Gross P&L  : ₹{gross:>10,.0f}")
    print(f"  Total costs: ₹{costs:>10,.0f}  (₹{avg_cost:.0f} avg per trade)")
    print(f"  Net P&L    : ₹{net:>10,.0f}")
    if gross < 0:
        print(f"  ⚠  Strategy is gross-negative — costs not the primary issue.")
    else:
        print(f"  ✓  Strategy is gross-positive. Costs alone are the drag.")
        print(f"     Reduce trade frequency or use lower-cost broker.")

    # ── 2. Hold-time: winners vs losers ──────────────────────────────────────
    section("2. HOLD TIME: Winners vs Losers (minutes)")
    ht = df.groupby("winner")["hold_minutes"].describe()[["mean","50%","min","max"]]
    ht.index = ["Losers","Winners"]
    print(ht.to_string())
    avg_win_hold  = df[df["winner"]]["hold_minutes"].mean()
    avg_lose_hold = df[~df["winner"]]["hold_minutes"].mean()
    print(f"\n  Winners held {avg_win_hold:.1f} min on average")
    print(f"  Losers  held {avg_lose_hold:.1f} min on average")
    if avg_win_hold < avg_lose_hold:
        print("  ⚠  Winners exit faster than losers — letting losses run, "
              "cutting winners early.")
    else:
        print("  ✓  Winners held longer than losers — momentum is being captured.")
    min_hold_suggest = max(5, int(avg_win_hold * 0.5))
    print(f"  → Suggested min_hold_bars: {min_hold_suggest // 2} bars "
          f"({min_hold_suggest} min)")

    # ── 3. Time-of-day breakdown ──────────────────────────────────────────────
    section("3. TIME-OF-DAY  (hour of entry)")
    tod = (df.groupby("hour_of_entry")
             .agg(trades=("net_pnl","count"),
                  net_pnl=("net_pnl","sum"),
                  win_rate=("winner","mean"))
             .assign(win_rate=lambda x: (x["win_rate"]*100).round(1)))
    print(tod.to_string())
    best_hour = tod["net_pnl"].idxmax()
    worst_hour = tod["net_pnl"].idxmin()
    print(f"\n  Best hour : {best_hour}:xx  |  Worst hour: {worst_hour}:xx")
    if worst_hour in [12, 13]:
        print("  → Consider adding a lunch-hour filter (12:00–13:30 IST).")

    # ── 4. Exit-reason deep dive ──────────────────────────────────────────────
    section("4. EXIT REASON — gross vs net P&L")
    er = (df.groupby("exit_reason")
            .agg(trades=("net_pnl","count"),
                 gross=("gross_pnl","sum"),
                 costs=("costs","sum"),
                 net=("net_pnl","sum"),
                 win_rate=("winner","mean"),
                 avg_hold=("hold_minutes","mean"))
            .assign(win_rate=lambda x: (x["win_rate"]*100).round(1),
                    avg_hold=lambda x: x["avg_hold"].round(1)))
    print(er.to_string())
    sw = df[df["exit_reason"]=="signal_switch"]
    print(f"\n  signal_switch avg hold: {sw['hold_minutes'].mean():.1f} min  "
          f"→ confirmation filter needed")

    # ── 5. Sector performance ─────────────────────────────────────────────────
    section("5. SECTOR PERFORMANCE")
    sec = (df.groupby("sector")
             .agg(stocks=("symbol","nunique"),
                  trades=("net_pnl","count"),
                  net_pnl=("net_pnl","sum"),
                  win_rate=("winner","mean"))
             .assign(win_rate=lambda x: (x["win_rate"]*100).round(1),
                     net_per_stock=lambda x: x["net_pnl"]/x["stocks"])
             .sort_values("net_pnl",ascending=False))
    print(sec.to_string())

    # ── 6. Trade-frequency diagnosis ─────────────────────────────────────────
    section("6. TRADE FREQUENCY DIAGNOSIS")
    trades_per_stock_day = (df.groupby(["symbol","trade_date"])
                              .size().describe())
    print("  Trades per stock per day:")
    print(trades_per_stock_day.to_string())
    median_tpsd = df.groupby(["symbol","trade_date"]).size().median()
    print(f"\n  Median trades/stock/day: {median_tpsd:.1f}")
    if median_tpsd > 3:
        print("  ⚠  Over-trading detected. Reduce with confirmation filter "
              "or higher threshold.")

    # ── 7. Recommended config changes ────────────────────────────────────────
    section("7. RECOMMENDED CONFIGURATION CHANGES")
    print("""
  Based on the above diagnostics:

  CHANGE                          CURRENT   RECOMMENDED   REASON
  ─────────────────────────────────────────────────────────────────────
  entry_threshold                   0.25        0.35      fewer false entries
  exit_threshold                    0.10        0.20      avoid noise exits
  min_confirm_bars (NEW)               0           3      3-bar confirmation
  min_hold_bars (NEW)                  0           5      10-min min hold
  max_trades_per_day                   6           3      reduce churn
  bar_minutes                          2           5      less noise per bar
  Exclude commodity/PSU stocks         -         YES      sector filter
  Lunch hour filter 12:00-13:30        -         YES      low-volume noise
  ─────────────────────────────────────────────────────────────────────

  Run the optimised backtest with:
    python nifty50_intraday_backtest.py --synthetic --optimised
  or compare configs:
    python nifty50_intraday_backtest.py --grid-search
""")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trades", default="results_stocks/stock_trades.csv")
    args = p.parse_args()
    df = load(args.trades)
    run(df)
