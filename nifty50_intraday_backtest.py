"""
Nifty 50 Intraday Long/Short Backtest Engine
=============================================

Applies the exact same 10-factor composite-score framework used for the
Nifty ATM option selling strategy to all 50 Nifty constituent stocks on
2-minute intraday bars.

Trade logic (equity, not options):
    Composite Score > +threshold  →  LONG  the stock
    Composite Score < -threshold  →  SHORT the stock
    Score drifts to |x| < exit_threshold → Exit to flat
    Score flips sign cleanly → Reverse (stop-and-reverse)
    15:15 IST → Mandatory square-off of all positions

P&L:
    Long  : (exit_price − entry_price) × qty × (1 − costs)
    Short : (entry_price − exit_price) × qty × (1 − costs)
    qty   = floor(capital_per_stock / entry_price)

Costs modelled (per side):
    Brokerage  : ₹20 flat OR 0.03% (whichever lower) — Zerodha intraday
    STT        : 0.025% on sell leg only
    Exchange   : 0.00297% per side
    SEBI       : 0.0001% per side
    Stamp duty : 0.003% on buy side
    GST        : 18% on brokerage

Data sources:
    --synthetic  : regime-switching model (fast, always available)
    --data-dir   : folder of CSVs named <SYMBOL>.csv (real data)
    yfinance     : fetched automatically if neither above is given
                   (only last 7 days at 1-min, resampled to 2-min)

Usage:
    python nifty50_intraday_backtest.py --synthetic --days 40
    python nifty50_intraday_backtest.py --data-dir data/stocks/
    python nifty50_intraday_backtest.py               # fetch via yfinance
"""

import os
import sys
import json
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# Reuse factor engine from the main strategy file
from multi_factor_nifty_strategy import FactorEngine, StrategyConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("nifty50_bt")


# ─── Nifty 50 Universe ────────────────────────────────────────────────────────

NIFTY50_SYMBOLS = [
    "ADANIENT",  "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO","BAJAJFINSV", "BAJFINANCE", "BHARTIARTL", "BPCL",
    "BRITANNIA", "CIPLA",      "COALINDIA",  "DIVISLAB",   "DRREDDY",
    "EICHERMOT", "GRASIM",     "HCLTECH",    "HDFCBANK",   "HDFCLIFE",
    "HEROMOTOCO","HINDALCO",   "HINDUNILVR", "ICICIBANK",  "INDUSINDBK",
    "INFY",      "ITC",        "JSWSTEEL",   "KOTAKBANK",  "LT",
    "M&M",       "MARUTI",     "NESTLEIND",  "NTPC",       "ONGC",
    "POWERGRID", "RELIANCE",   "SBILIFE",    "SBIN",       "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL",  "TCS",
    "TECHM",     "TITAN",      "TRENT",      "ULTRACEMCO", "WIPRO",
]

# yfinance uses .NS suffix for NSE-listed stocks
def yf_symbol(sym: str) -> str:
    return sym.replace("&", "") + ".NS"   # M&M → MM.NS


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class StockConfig:
    # Session
    market_open:        str   = "09:15"
    squareoff_time:     str   = "15:15"
    signal_start:       str   = "09:30"
    bar_minutes:        int   = 2

    # Factor parameters (mirrors StrategyConfig)
    ema_fast:           int   = 9
    ema_slow:           int   = 21
    rsi_period:         int   = 14
    adx_period:         int   = 14
    bb_period:          int   = 20
    bb_std:             float = 2.0
    supertrend_period:  int   = 7
    supertrend_factor:  float = 3.0
    roc_period:         int   = 5
    vol_avg_period:     int   = 20
    or_minutes:         int   = 15
    corr_lookback:      int   = 60

    # Signal thresholds
    entry_threshold:    float = 0.25
    exit_threshold:     float = 0.10

    # Risk / sizing
    capital_per_stock:  float = 50_000.0   # ₹ allocated per stock per day
    stop_loss_pct:      float = 0.006       # 0.6% move against position
    max_trades_per_day: int   = 6           # per stock, to avoid over-trading

    # Transaction costs (per side, as fraction of trade value)
    cost_per_side:      float = 0.0005     # ~0.05% covers brokerage+STT+exchange

    def to_strategy_config(self) -> StrategyConfig:
        """Convert to StrategyConfig so FactorEngine can be reused directly."""
        return StrategyConfig(
            market_open       = self.market_open,
            squareoff_time    = self.squareoff_time,
            signal_start      = self.signal_start,
            bar_minutes       = self.bar_minutes,
            ema_fast          = self.ema_fast,
            ema_slow          = self.ema_slow,
            rsi_period        = self.rsi_period,
            adx_period        = self.adx_period,
            bb_period         = self.bb_period,
            bb_std            = self.bb_std,
            supertrend_period = self.supertrend_period,
            supertrend_factor = self.supertrend_factor,
            roc_period        = self.roc_period,
            vol_avg_period    = self.vol_avg_period,
            or_minutes        = self.or_minutes,
            corr_lookback     = self.corr_lookback,
            entry_threshold   = self.entry_threshold,
            exit_threshold    = self.exit_threshold,
        )


# ─── Transaction Cost Calculator ─────────────────────────────────────────────

def compute_costs(price: float, qty: int, side: str,
                  cfg: StockConfig) -> float:
    """
    Return total transaction cost (₹) for one leg.
    side: 'buy' or 'sell'
    Includes: flat brokerage cap, STT, exchange charges, SEBI, stamp, GST.
    """
    value      = price * qty
    brokerage  = min(20.0, value * 0.0003)          # ₹20 or 0.03%
    stt        = value * 0.00025 if side == "sell" else 0.0
    exchange   = value * 0.0000297
    sebi       = value * 0.000001
    stamp      = value * 0.00003 if side == "buy" else 0.0
    gst        = brokerage * 0.18
    return brokerage + stt + exchange + sebi + stamp + gst


# ─── Synthetic Data Generator for 50 Stocks ──────────────────────────────────

def generate_stock_data(symbol: str, days: int = 40,
                        bar_minutes: int = 2,
                        base_price: float = None,
                        beta: float = 1.0,
                        seed: int = None) -> pd.DataFrame:
    """
    Regime-switching synthetic OHLCV for a single stock.
    Uses a market factor + idiosyncratic component to create
    realistic cross-stock correlation (Nifty-like universe).
    """
    rng = np.random.default_rng(seed or abs(hash(symbol)) % (2**31))
    bars_per_day = 375 // bar_minutes
    if base_price is None:
        base_price = rng.uniform(200, 4000)

    # Build timestamp index (skip weekends)
    dates = []
    trading_days = 0
    d = 0
    while trading_days < days:
        start = pd.Timestamp('2025-01-02') + pd.Timedelta(days=d)
        d += 1
        if start.weekday() >= 5:
            continue
        for b in range(bars_per_day):
            dates.append(start + pd.Timedelta(
                minutes=b * bar_minutes + 9 * 60 + 15))
        trading_days += 1

    n = len(dates)
    closes = np.empty(n)
    closes[0] = base_price

    for day in range(days):
        day_start = day * bars_per_day
        day_end   = min(day_start + bars_per_day, n)

        regime      = 'trend'
        bars_left   = rng.integers(15, 40)
        direction   = rng.choice([-1, 1])
        mean_level  = closes[day_start]

        for i in range(day_start, day_end):
            if i == day_start:
                continue
            if bars_left <= 0:
                if regime == 'trend':
                    regime = 'range'
                    bars_left = rng.integers(8, 20)
                    mean_level = closes[i - 1]
                else:
                    regime = 'trend'
                    bars_left = rng.integers(15, 40)
                    direction = rng.choice([-1, 1])

            if regime == 'trend':
                mkt_drift  = direction * 0.0010
                idio_drift = rng.normal(0, 0.0007)
                drift      = beta * mkt_drift + idio_drift
            else:
                gap   = (closes[i - 1] - mean_level) / mean_level
                drift = -0.12 * gap + rng.normal(0, 0.0008)

            closes[i] = closes[i - 1] * (1.0 + drift)
            bars_left -= 1

        if day_end < n:
            closes[day_end] = closes[day_end - 1] * (
                1.0 + rng.normal(0, 0.004))

    bar_range = rng.uniform(0.001, 0.004, n)
    high  = closes * (1.0 + bar_range * rng.uniform(0.3, 0.9, n))
    low   = closes * (1.0 - bar_range * rng.uniform(0.3, 0.9, n))
    open_ = np.empty(n)
    open_[0] = closes[0]
    open_[1:] = closes[:-1] * (1.0 + rng.normal(0, 0.0004, n - 1))
    high  = np.maximum(high, np.maximum(open_, closes))
    low   = np.minimum(low,  np.minimum(open_, closes))

    bar_pos  = np.tile(np.linspace(0, 1, bars_per_day), days)[:n]
    vol_mult = 1.0 + 1.5 * (1.0 - 4.0 * (bar_pos - 0.5) ** 2)
    volume   = (rng.integers(10_000, 500_000, n) * vol_mult).astype(float)

    return pd.DataFrame(
        {'open': open_, 'high': high, 'low': low,
         'close': closes, 'volume': volume},
        index=pd.DatetimeIndex(dates)
    )


# ─── Real Data Fetcher (yfinance) ─────────────────────────────────────────────

def fetch_yfinance_stock(symbol: str, bar_minutes: int = 2,
                          days_back: int = 7) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")

        ticker = yf.Ticker(yf_symbol(symbol))
        raw    = ticker.history(period=f"{days_back}d", interval="1m",
                                auto_adjust=True)
        if raw.empty:
            return None

        raw.columns = [c.lower() for c in raw.columns]
        raw = raw[["open", "high", "low", "close", "volume"]]
        if bar_minutes > 1:
            raw = raw.resample(f"{bar_minutes}min").agg(
                {"open":"first","high":"max","low":"min",
                 "close":"last","volume":"sum"}).dropna(subset=["close"])

        if raw.index.tzinfo is None:
            raw.index = raw.index.tz_localize("UTC")
        raw.index = raw.index.tz_convert(IST).tz_localize(None)

        # Filter to market hours only
        raw = raw[raw.index.time >= pd.Timestamp("09:15").time()]
        raw = raw[raw.index.time <= pd.Timestamp("15:30").time()]
        return raw if len(raw) > 50 else None
    except Exception as e:
        log.debug(f"yfinance {symbol}: {e}")
        return None


# ─── Per-Stock Backtest ───────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:      str
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    side:        str            # 'long' or 'short'
    entry_price: float
    exit_price:  float
    qty:         int
    entry_cost:  float
    exit_cost:   float
    exit_reason: str

    @property
    def gross_pnl(self) -> float:
        if self.side == 'long':
            return (self.exit_price - self.entry_price) * self.qty
        else:
            return (self.entry_price - self.exit_price) * self.qty

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.entry_cost - self.exit_cost

    @property
    def return_pct(self) -> float:
        invested = self.entry_price * self.qty
        return self.net_pnl / invested * 100 if invested > 0 else 0.0


def backtest_single_stock(symbol: str, df: pd.DataFrame,
                           cfg: StockConfig) -> list[Trade]:
    """Run the full factor-signal backtest on one stock's 2-min data."""
    if df is None or len(df) < 50:
        return []

    # Compute all 10 factors + composite score
    strategy_cfg = cfg.to_strategy_config()
    engine       = FactorEngine(strategy_cfg)
    try:
        signal_df = engine.compute_all(df)
    except Exception as e:
        log.warning(f"{symbol}: factor computation failed — {e}")
        return []

    trades: list[Trade] = []
    position    = None    # dict or None
    trades_today = 0
    last_date   = None

    signal_start = pd.to_datetime(cfg.signal_start).time()
    squareoff    = pd.to_datetime(cfg.squareoff_time).time()

    for ts, row in signal_df.iterrows():
        t     = ts.time()
        price = row['close']
        score = row.get('composite_score', np.nan)

        # Reset daily trade counter
        if ts.date() != last_date:
            last_date    = ts.date()
            trades_today = 0

        # Skip pre-signal bars
        if t < signal_start:
            continue

        # ── Mandatory square-off ─────────────────────────────────────────────
        if t >= squareoff:
            if position is not None:
                cost = compute_costs(price, position['qty'],
                                     'sell' if position['side'] == 'long' else 'buy',
                                     cfg)
                trades.append(Trade(
                    symbol      = symbol,
                    entry_time  = position['entry_time'],
                    exit_time   = ts,
                    side        = position['side'],
                    entry_price = position['entry_price'],
                    exit_price  = price,
                    qty         = position['qty'],
                    entry_cost  = position['entry_cost'],
                    exit_cost   = cost,
                    exit_reason = 'squareoff',
                ))
                position = None
            continue

        if np.isnan(score):
            continue

        # ── Determine desired side ────────────────────────────────────────────
        if score >= cfg.entry_threshold:
            desired = 'long'
        elif score <= -cfg.entry_threshold:
            desired = 'short'
        else:
            desired = None

        # ── Check stop-loss ───────────────────────────────────────────────────
        if position is not None:
            sl_hit = False
            if position['side'] == 'long'  and price <= position['sl_price']:
                sl_hit = True
            if position['side'] == 'short' and price >= position['sl_price']:
                sl_hit = True

            if sl_hit:
                cost = compute_costs(price, position['qty'],
                                     'sell' if position['side'] == 'long' else 'buy',
                                     cfg)
                trades.append(Trade(
                    symbol      = symbol,
                    entry_time  = position['entry_time'],
                    exit_time   = ts,
                    side        = position['side'],
                    entry_price = position['entry_price'],
                    exit_price  = price,
                    qty         = position['qty'],
                    entry_cost  = position['entry_cost'],
                    exit_cost   = cost,
                    exit_reason = 'stop_loss',
                ))
                position = None

        # ── Position management ───────────────────────────────────────────────
        if position is None:
            if desired is not None and trades_today < cfg.max_trades_per_day:
                qty  = max(1, int(cfg.capital_per_stock / price))
                cost = compute_costs(price, qty,
                                     'buy' if desired == 'long' else 'sell', cfg)
                sl   = (price * (1 - cfg.stop_loss_pct) if desired == 'long'
                        else price * (1 + cfg.stop_loss_pct))
                position = {
                    'side':       desired,
                    'entry_time': ts,
                    'entry_price': price,
                    'qty':         qty,
                    'entry_cost':  cost,
                    'sl_price':    sl,
                }
                trades_today += 1
        else:
            cur_side = position['side']
            if desired is not None and desired != cur_side:
                # Reverse (SAR)
                exit_cost = compute_costs(price, position['qty'],
                                          'sell' if cur_side == 'long' else 'buy',
                                          cfg)
                trades.append(Trade(
                    symbol      = symbol,
                    entry_time  = position['entry_time'],
                    exit_time   = ts,
                    side        = cur_side,
                    entry_price = position['entry_price'],
                    exit_price  = price,
                    qty         = position['qty'],
                    entry_cost  = position['entry_cost'],
                    exit_cost   = exit_cost,
                    exit_reason = 'signal_switch',
                ))
                # Open reverse position
                if trades_today < cfg.max_trades_per_day:
                    qty  = max(1, int(cfg.capital_per_stock / price))
                    cost = compute_costs(price, qty,
                                         'buy' if desired == 'long' else 'sell', cfg)
                    sl   = (price * (1 - cfg.stop_loss_pct) if desired == 'long'
                            else price * (1 + cfg.stop_loss_pct))
                    position = {
                        'side':        desired,
                        'entry_time':  ts,
                        'entry_price': price,
                        'qty':         qty,
                        'entry_cost':  cost,
                        'sl_price':    sl,
                    }
                    trades_today += 1
                else:
                    position = None

            elif desired is None and abs(score) < cfg.exit_threshold:
                # Score collapsed → exit to flat
                exit_cost = compute_costs(price, position['qty'],
                                          'sell' if cur_side == 'long' else 'buy',
                                          cfg)
                trades.append(Trade(
                    symbol      = symbol,
                    entry_time  = position['entry_time'],
                    exit_time   = ts,
                    side        = cur_side,
                    entry_price = position['entry_price'],
                    exit_price  = price,
                    qty         = position['qty'],
                    entry_cost  = position['entry_cost'],
                    exit_cost   = exit_cost,
                    exit_reason = 'signal_neutral',
                ))
                position = None

    return trades


# ─── Portfolio Backtest Orchestrator ─────────────────────────────────────────

class PortfolioBacktest:

    def __init__(self, cfg: StockConfig):
        self.cfg = cfg

    def run(self, stock_data: dict[str, pd.DataFrame],
             max_workers: int = 8) -> pd.DataFrame:
        """
        Run backtest across all stocks in parallel.
        Returns a DataFrame of all trades across all stocks.
        """
        all_trades = []
        symbols    = list(stock_data.keys())
        log.info(f"Running backtest on {len(symbols)} stocks ...")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(backtest_single_stock, sym,
                          stock_data[sym], self.cfg): sym
                for sym in symbols
            }
            done = 0
            for fut in as_completed(futures):
                sym    = futures[fut]
                done  += 1
                try:
                    trades = fut.result()
                    all_trades.extend(trades)
                    log.info(f"  [{done:>2}/{len(symbols)}] {sym:<14} "
                             f"{len(trades):>4} trades")
                except Exception as e:
                    log.warning(f"  [{done:>2}/{len(symbols)}] {sym}: ERROR — {e}")

        if not all_trades:
            return pd.DataFrame()

        rows = []
        for t in all_trades:
            rows.append({
                'symbol':       t.symbol,
                'side':         t.side,
                'entry_time':   t.entry_time,
                'exit_time':    t.exit_time,
                'entry_price':  t.entry_price,
                'exit_price':   t.exit_price,
                'qty':          t.qty,
                'gross_pnl':    t.gross_pnl,
                'costs':        t.entry_cost + t.exit_cost,
                'net_pnl':      t.net_pnl,
                'return_pct':   t.return_pct,
                'exit_reason':  t.exit_reason,
                'trade_date':   t.entry_time.date(),
            })
        return pd.DataFrame(rows)


# ─── Analytics ───────────────────────────────────────────────────────────────

def analyse(trades_df: pd.DataFrame, cfg: StockConfig) -> dict:
    if trades_df.empty:
        return {"error": "No trades generated"}

    t = trades_df

    # Overall stats
    n          = len(t)
    winners    = t[t['net_pnl'] > 0]
    losers     = t[t['net_pnl'] <= 0]
    win_rate   = len(winners) / n * 100
    total_net  = t['net_pnl'].sum()
    avg_win    = winners['net_pnl'].mean() if len(winners) else 0
    avg_loss   = losers['net_pnl'].mean()  if len(losers)  else 0
    pf         = (winners['net_pnl'].sum() / -losers['net_pnl'].sum()
                  if losers['net_pnl'].sum() < 0 else float('inf'))

    # Expectancy per trade
    expectancy = total_net / n

    # Daily P&L
    daily = t.groupby('trade_date')['net_pnl'].sum()
    daily_win_rate = (daily > 0).mean() * 100
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
              if daily.std() > 0 else 0)

    # Drawdown
    cum  = t['net_pnl'].cumsum()
    dd   = cum - cum.cummax()
    max_dd = dd.min()

    # Per-stock summary
    per_stock = (t.groupby('symbol')
                  .agg(trades=('net_pnl','count'),
                       net_pnl=('net_pnl','sum'),
                       win_rate=('net_pnl', lambda x: (x>0).mean()*100),
                       avg_return_pct=('return_pct','mean'))
                  .sort_values('net_pnl', ascending=False))

    # Long vs short breakdown
    by_side = t.groupby('side')['net_pnl'].agg(['count','sum','mean'])

    # Exit reason breakdown
    by_exit = t.groupby('exit_reason')['net_pnl'].agg(['count','sum','mean'])

    return {
        'total_trades':      n,
        'total_net_pnl':     round(total_net, 2),
        'win_rate_%':        round(win_rate, 1),
        'avg_win_₹':         round(avg_win, 2),
        'avg_loss_₹':        round(avg_loss, 2),
        'profit_factor':     round(pf, 3),
        'expectancy_₹':      round(expectancy, 2),
        'max_drawdown_₹':    round(max_dd, 2),
        'sharpe_annualised': round(sharpe, 3),
        'daily_win_rate_%':  round(daily_win_rate, 1),
        'total_days':        len(daily),
        'capital_deployed':  round(cfg.capital_per_stock * 50, 0),
        'per_stock':         per_stock,
        'by_side':           by_side,
        'by_exit':           by_exit,
    }


def print_report(stats: dict, top_n: int = 10):
    SEP = "=" * 68
    print(f"\n{SEP}")
    print("  NIFTY 50 INTRADAY L/S BACKTEST  —  MULTI-FACTOR COMPOSITE SCORE")
    print(SEP)
    if 'error' in stats:
        print(f"  ERROR: {stats['error']}")
        return

    print(f"  Total Trades        : {stats['total_trades']}")
    print(f"  Total Net P&L (₹)   : {stats['total_net_pnl']:>12,.0f}")
    print(f"  Win Rate            : {stats['win_rate_%']}%")
    print(f"  Avg Win  (₹)        : {stats['avg_win_₹']:>12,.0f}")
    print(f"  Avg Loss (₹)        : {stats['avg_loss_₹']:>12,.0f}")
    print(f"  Profit Factor       : {stats['profit_factor']}")
    print(f"  Expectancy / trade  : ₹{stats['expectancy_₹']:,.0f}")
    print(f"  Max Drawdown (₹)    : {stats['max_drawdown_₹']:>12,.0f}")
    print(f"  Sharpe (ann.)       : {stats['sharpe_annualised']}")
    print(f"  Daily Win Rate      : {stats['daily_win_rate_%']}%  "
          f"({stats['total_days']} trading days)")
    print(f"  Capital Deployed    : ₹{stats['capital_deployed']:,.0f}  "
          f"(₹50k × 50 stocks)")
    print()

    print("  ── By Side ─────────────────────────────────────────────────")
    print(f"  {'Side':<10} {'Trades':>8} {'Net P&L':>12} {'Avg/trade':>10}")
    for side, row in stats['by_side'].iterrows():
        print(f"  {side:<10} {int(row['count']):>8} "
              f"{row['sum']:>12,.0f}  {row['mean']:>10,.0f}")

    print()
    print("  ── By Exit Reason ──────────────────────────────────────────")
    print(f"  {'Reason':<18} {'Trades':>8} {'Net P&L':>12} {'Avg/trade':>10}")
    for reason, row in stats['by_exit'].iterrows():
        print(f"  {reason:<18} {int(row['count']):>8} "
              f"{row['sum']:>12,.0f}  {row['mean']:>10,.0f}")

    print()
    print(f"  ── Top {top_n} Stocks by Net P&L ─────────────────────────────")
    print(f"  {'Symbol':<14} {'Trades':>7} {'Net P&L':>10} "
          f"{'Win%':>7} {'Avg Ret%':>9}")
    top    = stats['per_stock'].head(top_n)
    bottom = stats['per_stock'].tail(top_n)
    for sym, row in top.iterrows():
        print(f"  {sym:<14} {int(row['trades']):>7} {row['net_pnl']:>10,.0f} "
              f"{row['win_rate']:>7.1f} {row['avg_return_pct']:>9.3f}%")

    print(f"\n  ── Bottom {top_n} Stocks ────────────────────────────────────")
    print(f"  {'Symbol':<14} {'Trades':>7} {'Net P&L':>10} "
          f"{'Win%':>7} {'Avg Ret%':>9}")
    for sym, row in bottom.iterrows():
        print(f"  {sym:<14} {int(row['trades']):>7} {row['net_pnl']:>10,.0f} "
              f"{row['win_rate']:>7.1f} {row['avg_return_pct']:>9.3f}%")
    print(SEP)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Nifty 50 intraday L/S backtest using multi-factor composite score")
    p.add_argument("--synthetic",    action="store_true",
                   help="Use synthetic regime-switching data (default if no source given)")
    p.add_argument("--days",         type=int,   default=40,
                   help="Days of synthetic data per stock")
    p.add_argument("--data-dir",     type=str,   default=None,
                   help="Folder containing <SYMBOL>.csv files (real data)")
    p.add_argument("--symbols",      type=str,   default=None,
                   help="Comma-separated subset of symbols (default: all 50)")
    p.add_argument("--bar-minutes",  type=int,   default=2)
    p.add_argument("--capital",      type=float, default=50_000,
                   help="₹ capital per stock per day")
    p.add_argument("--entry-thr",    type=float, default=0.25)
    p.add_argument("--exit-thr",     type=float, default=0.10)
    p.add_argument("--stop-pct",     type=float, default=0.006)
    p.add_argument("--output-dir",   type=str,   default="results_stocks")
    p.add_argument("--workers",      type=int,   default=8,
                   help="Parallel workers for per-stock factor computation")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = StockConfig(
        bar_minutes       = args.bar_minutes,
        capital_per_stock = args.capital,
        entry_threshold   = args.entry_thr,
        exit_threshold    = args.exit_thr,
        stop_loss_pct     = args.stop_pct,
    )

    symbols = (args.symbols.split(",") if args.symbols
               else NIFTY50_SYMBOLS)

    # ── Load data ──────────────────────────────────────────────────────────
    stock_data: dict[str, pd.DataFrame] = {}

    if args.data_dir:
        log.info(f"Loading CSVs from {args.data_dir} ...")
        for sym in symbols:
            path = os.path.join(args.data_dir, f"{sym}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.columns = [c.lower() for c in df.columns]
                stock_data[sym] = df
                log.info(f"  Loaded {sym}: {len(df)} bars")
            else:
                log.warning(f"  {sym}: CSV not found, skipping")

    elif not args.synthetic:
        log.info("Fetching real data via yfinance (last 7 days) ...")
        log.info("Note: for longer history use --data-dir with broker CSVs")
        for sym in symbols:
            df = fetch_yfinance_stock(sym, args.bar_minutes)
            if df is not None:
                stock_data[sym] = df
                log.info(f"  Fetched {sym}: {len(df)} bars")
            else:
                log.warning(f"  {sym}: no data, skipping")

    if not stock_data:
        log.info("Generating synthetic data for all symbols ...")
        betas = np.random.default_rng(42).uniform(0.6, 1.4, len(symbols))
        for sym, beta in zip(symbols, betas):
            stock_data[sym] = generate_stock_data(
                sym, days=args.days, bar_minutes=args.bar_minutes,
                beta=float(beta))
        log.info(f"Generated synthetic data: {len(stock_data)} stocks × "
                 f"{args.days} days × {375 // args.bar_minutes} bars/day")

    # ── Run backtest ───────────────────────────────────────────────────────
    bt     = PortfolioBacktest(cfg)
    trades = bt.run(stock_data, max_workers=args.workers)

    stats  = analyse(trades, cfg)
    print_report(stats)

    # ── Save outputs ───────────────────────────────────────────────────────
    if not trades.empty:
        trades_path = os.path.join(args.output_dir, "stock_trades.csv")
        trades.to_csv(trades_path, index=False)
        print(f"\n  Trades saved → {trades_path}")

        # Per-stock summary CSV
        stock_path = os.path.join(args.output_dir, "per_stock_summary.csv")
        stats['per_stock'].to_csv(stock_path)
        print(f"  Per-stock summary → {stock_path}")

        # JSON summary (without DataFrames)
        json_stats = {k: v for k, v in stats.items()
                      if not isinstance(v, pd.DataFrame)}
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(json_stats, f, indent=2, default=str)
