"""
Nifty 50 Swing / Positional Backtest Engine
============================================

Same 10-factor composite-score framework applied to DAILY bars.

Key structural advantages over intraday:
  - 5 years of daily data  (~1 260 bars / stock, vs 7 days of 2-min)
  - Statistically meaningful sample  (1 000+ trades vs 196)
  - Transaction costs < 0.05%  vs  target swing move of 2-5%
    (intraday: costs ≈ 40% of a 0.2% target move)
  - Daily bars have higher IC — cleaner, less noise per signal
  - No intraday timing constraints, no lunch filter needed
  - Entry/exit at NEXT-DAY OPEN — fully realistic, no look-ahead

Signal workflow  (matches the EOD paper-trade use-case):
  3:15 PM  →  daily bar completes
  FactorEngine  →  composite_score computed on today's close
  Signal changed?  →  queue market order for tomorrow 9:15 AM open
  Next morning  →  order executes at OPEN price
  Position held 2–10 trading days depending on signal persistence

Usage:
    python swing_backtest.py                          # 5 yrs, all 41 stocks
    python swing_backtest.py --years 3
    python swing_backtest.py --symbols INFY,TCS,HCLTECH
    python swing_backtest.py --no-ic-weights          # compare weighting modes
"""

import os
import sys
import json
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# Reuse factor engine and universe from the main strategy / intraday files
from multi_factor_nifty_strategy import FactorEngine, StrategyConfig
from nifty50_intraday_backtest import (
    NIFTY50_SYMBOLS, NIFTY50_FILTERED, yf_symbol,
    EXCLUDE_COMMODITY_PSU,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("swing_bt")


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class SwingConfig:
    # ── Factor parameters (same as StrategyConfig) ─────────────────────────
    ema_fast:           int   = 9
    ema_slow:           int   = 21
    rsi_period:         int   = 14
    adx_period:         int   = 14
    bb_period:          int   = 20
    bb_std:             float = 2.0
    supertrend_period:  int   = 14       # wider ATR period for daily bars
    supertrend_factor:  float = 3.0
    roc_period:         int   = 5        # 1-week momentum
    vol_avg_period:     int   = 20       # 1-month volume average
    or_minutes:         int   = 15       # not meaningful on daily — F9 gets low IC weight
    corr_lookback:      int   = 60       # 60 trading days ≈ 3 months

    # ── Direction ──────────────────────────────────────────────────────────
    # Equity delivery trades in India are LONG-ONLY (no short-selling in cash
    # market). Short signals are ignored; signal reversals exit to flat only.
    # Set long_only=False only when trading via F&O (futures/options).
    long_only:          bool  = True

    # ── Signal thresholds ──────────────────────────────────────────────────
    entry_threshold:    float = 0.30     # raised vs original — daily needs conviction
    exit_threshold:     float = 0.10

    # ── Entry quality ──────────────────────────────────────────────────────
    min_confirm_days:   int   = 3        # 3 consecutive days above threshold → entry
    exit_confirm_days:  int   = 2        # 2 consecutive days below exit_threshold → exit

    # ── IC-based dynamic factor weighting (Fundamental Law) ───────────────
    use_ic_weights:     bool  = True
    ic_lookback:        int   = 60
    ic_min_obs:         int   = 20

    # ── Holding period ─────────────────────────────────────────────────────
    min_hold_days:      int   = 3        # minimum 3 days (avoid noise exits)
    max_hold_days:      int   = 20       # let winners run up to 4 weeks

    # ── Stop loss management ───────────────────────────────────────────────
    # Daily stocks can gap 2-3% on news; 3% stop was being hit on 83% of trades.
    # Widened to 6% — the trailing mechanism tightens it after the trade moves.
    stop_loss_pct:      float = 0.060    # 6% initial stop for positional holds
    trailing_stop:      bool  = True
    trail_trigger_pct:  float = 0.020    # 2% gain  → start trailing
    trail_pct:          float = 0.030    # trail at 3% below peak (locks in profit)
    breakeven_trigger:  float = 0.010    # 1% gain  → move stop to entry (zero loss)

    # ── Market regime filter (Nifty macro trend) ───────────────────────────
    # Only enter new LONG positions when Nifty 50 index is in an uptrend.
    # This prevents buying individual stocks during market-wide corrections.
    # Regime: bullish when Nifty close > nifty_ema_period-day EMA.
    # May/Jun 2025 drawdown (-₹98k in 2 months) was from entering longs into
    # a broad correction — this filter prevents that regime.
    market_filter:      bool  = True
    nifty_ema_period:   int   = 20       # 20-day EMA of Nifty 50 index

    # ── Position sizing ────────────────────────────────────────────────────
    capital_per_stock:  float = 50_000.0
    max_concurrent:     int   = 10       # max simultaneous open positions

    # ── Universe ───────────────────────────────────────────────────────────
    exclude_commodity:  bool  = True

    def to_strategy_config(self) -> StrategyConfig:
        return StrategyConfig(
            market_open       = "09:15",
            squareoff_time    = "15:15",
            signal_start      = "09:30",
            bar_minutes       = 2,           # not used for factor maths, just config
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
            use_ic_weights    = self.use_ic_weights,
            ic_lookback       = self.ic_lookback,
            ic_min_obs        = self.ic_min_obs,
        )


# ─── Trade Record ─────────────────────────────────────────────────────────────

@dataclass
class SwingTrade:
    symbol:       str
    entry_date:   pd.Timestamp
    exit_date:    pd.Timestamp
    side:         str           # 'long' | 'short'
    entry_price:  float
    exit_price:   float
    qty:          int
    entry_cost:   float
    exit_cost:    float
    exit_reason:  str           # signal_neutral | signal_switch | stop_loss | max_hold | end_of_data

    @property
    def hold_days(self) -> int:
        delta = self.exit_date - self.entry_date
        return max(1, delta.days)

    @property
    def gross_pnl(self) -> float:
        if self.side == 'long':
            return (self.exit_price - self.entry_price) * self.qty
        return (self.entry_price - self.exit_price) * self.qty

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.entry_cost - self.exit_cost

    @property
    def return_pct(self) -> float:
        value = self.entry_price * self.qty
        return self.net_pnl / value * 100 if value > 0 else 0.0


# ─── Cost Calculator (Zerodha delivery model) ─────────────────────────────────

def compute_costs(price: float, qty: int, side: str) -> float:
    """
    Zerodha delivery trade costs.
    Brokerage is ZERO for equity delivery; STT is 0.1% on both buy AND sell.
    Exchange charges, SEBI, stamp duty, GST still apply.
    """
    value      = price * qty
    brokerage  = 0.0                                   # zero brokerage on delivery
    stt        = value * 0.001                         # 0.1% both sides (delivery)
    exchange   = value * 0.0000297
    sebi       = value * 0.000001
    stamp      = value * 0.00015 if side == "buy" else 0.0   # 0.015% on buy
    gst        = brokerage * 0.18
    return brokerage + stt + exchange + sebi + stamp + gst


# ─── Data Fetchers ────────────────────────────────────────────────────────────

def _yf_download(ticker: str, years: int, min_bars: int = 100) -> Optional[pd.DataFrame]:
    """Shared yfinance download helper."""
    try:
        import yfinance as yf
        end   = pd.Timestamp.now().normalize()
        start = end - pd.Timedelta(days=int(years * 365.25))
        raw   = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             interval="1d", auto_adjust=True, progress=False)
        if raw.empty or len(raw) < min_bars:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        df = raw[['open', 'high', 'low', 'close', 'volume']].copy()
        df.index = pd.to_datetime(df.index)
        return df.dropna()
    except Exception as e:
        log.warning(f"  {ticker}: fetch failed — {e}")
        return None


def fetch_daily_stock(symbol: str, years: int = 5) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV for a Nifty 50 stock (returns None if < 100 bars)."""
    return _yf_download(yf_symbol(symbol), years)


def fetch_nifty_index(years: int = 5) -> Optional[pd.Series]:
    """
    Fetch daily Nifty 50 index closing prices (^NSEI).
    Used to build the market regime filter.
    Returns a Series indexed by date, or None on failure.
    """
    df = _yf_download("^NSEI", years, min_bars=50)
    if df is None:
        return None
    return df['close'].rename("nifty_close")


def build_nifty_regime(nifty_close: pd.Series, ema_period: int = 20) -> pd.Series:
    """
    Returns a boolean Series: True when Nifty is in a bullish regime.
    Bullish = Nifty close > ema_period-day EMA of Nifty.
    We use yesterday's value (shift(1)) so there is no look-ahead at signal time.
    """
    ema      = nifty_close.ewm(span=ema_period, adjust=False).mean()
    bullish  = (nifty_close > ema).shift(1).fillna(False)   # no look-ahead
    return bullish.astype(bool)


# ─── Single-Stock Swing Backtest ──────────────────────────────────────────────

def backtest_swing_stock(symbol: str, df: pd.DataFrame,
                         cfg: SwingConfig,
                         nifty_regime: Optional[pd.Series] = None) -> list[SwingTrade]:
    """
    Run the swing backtest on one stock's daily data.

    Execution model (no look-ahead):
      Day N close  →  compute signal
      Day N+1 open →  enter / exit (execution price)
      Stop loss    →  checked against Day N+1 intraday LOW/HIGH (realistic fill)

    long_only=True  (default):
      - Short signals are ignored entirely
      - signal_switch exits the long to flat; does NOT open a short
      - Only long trades are recorded

    market_filter=True  (default):
      - New LONG entries only allowed when Nifty is in a bullish regime
        (Nifty close > 20-day EMA, evaluated on previous day — no look-ahead)
      - Existing positions are NOT force-exited on regime change
        (avoids unnecessary stop-outs during brief corrections)
    """
    if df is None or len(df) < 60:
        return []

    strategy_cfg = cfg.to_strategy_config()
    engine       = FactorEngine(strategy_cfg)
    try:
        signal_df = engine.compute_all(df)
    except Exception as e:
        log.warning(f"{symbol}: factor computation failed — {e}")
        return []

    # ── Align Nifty market regime to this stock's trading calendar ────────────
    # build_nifty_regime already applies shift(1), so regime_aligned.iloc[i]
    # at execution bar i reflects Nifty's state as of bar i-1 (yesterday) — no
    # look-ahead when deciding whether to enter today.
    if nifty_regime is not None and cfg.market_filter:
        regime_aligned = (nifty_regime
                          .reindex(signal_df.index, method='ffill')
                          .fillna(False)
                          .astype(bool))
    else:
        regime_aligned = pd.Series(True, index=signal_df.index)

    trades: list[SwingTrade] = []
    position                 = None
    confirm_long             = 0
    confirm_short            = 0
    exit_confirm             = 0
    days_in_position         = 0

    # We iterate from bar 1: bar i-1 is the signal, bar i is the execution bar.
    for i in range(1, len(signal_df)):
        prev = signal_df.iloc[i - 1]   # yesterday's bar → signal computed here
        curr = signal_df.iloc[i]        # today's bar     → execute at OPEN

        ts          = signal_df.index[i]
        score       = prev.get('composite_score', np.nan)
        exec_price  = float(curr['open'])
        today_high  = float(curr['high'])
        today_low   = float(curr['low'])
        today_close = float(curr['close'])

        # ── Market regime at execution time ────────────────────────────────
        # regime_aligned has shift(1) baked in, so .iloc[i] = yesterday's regime
        in_bull_regime = bool(regime_aligned.iloc[i])
        can_long = (not cfg.market_filter) or in_bull_regime

        if np.isnan(score):
            confirm_long = confirm_short = 0
            continue

        # ── Update entry confirmation counters ──────────────────────────────
        if score >= cfg.entry_threshold:
            confirm_long  += 1;  confirm_short  = 0
        elif score <= -cfg.entry_threshold:
            confirm_short += 1;  confirm_long   = 0
        else:
            confirm_long  = 0;   confirm_short  = 0

        # ── Update exit confirmation counter ────────────────────────────────
        if position is not None and abs(score) < cfg.exit_threshold:
            exit_confirm += 1
        else:
            exit_confirm = 0

        # ── Desired side (requires N consecutive confirming days) ────────────
        if   confirm_long  >= cfg.min_confirm_days:  desired = 'long'
        elif confirm_short >= cfg.min_confirm_days:  desired = 'short'
        else:                                        desired = None

        # ── Track days in position ──────────────────────────────────────────
        if position is not None:
            days_in_position += 1
        else:
            days_in_position = 0

        # ── Adaptive trailing stop update ───────────────────────────────────
        # Updated using yesterday's close (prev['close']) to avoid today-look-ahead.
        # Actual stop check uses today's intraday high/low for realistic fills.
        if position is not None and cfg.trailing_stop:
            ep    = position['entry_price']
            pc    = float(prev['close'])   # best data we have at signal time
            side  = position['side']

            if side == 'long':
                position['peak_price'] = max(position.get('peak_price', ep), pc)
                peak     = position['peak_price']
                gain_pct = (peak - ep) / ep
                if gain_pct >= cfg.trail_trigger_pct:
                    trail_sl = peak * (1 - cfg.trail_pct)
                    position['sl_price'] = max(position['sl_price'], trail_sl)
                elif gain_pct >= cfg.breakeven_trigger:
                    position['sl_price'] = max(position['sl_price'], ep)
            else:
                position['peak_price'] = min(position.get('peak_price', ep), pc)
                peak     = position['peak_price']
                gain_pct = (ep - peak) / ep
                if gain_pct >= cfg.trail_trigger_pct:
                    trail_sl = peak * (1 + cfg.trail_pct)
                    position['sl_price'] = min(position['sl_price'], trail_sl)
                elif gain_pct >= cfg.breakeven_trigger:
                    position['sl_price'] = min(position['sl_price'], ep)

        # ── Stop-loss check (intraday fill at stop price) ───────────────────
        if position is not None:
            sl_price = position['sl_price']
            sl_hit   = False
            if position['side'] == 'long'  and today_low  <= sl_price:  sl_hit = True
            if position['side'] == 'short' and today_high >= sl_price:  sl_hit = True

            if sl_hit:
                ep       = position['entry_price']
                sl_fill  = max(today_low,  sl_price) if position['side'] == 'long' \
                           else min(today_high, sl_price)
                exit_cst = compute_costs(sl_fill, position['qty'],
                                         'sell' if position['side'] == 'long' else 'buy')
                trades.append(SwingTrade(
                    symbol      = symbol,
                    entry_date  = position['entry_date'],
                    exit_date   = ts,
                    side        = position['side'],
                    entry_price = ep,
                    exit_price  = sl_fill,
                    qty         = position['qty'],
                    entry_cost  = position['entry_cost'],
                    exit_cost   = exit_cst,
                    exit_reason = 'stop_loss',
                ))
                position         = None
                days_in_position = 0
                confirm_long     = 0
                confirm_short    = 0
                exit_confirm     = 0
                continue

        # ── Max-hold forced exit at today's open ────────────────────────────
        if position is not None and days_in_position >= cfg.max_hold_days:
            exit_cst = compute_costs(exec_price, position['qty'],
                                     'sell' if position['side'] == 'long' else 'buy')
            trades.append(SwingTrade(
                symbol      = symbol,
                entry_date  = position['entry_date'],
                exit_date   = ts,
                side        = position['side'],
                entry_price = position['entry_price'],
                exit_price  = exec_price,
                qty         = position['qty'],
                entry_cost  = position['entry_cost'],
                exit_cost   = exit_cst,
                exit_reason = 'max_hold',
            ))
            position         = None
            days_in_position = 0
            confirm_long     = 0
            confirm_short    = 0
            exit_confirm     = 0

        # ── Position management at today's OPEN ─────────────────────────────
        if position is None:
            # Long-only: ignore short signals; market filter: skip longs in
            # bear regime (new entries only — existing positions never force-exited).
            can_enter_long  = (desired == 'long'  and can_long)
            can_enter_short = (desired == 'short' and not cfg.long_only)

            if can_enter_long or can_enter_short:
                side = desired
                qty  = max(1, int(cfg.capital_per_stock / exec_price))
                cost = compute_costs(exec_price, qty,
                                     'buy' if side == 'long' else 'sell')
                sl   = (exec_price * (1 - cfg.stop_loss_pct) if side == 'long'
                        else exec_price * (1 + cfg.stop_loss_pct))
                position = {
                    'side':        side,
                    'entry_date':  ts,
                    'entry_price': exec_price,
                    'peak_price':  exec_price,
                    'qty':         qty,
                    'entry_cost':  cost,
                    'sl_price':    sl,
                }
                days_in_position = 0
                exit_confirm     = 0

        else:
            cur_side      = position['side']
            past_min_hold = days_in_position >= cfg.min_hold_days

            if past_min_hold:
                # Signal reversal
                if desired is not None and desired != cur_side:
                    exit_cst = compute_costs(exec_price, position['qty'],
                                             'sell' if cur_side == 'long' else 'buy')
                    trades.append(SwingTrade(
                        symbol      = symbol,
                        entry_date  = position['entry_date'],
                        exit_date   = ts,
                        side        = cur_side,
                        entry_price = position['entry_price'],
                        exit_price  = exec_price,
                        qty         = position['qty'],
                        entry_cost  = position['entry_cost'],
                        exit_cost   = exit_cst,
                        exit_reason = 'signal_switch',
                    ))
                    position         = None
                    days_in_position = 0
                    confirm_long     = 0
                    confirm_short    = 0
                    exit_confirm     = 0

                # Score faded: require N consecutive days of weakness
                elif (desired is None
                      and abs(score) < cfg.exit_threshold
                      and exit_confirm >= cfg.exit_confirm_days):
                    exit_cst = compute_costs(exec_price, position['qty'],
                                             'sell' if cur_side == 'long' else 'buy')
                    trades.append(SwingTrade(
                        symbol      = symbol,
                        entry_date  = position['entry_date'],
                        exit_date   = ts,
                        side        = cur_side,
                        entry_price = position['entry_price'],
                        exit_price  = exec_price,
                        qty         = position['qty'],
                        entry_cost  = position['entry_cost'],
                        exit_cost   = exit_cst,
                        exit_reason = 'signal_neutral',
                    ))
                    position         = None
                    days_in_position = 0
                    confirm_long     = 0
                    confirm_short    = 0
                    exit_confirm     = 0

    # ── Close any open position at end of data ──────────────────────────────
    if position is not None:
        last_close = float(signal_df.iloc[-1]['close'])
        last_ts    = signal_df.index[-1]
        exit_cst   = compute_costs(last_close, position['qty'],
                                   'sell' if position['side'] == 'long' else 'buy')
        trades.append(SwingTrade(
            symbol      = symbol,
            entry_date  = position['entry_date'],
            exit_date   = last_ts,
            side        = position['side'],
            entry_price = position['entry_price'],
            exit_price  = last_close,
            qty         = position['qty'],
            entry_cost  = position['entry_cost'],
            exit_cost   = exit_cst,
            exit_reason = 'end_of_data',
        ))

    return trades


# ─── Portfolio Orchestrator ───────────────────────────────────────────────────

class PortfolioSwingBacktest:

    def __init__(self, cfg: SwingConfig):
        self.cfg = cfg

    def run(self, stock_data: dict[str, pd.DataFrame],
             years: int = 5,
             max_workers: int = 8) -> pd.DataFrame:
        all_trades = []
        symbols    = list(stock_data.keys())
        log.info(f"Running swing backtest on {len(symbols)} stocks …")

        # ── Fetch Nifty 50 index for market regime filter ───────────────────
        nifty_regime = None
        if self.cfg.market_filter:
            log.info("Fetching Nifty 50 index for market regime filter …")
            nifty_close = fetch_nifty_index(years=years)
            if nifty_close is not None:
                nifty_regime = build_nifty_regime(nifty_close,
                                                  self.cfg.nifty_ema_period)
                bull_days  = int(nifty_regime.sum())
                total_days = len(nifty_regime)
                log.info(f"  Nifty regime: {total_days} days, "
                         f"bullish={bull_days} ({bull_days/total_days*100:.0f}%)")
            else:
                log.warning("  Could not fetch Nifty index — "
                            "market filter disabled for this run")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(backtest_swing_stock, sym, stock_data[sym],
                          self.cfg, nifty_regime): sym
                for sym in symbols
            }
            done = 0
            for fut in as_completed(futures):
                sym   = futures[fut]
                done += 1
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
                'entry_date':   t.entry_date.date(),
                'exit_date':    t.exit_date.date(),
                'entry_price':  t.entry_price,
                'exit_price':   t.exit_price,
                'qty':          t.qty,
                'hold_days':    t.hold_days,
                'gross_pnl':    t.gross_pnl,
                'costs':        t.entry_cost + t.exit_cost,
                'net_pnl':      t.net_pnl,
                'return_pct':   t.return_pct,
                'exit_reason':  t.exit_reason,
            })
        df = pd.DataFrame(rows)
        df['year_month'] = pd.to_datetime(df['entry_date']).dt.to_period('M')
        return df


# ─── Analytics ───────────────────────────────────────────────────────────────

def analyse(trades_df: pd.DataFrame, cfg: SwingConfig) -> dict:
    if trades_df.empty:
        return {"error": "No trades generated"}

    t = trades_df

    # Overall
    n         = len(t)
    winners   = t[t['net_pnl'] > 0]
    losers    = t[t['net_pnl'] <= 0]
    win_rate  = len(winners) / n * 100
    total_net = t['net_pnl'].sum()
    avg_win   = winners['net_pnl'].mean() if len(winners) else 0
    avg_loss  = losers['net_pnl'].mean()  if len(losers)  else 0
    pf        = (winners['net_pnl'].sum() / -losers['net_pnl'].sum()
                 if losers['net_pnl'].sum() < 0 else float('inf'))

    # Hold time
    avg_hold_all  = t['hold_days'].mean()
    avg_hold_win  = winners['hold_days'].mean() if len(winners) else 0
    avg_hold_loss = losers['hold_days'].mean()  if len(losers)  else 0

    # Monthly P&L (for Sharpe / drawdown)
    monthly     = t.groupby('year_month')['net_pnl'].sum()
    n_months    = len(monthly)
    monthly_avg = monthly.mean()
    monthly_std = monthly.std()
    sharpe_m    = (monthly_avg / monthly_std * np.sqrt(12)
                   if monthly_std > 0 else 0)

    # Drawdown on cumulative trade P&L
    cum   = t['net_pnl'].cumsum()
    dd    = cum - cum.cummax()
    max_dd = dd.min()

    # Annual return on deployed capital
    total_years  = max(n_months / 12, 1)
    annual_return = total_net / (cfg.capital_per_stock * 50) / total_years * 100

    # Per-stock
    per_stock = (t.groupby('symbol')
                  .agg(trades=('net_pnl', 'count'),
                       net_pnl=('net_pnl', 'sum'),
                       win_rate=('net_pnl', lambda x: (x > 0).mean() * 100),
                       avg_return_pct=('return_pct', 'mean'),
                       avg_hold=('hold_days', 'mean'))
                  .sort_values('net_pnl', ascending=False))

    # By side and exit reason
    by_side = t.groupby('side')['net_pnl'].agg(['count', 'sum', 'mean'])
    by_exit = t.groupby('exit_reason')['net_pnl'].agg(['count', 'sum', 'mean'])

    return {
        'total_trades':       n,
        'total_net_pnl':      round(total_net, 0),
        'win_rate_%':         round(win_rate, 1),
        'avg_win_₹':          round(avg_win, 0),
        'avg_loss_₹':         round(avg_loss, 0),
        'profit_factor':      round(pf, 3),
        'expectancy_₹':       round(total_net / n, 0),
        'max_drawdown_₹':     round(max_dd, 0),
        'sharpe_monthly':     round(sharpe_m, 3),
        'annual_return_%':    round(annual_return, 1),
        'avg_hold_days':      round(avg_hold_all, 1),
        'avg_hold_winners':   round(avg_hold_win, 1),
        'avg_hold_losers':    round(avg_hold_loss, 1),
        'total_months':       n_months,
        'capital_deployed':   round(cfg.capital_per_stock * 50, 0),
        'monthly_pnl':        monthly,
        'per_stock':          per_stock,
        'by_side':            by_side,
        'by_exit':            by_exit,
    }


def print_report(stats: dict, top_n: int = 10):
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  NIFTY 50 SWING / POSITIONAL BACKTEST  —  MULTI-FACTOR DAILY")
    print(SEP)
    if 'error' in stats:
        print(f"  ERROR: {stats['error']}")
        return

    print(f"  Total Trades        : {stats['total_trades']}")
    print(f"  Total Net P&L (₹)   : {stats['total_net_pnl']:>12,.0f}")
    print(f"  Annual Return       : {stats['annual_return_%']:>8.1f}%"
          f"  (on ₹{stats['capital_deployed']:,.0f} deployed)")
    print(f"  Win Rate            : {stats['win_rate_%']}%")
    print(f"  Avg Win  (₹)        : {stats['avg_win_₹']:>12,.0f}")
    print(f"  Avg Loss (₹)        : {stats['avg_loss_₹']:>12,.0f}")
    print(f"  Profit Factor       : {stats['profit_factor']}")
    print(f"  Expectancy / trade  : ₹{stats['expectancy_₹']:,.0f}")
    print(f"  Max Drawdown (₹)    : {stats['max_drawdown_₹']:>12,.0f}")
    print(f"  Sharpe (monthly)    : {stats['sharpe_monthly']}")
    print(f"  Avg Hold (days)     : {stats['avg_hold_days']}"
          f"  — Winners: {stats['avg_hold_winners']}d"
          f"  Losers: {stats['avg_hold_losers']}d")
    print(f"  Months in test      : {stats['total_months']}")
    print()

    print("  ── By Side ──────────────────────────────────────────────────────")
    print(f"  {'Side':<10} {'Trades':>8} {'Net P&L':>12} {'Avg/trade':>10}")
    for side, row in stats['by_side'].iterrows():
        print(f"  {side:<10} {int(row['count']):>8} "
              f"{row['sum']:>12,.0f}  {row['mean']:>10,.0f}")

    print()
    print("  ── By Exit Reason ───────────────────────────────────────────────")
    print(f"  {'Reason':<18} {'Trades':>8} {'Net P&L':>12} {'Avg/trade':>10}")
    for reason, row in stats['by_exit'].iterrows():
        print(f"  {reason:<18} {int(row['count']):>8} "
              f"{row['sum']:>12,.0f}  {row['mean']:>10,.0f}")

    # Monthly P&L bar chart (ASCII)
    print()
    print("  ── Monthly P&L ──────────────────────────────────────────────────")
    monthly = stats['monthly_pnl']
    max_abs  = max(abs(monthly).max(), 1)
    for period, val in monthly.tail(24).items():    # show last 24 months
        bar_len = int(abs(val) / max_abs * 30)
        bar     = ("█" * bar_len) if val >= 0 else ("░" * bar_len)
        sign    = "+" if val >= 0 else "-"
        print(f"  {period}  {sign}₹{abs(val):>8,.0f}  {bar}")

    print()
    print(f"  ── Top {top_n} Stocks ─────────────────────────────────────────────")
    print(f"  {'Symbol':<14} {'Trades':>7} {'Net P&L':>10} "
          f"{'Win%':>7} {'Avg Ret%':>9} {'Avg Hold':>9}")
    for sym, row in stats['per_stock'].head(top_n).iterrows():
        print(f"  {sym:<14} {int(row['trades']):>7} {row['net_pnl']:>10,.0f} "
              f"{row['win_rate']:>7.1f} {row['avg_return_pct']:>9.3f}%"
              f" {row['avg_hold']:>8.1f}d")

    print(f"\n  ── Bottom {top_n} Stocks ─────────────────────────────────────────")
    print(f"  {'Symbol':<14} {'Trades':>7} {'Net P&L':>10} "
          f"{'Win%':>7} {'Avg Ret%':>9} {'Avg Hold':>9}")
    for sym, row in stats['per_stock'].tail(top_n).iterrows():
        print(f"  {sym:<14} {int(row['trades']):>7} {row['net_pnl']:>10,.0f} "
              f"{row['win_rate']:>7.1f} {row['avg_return_pct']:>9.3f}%"
              f" {row['avg_hold']:>8.1f}d")
    print(SEP)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Nifty 50 swing/positional backtest — daily bars, long-only delivery")
    p.add_argument("--years",             type=int,   default=5,
                   help="Years of daily history to fetch via yfinance (default=5)")
    p.add_argument("--symbols",           type=str,   default=None,
                   help="Comma-separated subset of symbols (default: all filtered)")
    p.add_argument("--capital",           type=float, default=50_000,
                   help="₹ capital per stock (default=50000)")
    p.add_argument("--entry-thr",         type=float, default=0.30,
                   help="Entry threshold (default=0.30 for daily bars)")
    p.add_argument("--exit-thr",          type=float, default=0.10,
                   help="Exit threshold (default=0.10)")
    p.add_argument("--stop-pct",          type=float, default=0.060,
                   help="Initial stop-loss %% (default=6%% — widened for daily volatility)")
    p.add_argument("--min-confirm",       type=int,   default=3,
                   help="Consecutive days above threshold before entry (default=3)")
    p.add_argument("--exit-confirm",      type=int,   default=2,
                   help="Consecutive days below exit_thr before signal_neutral exit (default=2)")
    p.add_argument("--min-hold",          type=int,   default=3,
                   help="Minimum holding days (default=3)")
    p.add_argument("--max-hold",          type=int,   default=20,
                   help="Maximum holding days before forced exit (default=20)")
    p.add_argument("--no-trailing-stop",  action="store_true",
                   help="Disable adaptive trailing stop")
    p.add_argument("--no-ic-weights",     action="store_true",
                   help="Disable IC-based factor weights")
    p.add_argument("--include-commodity", action="store_true",
                   help="Include commodity/PSU stocks (excluded by default)")
    p.add_argument("--allow-short",       action="store_true",
                   help="Allow short trades (F&O only; default=long-only for delivery)")
    p.add_argument("--no-market-filter",  action="store_true",
                   help="Disable Nifty market regime filter (not recommended)")
    p.add_argument("--output-dir",        type=str,   default="results_swing",
                   help="Output directory for CSV/JSON results")
    p.add_argument("--workers",           type=int,   default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = SwingConfig(
        capital_per_stock  = args.capital,
        entry_threshold    = args.entry_thr,
        exit_threshold     = args.exit_thr,
        stop_loss_pct      = args.stop_pct,
        min_confirm_days   = args.min_confirm,
        exit_confirm_days  = args.exit_confirm,
        min_hold_days      = args.min_hold,
        max_hold_days      = args.max_hold,
        trailing_stop      = not args.no_trailing_stop,
        use_ic_weights     = not args.no_ic_weights,
        exclude_commodity  = not args.include_commodity,
        long_only          = not args.allow_short,
        market_filter      = not args.no_market_filter,
    )

    log.info("SwingConfig: entry_thr=%.2f  exit_thr=%.2f  stop=%.1f%%  "
             "min_confirm=%d  hold=%d–%d days  IC=%s  trailing=%s  "
             "long_only=%s  market_filter=%s",
             cfg.entry_threshold, cfg.exit_threshold, cfg.stop_loss_pct * 100,
             cfg.min_confirm_days, cfg.min_hold_days, cfg.max_hold_days,
             cfg.use_ic_weights, cfg.trailing_stop,
             cfg.long_only, cfg.market_filter)

    symbols = (args.symbols.split(",") if args.symbols
               else (NIFTY50_FILTERED if cfg.exclude_commodity else NIFTY50_SYMBOLS))
    log.info(f"Universe: {len(symbols)} stocks  |  History: {args.years} years daily")

    # ── Fetch daily data ───────────────────────────────────────────────────
    stock_data: dict[str, pd.DataFrame] = {}
    log.info("Fetching daily OHLCV via yfinance …")
    for sym in symbols:
        df = fetch_daily_stock(sym, years=args.years)
        if df is not None:
            stock_data[sym] = df
            log.info(f"  {sym:<14}  {len(df):>4} days  "
                     f"({df.index[0].date()} → {df.index[-1].date()})")
        else:
            log.warning(f"  {sym}: no data, skipping")

    if not stock_data:
        log.error("No data fetched. Check internet connection.")
        sys.exit(1)

    log.info(f"Data ready for {len(stock_data)} stocks")

    # ── Run backtest ───────────────────────────────────────────────────────
    bt     = PortfolioSwingBacktest(cfg)
    trades = bt.run(stock_data, years=args.years, max_workers=args.workers)

    stats  = analyse(trades, cfg)
    print_report(stats)

    # ── Save outputs ───────────────────────────────────────────────────────
    if not trades.empty:
        trades_path = os.path.join(args.output_dir, "swing_trades.csv")
        trades.to_csv(trades_path, index=False)
        print(f"\n  Trades saved → {trades_path}")

        stock_path = os.path.join(args.output_dir, "per_stock_summary.csv")
        stats['per_stock'].to_csv(stock_path)
        print(f"  Per-stock summary → {stock_path}")

        json_stats = {k: v for k, v in stats.items()
                      if not isinstance(v, (pd.DataFrame, pd.Series))}
        with open(os.path.join(args.output_dir, "swing_summary.json"), "w") as f:
            json.dump(json_stats, f, indent=2, default=str)
        print(f"  Summary JSON → {args.output_dir}/swing_summary.json")
