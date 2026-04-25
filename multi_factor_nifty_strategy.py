"""
Multi-Factor Intraday Nifty ATM Option Selling Strategy
========================================================

Theoretical Basis: Fundamental Law of Active Management
    IR = IC × √N
    Combining N independent signals, each with small IC, beats one strong signal.

Strategy Logic:
    - Compute 10 independent technical factors on Nifty 5-min OHLCV data
    - Each factor scores -1.0 (bearish) to +1.0 (bullish) continuously
    - Correlation-adjust weights to maximize effective N (independent signals)
    - Composite Score > +threshold → Sell ATM Put  (market expected to stay up)
    - Composite Score < -threshold → Sell ATM Call (market expected to stay down)
    - Composite Score near 0 → Stay flat / close open position

Position Management:
    - Only ONE naked short position at a time (call OR put, never both)
    - Switch triggered when score crosses threshold in opposite direction
    - Stop-loss: underlying moves > stop_pct against position, or premium 2× entry
    - Square-off: 15:15 IST mandatory

Factors:
    F1  EMA Crossover      : 9-EMA vs 21-EMA slope and crossover
    F2  VWAP Deviation     : Price position relative to VWAP
    F3  RSI Momentum       : RSI(14) normalized around 50
    F4  ADX Directional    : ADX-weighted +DI/-DI imbalance
    F5  Supertrend         : 7-period ATR-based supertrend (factor=3)
    F6  Bollinger Position : Normalised position within Bollinger Bands
    F7  Rate of Change     : 5-bar ROC (price momentum)
    F8  Volume Ratio       : Buy/sell volume asymmetry vs rolling average
    F9  Opening Range      : Price relative to opening 30-min high/low
    F10 Candle Strength    : Body-to-range ratio with direction
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    # Session
    market_open:        str   = "09:15"
    market_close:       str   = "15:30"
    squareoff_time:     str   = "15:15"
    signal_start:       str   = "09:45"   # wait for opening range to form

    # Factor parameters
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
    or_minutes:         int   = 30         # opening range duration

    # Signal thresholds
    entry_threshold:    float = 0.25       # |score| > this to enter/switch
    exit_threshold:     float = 0.10       # |score| < this to go flat

    # Correlation adjustment
    corr_lookback:      int   = 60         # bars for rolling correlation matrix
    min_effective_n:    float = 3.0        # warn if effective N drops below this

    # Risk management
    stop_loss_pct:      float = 0.008      # 0.8% move in underlying → exit
    premium_sl_mult:    float = 2.0        # exit if premium > entry_premium × this
    lot_size:           int   = 25         # Nifty lot size

    # Option approximation (Black-Scholes delta hedge simulation)
    atm_delta:          float = 0.50       # ATM option delta approximation
    atm_vega_pct:       float = 0.0040     # approx vega per 1% IV change (of spot)
    iv_mean:            float = 0.15       # baseline annualised IV


# ─── Factor Computation ───────────────────────────────────────────────────────

class FactorEngine:
    """
    Computes 10 technical factors. Each returns a continuous score in [-1, +1].
    Positive = bullish (favour selling Put), Negative = bearish (favour selling Call).
    """

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df must have columns: open, high, low, close, volume (5-min bars).
        Returns df with columns f1..f10 and composite_score appended.
        """
        df = df.copy()
        df = self._f1_ema_crossover(df)
        df = self._f2_vwap_deviation(df)
        df = self._f3_rsi_momentum(df)
        df = self._f4_adx_directional(df)
        df = self._f5_supertrend(df)
        df = self._f6_bollinger_position(df)
        df = self._f7_rate_of_change(df)
        df = self._f8_volume_ratio(df)
        df = self._f9_opening_range(df)
        df = self._f10_candle_strength(df)
        df = self._composite_score(df)
        return df

    # F1 – EMA Crossover -------------------------------------------------------
    def _f1_ema_crossover(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = df['close'].ewm(span=self.cfg.ema_fast, adjust=False).mean()
        slow = df['close'].ewm(span=self.cfg.ema_slow, adjust=False).mean()
        gap  = (fast - slow) / slow                    # normalised gap
        # Sigmoid-like squash so extreme gaps don't dominate
        df['f1'] = np.tanh(gap * 100)
        return df

    # F2 – VWAP Deviation ------------------------------------------------------
    def _f2_vwap_deviation(self, df: pd.DataFrame) -> pd.DataFrame:
        # Reset VWAP each trading day
        df['_date'] = df.index.date
        tp          = (df['high'] + df['low'] + df['close']) / 3
        df['_tp_vol'] = tp * df['volume']
        df['_cum_vol'] = df.groupby('_date')['volume'].cumsum()
        df['_cum_tpvol'] = df.groupby('_date')['_tp_vol'].cumsum()
        df['vwap']  = df['_cum_tpvol'] / df['_cum_vol']

        dev = (df['close'] - df['vwap']) / df['vwap']
        df['f2'] = np.tanh(dev * 200)

        df.drop(columns=['_date', '_tp_vol', '_cum_vol', '_cum_tpvol'], inplace=True)
        return df

    # F3 – RSI Momentum --------------------------------------------------------
    def _f3_rsi_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df['close'].diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_g = gain.ewm(span=self.cfg.rsi_period, adjust=False).mean()
        avg_l = loss.ewm(span=self.cfg.rsi_period, adjust=False).mean()
        rs    = avg_g / avg_l.replace(0, np.nan)
        rsi   = 100 - 100 / (1 + rs)
        # Map [0,100] → [-1,+1], centred at 50; oversold/overbought regions squashed
        df['f3'] = np.tanh((rsi - 50) / 20)
        return df

    # F4 – ADX Directional Movement --------------------------------------------
    def _f4_adx_directional(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.cfg.adx_period
        high, low, close = df['high'], df['low'], df['close']

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)

        plus_dm  = np.where((high.diff() > 0) & (high.diff() > (-low.diff())),
                             high.diff(), 0.0)
        minus_dm = np.where((low.diff() < 0) & ((-low.diff()) > high.diff()),
                             -low.diff(), 0.0)

        atr      = pd.Series(tr).ewm(span=p, adjust=False).mean()
        plus_di  = 100 * pd.Series(plus_dm).ewm(span=p, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=p, adjust=False).mean() / atr
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx      = dx.ewm(span=p, adjust=False).mean() / 100   # 0-1 scale

        # Direction weighted by ADX strength
        di_diff = (plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        df['f4'] = (di_diff * adx).clip(-1, 1)

        df['f4'] = df['f4'].values
        return df

    # F5 – Supertrend ----------------------------------------------------------
    def _f5_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        p, mult = self.cfg.supertrend_period, self.cfg.supertrend_factor
        high, low, close = df['high'], df['low'], df['close']

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=p, adjust=False).mean()

        hl2  = (high + low) / 2
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr

        st     = pd.Series(np.nan, index=df.index)
        trend  = pd.Series(1, index=df.index)   # 1=bullish, -1=bearish

        for i in range(1, len(df)):
            prev_close = close.iloc[i - 1]
            prev_st    = st.iloc[i - 1] if not np.isnan(st.iloc[i - 1]) else lower.iloc[i]

            final_upper = upper.iloc[i] if upper.iloc[i] < prev_st else prev_st
            final_lower = lower.iloc[i] if lower.iloc[i] > prev_st else prev_st

            if trend.iloc[i - 1] == 1:
                st.iloc[i] = final_lower
                trend.iloc[i] = -1 if close.iloc[i] < st.iloc[i] else 1
            else:
                st.iloc[i] = final_upper
                trend.iloc[i] = 1 if close.iloc[i] > st.iloc[i] else -1

        # Score: trend direction modulated by distance from supertrend line
        dist = ((close - st) / st).clip(-0.01, 0.01) / 0.01   # normalised ±1
        df['f5'] = (trend * dist.abs()).clip(-1, 1)
        return df

    # F6 – Bollinger Band Position ---------------------------------------------
    def _f6_bollinger_position(self, df: pd.DataFrame) -> pd.DataFrame:
        mid   = df['close'].rolling(self.cfg.bb_period).mean()
        std   = df['close'].rolling(self.cfg.bb_period).std()
        upper = mid + self.cfg.bb_std * std
        lower = mid - self.cfg.bb_std * std

        # Position in band: 0 = lower band, 1 = upper band → centre at 0.5 → shift to [-1,+1]
        pos = (df['close'] - lower) / (upper - lower).replace(0, np.nan)
        df['f6'] = (pos * 2 - 1).clip(-1, 1)
        return df

    # F7 – Rate of Change ------------------------------------------------------
    def _f7_rate_of_change(self, df: pd.DataFrame) -> pd.DataFrame:
        roc = df['close'].pct_change(self.cfg.roc_period)
        df['f7'] = np.tanh(roc * 100)
        return df

    # F8 – Volume Ratio --------------------------------------------------------
    def _f8_volume_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        # Separate up-volume and down-volume; ratio vs rolling average
        up_vol   = df['volume'].where(df['close'] >= df['close'].shift(), 0)
        down_vol = df['volume'].where(df['close'] <  df['close'].shift(), 0)

        roll     = self.cfg.vol_avg_period
        avg_up   = up_vol.rolling(roll).mean().replace(0, 1)
        avg_down = down_vol.rolling(roll).mean().replace(0, 1)

        # Relative buy pressure
        ratio    = (up_vol / avg_up - down_vol / avg_down)
        df['f8'] = np.tanh(ratio / 2)
        return df

    # F9 – Opening Range Breakout ----------------------------------------------
    def _f9_opening_range(self, df: pd.DataFrame) -> pd.DataFrame:
        df['_date'] = df.index.date
        or_end_time = pd.to_timedelta(self.cfg.market_open) + \
                      pd.to_timedelta(f"{self.cfg.or_minutes}min")

        def _or_for_day(g):
            g = g.copy()
            cutoff = g.index[0].normalize() + or_end_time
            or_bars = g[g.index <= cutoff]
            if or_bars.empty:
                g['_or_high'] = np.nan
                g['_or_low']  = np.nan
            else:
                g['_or_high'] = or_bars['high'].max()
                g['_or_low']  = or_bars['low'].min()
            return g

        df = df.groupby('_date', group_keys=False).apply(_or_for_day)

        or_mid   = (df['_or_high'] + df['_or_low']) / 2
        or_range = (df['_or_high'] - df['_or_low']).replace(0, np.nan)

        # Score = how far price has broken above/below OR midpoint, normalised by OR range
        score = (df['close'] - or_mid) / or_range
        df['f9'] = score.clip(-1, 1)

        df.drop(columns=['_date', '_or_high', '_or_low'], inplace=True)
        return df

    # F10 – Candle Strength ----------------------------------------------------
    def _f10_candle_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        body       = df['close'] - df['open']
        total_rng  = (df['high'] - df['low']).replace(0, np.nan)
        body_ratio = body / total_rng          # +1 strong bull candle, -1 strong bear
        df['f10']  = body_ratio.clip(-1, 1)
        return df

    # Composite Score ----------------------------------------------------------
    def _composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine factors using correlation-adjusted equal weights.

        Steps follow the article's 11-step alpha combination logic:
          1. Compute rolling pairwise correlation matrix of factors
          2. Effective N = (sum of all correlations)^-1 per factor → weight
          3. Normalise weights to sum to 1
          4. Composite = weighted sum of factor scores

        This down-weights factors that are highly correlated with others,
        preserving the independent information each contributes.
        """
        factor_cols = [f'f{i}' for i in range(1, 11)]
        F = df[factor_cols].copy()

        n_factors = len(factor_cols)
        lookback  = self.cfg.corr_lookback
        scores    = []

        for idx in range(len(F)):
            start = max(0, idx - lookback + 1)
            window = F.iloc[start:idx + 1]

            if window.dropna().shape[0] < 10:
                # Not enough history: equal weights
                row_vals = F.iloc[idx].values
                if np.isnan(row_vals).any():
                    scores.append(np.nan)
                else:
                    scores.append(float(np.nanmean(row_vals)))
                continue

            corr = window.corr()
            # Effective weight for factor i = 1 / sum of |correlations| with others
            raw_weights = np.array([
                1.0 / (corr[col].abs().sum() - 1.0 + 1e-8)
                for col in factor_cols
            ])
            raw_weights = np.maximum(raw_weights, 0)
            w_sum = raw_weights.sum()
            if w_sum < 1e-8:
                weights = np.ones(n_factors) / n_factors
            else:
                weights = raw_weights / w_sum

            row_vals = F.iloc[idx].values
            if np.isnan(row_vals).any():
                # Use only available factors
                valid = ~np.isnan(row_vals)
                w_valid = weights[valid] / weights[valid].sum()
                scores.append(float(np.dot(w_valid, row_vals[valid])))
            else:
                scores.append(float(np.dot(weights, row_vals)))

        df['composite_score'] = scores
        df['composite_score'] = df['composite_score'].clip(-1, 1)
        return df


# ─── Option Premium Approximation ─────────────────────────────────────────────

class OptionPricer:
    """
    Lightweight ATM option premium approximation without requiring full B-S.
    ATM premium ≈ 0.4 × σ × S × √(T)
    where σ = daily vol, S = spot, T = fraction of day remaining.
    Used only for P&L simulation; real deployment uses live option chain quotes.
    """

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def atm_premium(self, spot: float, minutes_to_expiry: int,
                    iv_annual: float = None) -> float:
        if iv_annual is None:
            iv_annual = self.cfg.iv_mean
        T = max(minutes_to_expiry / (375), 1 / 375)  # fraction of trading day
        daily_vol = iv_annual / np.sqrt(252)
        premium = 0.4 * daily_vol * spot * np.sqrt(T * 252)
        return max(premium, 1.0)


# ─── Position & Trade State ───────────────────────────────────────────────────

@dataclass
class Position:
    side:           str           # 'short_call' or 'short_put'
    entry_time:     pd.Timestamp  = None
    entry_spot:     float         = 0.0
    entry_premium:  float         = 0.0
    lots:           int           = 1
    stop_spot:      float         = 0.0   # spot price stop level
    pnl:            float         = 0.0
    is_open:        bool          = True


@dataclass
class TradeLog:
    trades: list = field(default_factory=list)

    def record(self, entry_time, exit_time, side, entry_spot, exit_spot,
               entry_premium, exit_premium, lots, exit_reason):
        gross_pnl = (entry_premium - exit_premium) * lots * 25
        self.trades.append({
            'entry_time':     entry_time,
            'exit_time':      exit_time,
            'side':           side,
            'entry_spot':     entry_spot,
            'exit_spot':      exit_spot,
            'entry_premium':  entry_premium,
            'exit_premium':   exit_premium,
            'lots':           lots,
            'gross_pnl':      gross_pnl,
            'exit_reason':    exit_reason,
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)


# ─── Strategy Engine ──────────────────────────────────────────────────────────

class MultiFactorNiftyStrategy:
    """
    Main strategy loop: signal → position → exit logic.
    """

    def __init__(self, cfg: StrategyConfig = None):
        self.cfg     = cfg or StrategyConfig()
        self.factors = FactorEngine(self.cfg)
        self.pricer  = OptionPricer(self.cfg)

    def run(self, raw_df: pd.DataFrame, lots: int = 1,
            iv_series: pd.Series = None) -> tuple[pd.DataFrame, TradeLog]:
        """
        raw_df : DatetimeIndex, columns [open, high, low, close, volume]
        iv_series : optional per-bar annualised IV (if None uses cfg.iv_mean)
        Returns (signal_df, trade_log)
        """
        df       = self.factors.compute_all(raw_df)
        log      = TradeLog()
        position: Optional[Position] = None

        signal_start = pd.to_datetime(self.cfg.signal_start).time()
        squareoff    = pd.to_datetime(self.cfg.squareoff_time).time()

        for ts, row in df.iterrows():
            t = ts.time()

            # Skip bars before signal start
            if t < signal_start:
                continue

            score  = row['composite_score']
            spot   = row['close']
            iv     = float(iv_series.loc[ts]) if iv_series is not None else self.cfg.iv_mean

            # Minutes remaining to close
            close_dt = pd.Timestamp(ts.date().isoformat() + ' ' + self.cfg.squareoff_time)
            mins_left = max(int((close_dt - ts).total_seconds() / 60), 1)
            premium_now = self.pricer.atm_premium(spot, mins_left, iv)

            # ── Mandatory square-off ──────────────────────────────────────────
            if t >= squareoff and position is not None and position.is_open:
                position.pnl = (position.entry_premium - premium_now) * lots * self.cfg.lot_size
                log.record(position.entry_time, ts, position.side,
                           position.entry_spot, spot,
                           position.entry_premium, premium_now,
                           lots, 'squareoff')
                position = None
                continue

            if t >= squareoff:
                continue

            # ── Check stop-loss on open position ──────────────────────────────
            if position is not None and position.is_open:
                sl_hit = False
                if position.side == 'short_call':
                    # Stop if spot rises above entry_spot + stop_pct
                    if spot >= position.stop_spot:
                        sl_hit = True
                elif position.side == 'short_put':
                    # Stop if spot falls below entry_spot - stop_pct
                    if spot <= position.stop_spot:
                        sl_hit = True

                # Premium-based stop (2× entry premium)
                if premium_now >= position.entry_premium * self.cfg.premium_sl_mult:
                    sl_hit = True

                if sl_hit:
                    log.record(position.entry_time, ts, position.side,
                               position.entry_spot, spot,
                               position.entry_premium, premium_now,
                               lots, 'stop_loss')
                    position = None

            # ── Determine desired position from composite score ───────────────
            desired = self._desired_position(score)

            # ── No position: enter if signal strong enough ────────────────────
            if position is None:
                if desired == 'short_call':
                    position = self._open_position('short_call', ts, spot, premium_now, lots)
                elif desired == 'short_put':
                    position = self._open_position('short_put', ts, spot, premium_now, lots)
                continue

            # ── Open position: switch if signal flips convincingly ────────────
            if position.is_open:
                if desired is not None and desired != position.side:
                    # Close current
                    log.record(position.entry_time, ts, position.side,
                               position.entry_spot, spot,
                               position.entry_premium, premium_now,
                               lots, 'signal_switch')
                    # Open new (opposite)
                    position = self._open_position(desired, ts, spot, premium_now, lots)

                elif desired is None and abs(score) < self.cfg.exit_threshold:
                    # Score collapsed to neutral: go flat
                    log.record(position.entry_time, ts, position.side,
                               position.entry_spot, spot,
                               position.entry_premium, premium_now,
                               lots, 'signal_neutral')
                    position = None

        return df, log

    def _desired_position(self, score: float) -> Optional[str]:
        if np.isnan(score):
            return None
        if score >= self.cfg.entry_threshold:
            return 'short_put'    # bullish → sell put
        if score <= -self.cfg.entry_threshold:
            return 'short_call'   # bearish → sell call
        return None

    def _open_position(self, side: str, ts, spot: float,
                       premium: float, lots: int) -> Position:
        sl_spot = (
            spot * (1 + self.cfg.stop_loss_pct)
            if side == 'short_call'
            else spot * (1 - self.cfg.stop_loss_pct)
        )
        return Position(
            side          = side,
            entry_time    = ts,
            entry_spot    = spot,
            entry_premium = premium,
            lots          = lots,
            stop_spot     = sl_spot,
            is_open       = True,
        )


# ─── Backtest Analytics ───────────────────────────────────────────────────────

class BacktestAnalytics:

    @staticmethod
    def summarise(trade_log: TradeLog, signal_df: pd.DataFrame) -> dict:
        trades = trade_log.to_dataframe()
        if trades.empty:
            return {"error": "No trades generated"}

        total_pnl      = trades['gross_pnl'].sum()
        winning        = trades[trades['gross_pnl'] > 0]
        losing         = trades[trades['gross_pnl'] <= 0]
        win_rate       = len(winning) / len(trades) * 100
        avg_win        = winning['gross_pnl'].mean() if len(winning) > 0 else 0
        avg_loss       = losing['gross_pnl'].mean()  if len(losing)  > 0 else 0
        profit_factor  = (winning['gross_pnl'].sum() / (-losing['gross_pnl'].sum())
                          if losing['gross_pnl'].sum() < 0 else np.inf)

        # Drawdown
        cumulative     = trades['gross_pnl'].cumsum()
        rolling_max    = cumulative.cummax()
        drawdown       = cumulative - rolling_max
        max_drawdown   = drawdown.min()

        # Sharpe (daily)
        daily_pnl      = trades.groupby(trades['entry_time'].dt.date)['gross_pnl'].sum()
        sharpe         = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
                          if daily_pnl.std() > 0 else 0)

        # Factor IC diagnostic (correlation of each factor score with next-bar return)
        ret = signal_df['close'].pct_change().shift(-1)
        ic_values = {}
        for col in [f'f{i}' for i in range(1, 11)]:
            if col in signal_df.columns:
                ic_values[col] = signal_df[col].corr(ret)

        avg_ic    = np.nanmean(list(ic_values.values()))
        eff_n     = len(ic_values)
        theo_ir   = avg_ic * np.sqrt(eff_n)

        return {
            'total_trades':    len(trades),
            'total_pnl':       round(total_pnl, 2),
            'win_rate_%':      round(win_rate, 1),
            'avg_win':         round(avg_win, 2),
            'avg_loss':        round(avg_loss, 2),
            'profit_factor':   round(profit_factor, 3),
            'max_drawdown':    round(max_drawdown, 2),
            'sharpe_annualised': round(sharpe, 3),
            'exit_reasons':    trades['exit_reason'].value_counts().to_dict(),
            'factor_ICs':      {k: round(v, 4) for k, v in ic_values.items()},
            'avg_IC':          round(avg_ic, 4),
            'effective_N':     eff_n,
            'theoretical_IR':  round(theo_ir, 4),
        }

    @staticmethod
    def print_report(summary: dict):
        print("\n" + "=" * 60)
        print("  MULTI-FACTOR NIFTY ATM OPTION SELLING — BACKTEST REPORT")
        print("=" * 60)
        if 'error' in summary:
            print(f"  ERROR: {summary['error']}")
            return

        print(f"  Total Trades       : {summary['total_trades']}")
        print(f"  Total P&L (₹)      : {summary['total_pnl']:,.0f}")
        print(f"  Win Rate           : {summary['win_rate_%']}%")
        print(f"  Avg Win (₹)        : {summary['avg_win']:,.0f}")
        print(f"  Avg Loss (₹)       : {summary['avg_loss']:,.0f}")
        print(f"  Profit Factor      : {summary['profit_factor']}")
        print(f"  Max Drawdown (₹)   : {summary['max_drawdown']:,.0f}")
        print(f"  Sharpe (ann.)      : {summary['sharpe_annualised']}")
        print()
        print("  Exit Reasons:")
        for reason, count in summary['exit_reasons'].items():
            print(f"    {reason:<20}: {count}")
        print()
        print("  Factor Diagnostics (IC = correlation with next-bar return):")
        factor_names = {
            'f1': 'EMA Crossover',  'f2': 'VWAP Deviation',
            'f3': 'RSI Momentum',   'f4': 'ADX Directional',
            'f5': 'Supertrend',     'f6': 'Bollinger Position',
            'f7': 'Rate of Change', 'f8': 'Volume Ratio',
            'f9': 'Opening Range',  'f10': 'Candle Strength',
        }
        for k, v in summary['factor_ICs'].items():
            print(f"    {factor_names.get(k, k):<22}: IC = {v:+.4f}")
        print()
        print(f"  Average IC         : {summary['avg_IC']:+.4f}")
        print(f"  Effective N        : {summary['effective_N']}")
        print(f"  Theoretical IR     : {summary['theoretical_IR']:.4f}  "
              f"(= IC × √N = {summary['avg_IC']:.4f} × √{summary['effective_N']})")
        print("=" * 60)


# ─── Data Generator (for testing without live data) ───────────────────────────

def generate_synthetic_nifty(days: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic 5-minute Nifty-like OHLCV data for strategy testing.
    Replace with actual data from broker API, NSE, or data vendor in production.
    """
    rng   = np.random.default_rng(seed)
    bars_per_day = 75     # 09:15 to 15:30 = 375 min / 5 = 75 bars
    total = days * bars_per_day

    dates = []
    for d in range(days):
        start = pd.Timestamp('2025-01-02') + pd.Timedelta(days=d * 1)
        if start.weekday() >= 5:
            continue
        for b in range(bars_per_day):
            dates.append(start + pd.Timedelta(minutes=b * 5 + 9 * 60 + 15))

    n = len(dates)
    spot = 22000.0
    closes = [spot]
    for _ in range(n - 1):
        drift = rng.normal(0, 0.001)
        closes.append(closes[-1] * (1 + drift))

    closes = np.array(closes)
    high   = closes * (1 + rng.uniform(0.0002, 0.0015, n))
    low    = closes * (1 - rng.uniform(0.0002, 0.0015, n))
    open_  = np.roll(closes, 1)
    open_[0] = closes[0]
    volume = rng.integers(50_000, 300_000, n).astype(float)

    df = pd.DataFrame({
        'open':   open_,
        'high':   high,
        'low':    low,
        'close':  closes,
        'volume': volume,
    }, index=pd.DatetimeIndex(dates))
    return df


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def _parse_args():
    import argparse, os
    p = argparse.ArgumentParser(description="Multi-Factor Nifty ATM Option Selling Strategy")
    p.add_argument("--mode",             choices=["backtest", "signal"], default="backtest")
    p.add_argument("--days",             type=int,   default=40,      help="Synthetic backtest days")
    p.add_argument("--seed",             type=int,   default=42,      help="Random seed for synthetic data")
    p.add_argument("--data-csv",         type=str,   default=None,    help="Path to real OHLCV CSV (DatetimeIndex)")
    p.add_argument("--output-dir",       type=str,   default="results", help="Directory for output files")
    p.add_argument("--lots",             type=int,   default=1)
    p.add_argument("--entry-threshold",  type=float, default=0.25)
    p.add_argument("--exit-threshold",   type=float, default=0.10)
    p.add_argument("--stop-loss-pct",    type=float, default=0.008)
    p.add_argument("--premium-sl-mult",  type=float, default=2.0)
    p.add_argument("--signal-bars",      type=int,   default=1,       help="Bars of history for --mode signal")
    return p.parse_args()


if __name__ == "__main__":
    import os, json
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = StrategyConfig(
        entry_threshold = args.entry_threshold,
        exit_threshold  = args.exit_threshold,
        stop_loss_pct   = args.stop_loss_pct,
        premium_sl_mult = args.premium_sl_mult,
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.data_csv:
        df = pd.read_csv(args.data_csv, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        required = {'open', 'high', 'low', 'close', 'volume'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        print(f"Loaded {len(df)} bars from {args.data_csv}")
    else:
        print(f"No --data-csv provided. Generating {args.days} days of synthetic data...")
        df = generate_synthetic_nifty(days=args.days, seed=args.seed)

    # ── Mode: signal (compute current signal from most recent bars) ────────────
    if args.mode == "signal":
        strategy  = MultiFactorNiftyStrategy(cfg)
        signal_df = strategy.factors.compute_all(df)
        last      = signal_df.iloc[-1]
        score     = float(last.get('composite_score', np.nan))

        if np.isnan(score):
            desired = "insufficient_data"
        elif score >= cfg.entry_threshold:
            desired = "short_put"
        elif score <= -cfg.entry_threshold:
            desired = "short_call"
        else:
            desired = "flat"

        factor_vals = {f'f{i}': round(float(last.get(f'f{i}', np.nan)), 4)
                       for i in range(1, 11)}
        factor_names = {
            'f1': 'EMA_Crossover',    'f2': 'VWAP_Deviation',
            'f3': 'RSI_Momentum',     'f4': 'ADX_Directional',
            'f5': 'Supertrend',       'f6': 'Bollinger_Position',
            'f7': 'Rate_of_Change',   'f8': 'Volume_Ratio',
            'f9': 'Opening_Range',    'f10': 'Candle_Strength',
        }

        result = {
            "timestamp":       str(signal_df.index[-1]),
            "spot":            float(last['close']),
            "composite_score": round(score, 4),
            "signal":          desired,
            "factors":         {factor_names[k]: v for k, v in factor_vals.items()},
        }

        out_path = os.path.join(args.output_dir, "current_signal.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(json.dumps(result, indent=2))
        print(f"\nSignal written to: {out_path}")

    # ── Mode: backtest ─────────────────────────────────────────────────────────
    else:
        strategy  = MultiFactorNiftyStrategy(cfg)
        signal_df, trade_log = strategy.run(df, lots=args.lots)

        summary   = BacktestAnalytics.summarise(trade_log, signal_df)
        BacktestAnalytics.print_report(summary)

        trades_df = trade_log.to_dataframe()

        # Save outputs
        trades_path = os.path.join(args.output_dir, "trades.csv")
        summary_path = os.path.join(args.output_dir, "summary.json")
        signals_path = os.path.join(args.output_dir, "signals.csv")

        if not trades_df.empty:
            trades_df.to_csv(trades_path, index=False)
            print(f"\nTrades saved to:  {trades_path}")

        with open(summary_path, "w") as f:
            # Convert non-serializable values
            safe_summary = {k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                            for k, v in summary.items()}
            json.dump(safe_summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")

        factor_cols = ['close', 'f1', 'f2', 'f3', 'f4', 'f5',
                       'f6', 'f7', 'f8', 'f9', 'f10', 'composite_score']
        signal_df[factor_cols].to_csv(signals_path)
        print(f"Signals saved to: {signals_path}")
