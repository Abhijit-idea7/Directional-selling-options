"""
Live Trading Adapter — Multi-Factor Nifty ATM Option Selling
=============================================================

This module bridges the strategy engine (multi_factor_nifty_strategy.py)
with a real-time data feed and broker order API.

Supports:
  - Zerodha (Kite Connect API)
  - Upstox API v2
  - Any broker exposing 5-min OHLCV + option chain

Designed for GitHub Actions scheduled workflows.
All secrets are read from environment variables (set as GitHub Secrets):

  BROKER_API_KEY        — Kite/Upstox API key
  BROKER_ACCESS_TOKEN   — Session access token (refreshed daily via auth workflow)
  BROKER_TYPE           — "zerodha" | "upstox"  (default: zerodha)
  TELEGRAM_BOT_TOKEN    — Optional: Telegram bot token for trade alerts
  TELEGRAM_CHAT_ID      — Optional: Telegram chat/channel ID for alerts
  PAPER_TRADE           — "true" | "false"  (default: true)

Usage (GitHub Actions):
  python live_trading_adapter.py

Usage (local):
  BROKER_API_KEY=xxx BROKER_ACCESS_TOKEN=yyy python live_trading_adapter.py
"""

import os
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from multi_factor_nifty_strategy import (
    MultiFactorNiftyStrategy, StrategyConfig, FactorEngine, OptionPricer
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("nifty_live")

IST = ZoneInfo("Asia/Kolkata")

# ─── Environment config ────────────────────────────────────────────────────────
BROKER_API_KEY      = os.getenv("BROKER_API_KEY", "")
BROKER_ACCESS_TOKEN = os.getenv("BROKER_ACCESS_TOKEN", "")
BROKER_TYPE         = os.getenv("BROKER_TYPE", "zerodha").lower()
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
PAPER_TRADE         = os.getenv("PAPER_TRADE", "true").lower() == "true"


# ─── Telegram Notifications ───────────────────────────────────────────────────

def send_telegram(message: str):
    """Send alert to Telegram if credentials are configured."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
                      timeout=5)
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


# ─── Broker API (Zerodha / Upstox) ────────────────────────────────────────────

class BrokerAPI:
    """
    Thin wrapper around broker SDKs. Reads credentials from environment variables.
    Set BROKER_TYPE to "zerodha" or "upstox".

    To add a new broker:
      1. Add a new branch in each method keyed by BROKER_TYPE
      2. Import the broker SDK at the top of the method (lazy import)

    Paper trade mode (PAPER_TRADE=true): orders are logged but not placed.
    """

    def __init__(self):
        self._kite = None
        self._upstox = None
        self._connected = False
        if not PAPER_TRADE:
            self._connect()

    def _connect(self):
        if BROKER_TYPE == "zerodha":
            try:
                from kiteconnect import KiteConnect  # pip install kiteconnect
                self._kite = KiteConnect(api_key=BROKER_API_KEY)
                self._kite.set_access_token(BROKER_ACCESS_TOKEN)
                self._connected = True
                log.info("Connected to Zerodha Kite")
            except ImportError:
                raise RuntimeError("Install kiteconnect: pip install kiteconnect")
        elif BROKER_TYPE == "upstox":
            try:
                import upstox_client                 # pip install upstox-client
                configuration = upstox_client.Configuration()
                configuration.access_token = BROKER_ACCESS_TOKEN
                self._upstox_mq = upstox_client.MarketQuoteApi(
                    upstox_client.ApiClient(configuration))
                self._upstox_ord = upstox_client.OrderApi(
                    upstox_client.ApiClient(configuration))
                self._upstox_hist = upstox_client.HistoryApi(
                    upstox_client.ApiClient(configuration))
                self._connected = True
                log.info("Connected to Upstox")
            except ImportError:
                raise RuntimeError("Install upstox-client: pip install upstox-client")
        else:
            raise ValueError(f"Unknown BROKER_TYPE: {BROKER_TYPE}")

    def get_nifty_spot(self) -> float:
        if PAPER_TRADE or not self._connected:
            raise NotImplementedError("paper_trade — no live spot")

        if BROKER_TYPE == "zerodha":
            ltp = self._kite.ltp(["NSE:NIFTY 50"])
            return ltp["NSE:NIFTY 50"]["last_price"]
        elif BROKER_TYPE == "upstox":
            resp = self._upstox_mq.ltp("NSE_INDEX|Nifty 50", "v2")
            return resp.data["NSE_INDEX:Nifty 50"].last_price

    def get_atm_option_quote(self, strike: int, option_type: str,
                              expiry: date) -> dict:
        """Returns {'ltp': float, 'iv': float, 'bid': float, 'ask': float}"""
        if PAPER_TRADE or not self._connected:
            return {"ltp": 100.0, "iv": 0.15, "bid": 99.0, "ask": 101.0}

        if BROKER_TYPE == "zerodha":
            sym = f"NFO:NIFTY{expiry.strftime('%y%b').upper()}{strike}{option_type}"
            quote = self._kite.quote([sym])[sym]
            return {
                "ltp": quote["last_price"],
                "iv":  quote.get("oi", 0) / 1e6,  # placeholder; use greeks if available
                "bid": quote["depth"]["buy"][0]["price"],
                "ask": quote["depth"]["sell"][0]["price"],
            }
        elif BROKER_TYPE == "upstox":
            sym = f"NFO_OPT|NIFTY|{expiry.strftime('%Y-%m-%d')}|{strike}|{option_type}"
            resp = self._upstox_mq.get_full_market_quote(sym, "v2")
            d    = resp.data[sym]
            return {"ltp": d.last_price, "iv": 0.15, "bid": d.depth.buy[0].price,
                    "ask": d.depth.sell[0].price}

    def get_historical_candles(self, symbol: str, interval: str,
                                from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
        if PAPER_TRADE or not self._connected:
            raise NotImplementedError("paper_trade — no historical data")

        if BROKER_TYPE == "zerodha":
            # Requires instrument token; use 256265 for Nifty 50 index
            token = 256265
            data  = self._kite.historical_data(token, from_dt, to_dt, interval)
            df    = pd.DataFrame(data)
            df    = df.rename(columns={"date": "datetime"}).set_index("datetime")
            return df[["open", "high", "low", "close", "volume"]]
        elif BROKER_TYPE == "upstox":
            resp = self._upstox_hist.get_historical_candle_data1(
                "NSE_INDEX|Nifty 50", interval,
                to_dt.strftime("%Y-%m-%d"), from_dt.strftime("%Y-%m-%d"), "v2")
            candles = resp.data.candles
            df = pd.DataFrame(candles,
                              columns=["datetime","open","high","low","close","volume","oi"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df.set_index("datetime")[["open","high","low","close","volume"]]

    def place_order(self, symbol: str, qty: int, transaction_type: str,
                    order_type: str = "MARKET", price: float = 0.0) -> str:
        tag = "[PAPER]" if PAPER_TRADE else "[LIVE]"
        log.info(f"{tag} {transaction_type} {qty} × {symbol} @ {price or 'MKT'}")

        if PAPER_TRADE:
            return f"PAPER_{datetime.now().strftime('%H%M%S%f')}"

        if BROKER_TYPE == "zerodha":
            from kiteconnect import KiteConnect
            order_id = self._kite.place_order(
                variety  = KiteConnect.VARIETY_REGULAR,
                exchange = "NFO",
                tradingsymbol = symbol,
                transaction_type = transaction_type,
                quantity = qty,
                product  = KiteConnect.PRODUCT_MIS,
                order_type = KiteConnect.ORDER_TYPE_MARKET,
            )
            return str(order_id)
        elif BROKER_TYPE == "upstox":
            import upstox_client
            body = upstox_client.PlaceOrderRequest(
                quantity         = qty,
                product          = "I",
                validity         = "DAY",
                price            = 0,
                tag              = "mf_nifty",
                instrument_token = symbol,
                order_type       = "MARKET",
                transaction_type = transaction_type,
                disclosed_quantity = 0,
                trigger_price    = 0,
                is_amo           = False,
            )
            resp = self._upstox_ord.place_order(body, "v2")
            return resp.data.order_id

    def get_order_status(self, order_id: str) -> str:
        if PAPER_TRADE:
            return "COMPLETE"
        if BROKER_TYPE == "zerodha":
            orders = self._kite.orders()
            for o in orders:
                if str(o["order_id"]) == order_id:
                    return o["status"]
        elif BROKER_TYPE == "upstox":
            resp = self._upstox_ord.get_order_details(order_id, "v2")
            return resp.data.status
        return "UNKNOWN"


# ─── ATM Strike Helper ────────────────────────────────────────────────────────

def get_atm_strike(spot: float, step: int = 50) -> int:
    return int(round(spot / step) * step)


def get_nearest_expiry() -> date:
    """Return nearest Thursday (Nifty weekly expiry)."""
    today = date.today()
    days_to_thursday = (3 - today.weekday()) % 7
    expiry = today + timedelta(days=days_to_thursday)
    if expiry == today:
        expiry += timedelta(days=7)
    return expiry


def option_symbol(strike: int, opt_type: str, expiry: date) -> str:
    """Construct NSE option symbol string."""
    return f"NIFTY{expiry.strftime('%y%b').upper()}{strike}{opt_type}"


# ─── Incremental Signal Engine ────────────────────────────────────────────────

class LiveSignalEngine:
    """
    Maintains a rolling DataFrame of 5-min bars and recomputes factors
    on each new bar close. Issues position directives in real time.
    """

    def __init__(self, cfg: StrategyConfig, warmup_bars: int = 80):
        self.cfg         = cfg
        self.factors     = FactorEngine(cfg)
        self.pricer      = OptionPricer(cfg)
        self.warmup_bars = warmup_bars
        self.bars        = pd.DataFrame()

    def ingest_bar(self, ts: pd.Timestamp, o: float, h: float,
                   l: float, c: float, vol: float) -> dict:
        """
        Feed one completed 5-min bar. Returns signal dict:
          {'score': float, 'desired': 'short_call'|'short_put'|None,
           'factors': {f1..f10}}
        """
        new_row = pd.DataFrame(
            {'open': o, 'high': h, 'low': l, 'close': c, 'volume': vol},
            index=[ts]
        )
        self.bars = pd.concat([self.bars, new_row]).tail(200)  # keep 200 bars

        if len(self.bars) < self.warmup_bars:
            return {'score': np.nan, 'desired': None, 'factors': {}}

        signal_df = self.factors.compute_all(self.bars)
        last      = signal_df.iloc[-1]
        score     = float(last.get('composite_score', np.nan))

        if np.isnan(score):
            desired = None
        elif score >= self.cfg.entry_threshold:
            desired = 'short_put'
        elif score <= -self.cfg.entry_threshold:
            desired = 'short_call'
        else:
            desired = None

        factors = {f'f{i}': round(float(last.get(f'f{i}', np.nan)), 4)
                   for i in range(1, 11)}
        return {'score': round(score, 4), 'desired': desired, 'factors': factors}


# ─── Live Session Manager ─────────────────────────────────────────────────────

class LiveSession:
    """
    Orchestrates the intraday session:
      1. Load morning historical bars for warmup
      2. Poll for new bars every 5 minutes at bar close
      3. Execute position switches via broker API
      4. Mandatory square-off at 15:15
    """

    def __init__(self, cfg: StrategyConfig = None, paper_trade: bool = True):
        self.cfg          = cfg or StrategyConfig()
        self.broker       = BrokerAPI()
        self.engine       = LiveSignalEngine(self.cfg)
        self.paper_trade  = paper_trade

        self.position     = None      # {'side', 'strike', 'expiry', 'order_id', 'entry_prem'}
        self.lots         = 1

    def run(self):
        log.info("=" * 55)
        log.info("  MULTI-FACTOR NIFTY ATM OPTION SELLING — LIVE")
        log.info(f"  Mode: {'PAPER TRADE' if self.paper_trade else 'LIVE TRADE'}")
        log.info("=" * 55)

        self._warmup()

        while True:
            now = datetime.now(IST)
            t   = now.time()

            # Mandatory square-off
            squareoff_t = datetime.strptime(self.cfg.squareoff_time, "%H:%M").time()
            if t >= squareoff_t:
                if self.position:
                    log.info("15:15 — mandatory square-off")
                    self._close_position("squareoff")
                log.info("Session complete.")
                break

            # Wait for next 5-min bar close
            self._sleep_to_next_bar()
            self._on_bar_close()

    def _warmup(self):
        """Load today's historical bars up to now for factor warmup."""
        log.info("Loading historical bars for warmup...")
        today_start = datetime.now(IST).replace(hour=9, minute=15, second=0)
        now         = datetime.now(IST)
        try:
            hist = self.broker.get_historical_candles(
                "NSE:NIFTY 50", "5minute", today_start, now)
            for ts, row in hist.iterrows():
                self.engine.ingest_bar(ts, row['open'], row['high'],
                                       row['low'], row['close'], row['volume'])
            log.info(f"Warmup complete: {len(hist)} bars loaded")
        except NotImplementedError:
            log.warning("Broker API not connected — running in simulation mode")

    def _on_bar_close(self):
        """Called at close of each 5-min bar."""
        now = datetime.now(IST)
        log.info(f"── Bar close: {now.strftime('%H:%M')} ──────────────────────────")

        try:
            spot = self.broker.get_nifty_spot()
        except NotImplementedError:
            spot = 22000.0  # simulation fallback

        # In production, use real bar data; here we approximate from tick
        o = h = l = c = spot
        vol = 100_000

        signal = self.engine.ingest_bar(pd.Timestamp(now), o, h, l, c, vol)
        score  = signal['score']
        desired = signal['desired']

        log.info(f"  Spot: {spot:.1f}  |  Score: {score:+.4f}  |  Signal: {desired}")
        for fname, fval in signal['factors'].items():
            log.debug(f"    {fname}: {fval:+.4f}")

        if np.isnan(score):
            log.info("  Score NaN — warmup in progress, no action")
            return

        # ── Position logic ────────────────────────────────────────────────────
        if self.position is None:
            if desired in ('short_call', 'short_put'):
                self._open_position(desired, spot)
        else:
            cur_side = self.position['side']
            # Check stop-loss
            if self._stop_loss_hit(spot):
                log.warning(f"  STOP LOSS hit — closing {cur_side}")
                self._close_position("stop_loss")
                return

            if desired is not None and desired != cur_side:
                log.info(f"  Signal switch: {cur_side} → {desired}")
                self._close_position("signal_switch")
                self._open_position(desired, spot)
            elif desired is None and abs(score) < self.cfg.exit_threshold:
                log.info(f"  Score neutral — closing {cur_side}")
                self._close_position("signal_neutral")

    def _open_position(self, side: str, spot: float):
        strike  = get_atm_strike(spot)
        expiry  = get_nearest_expiry()
        opt     = 'CE' if side == 'short_call' else 'PE'
        symbol  = option_symbol(strike, opt, expiry)
        qty     = self.lots * self.cfg.lot_size

        log.info(f"  SELL {symbol}  qty={qty}")
        order_id = self.broker.place_order(symbol, qty, "SELL")

        try:
            quote = self.broker.get_atm_option_quote(strike, opt, expiry)
            prem  = quote['ltp']
        except NotImplementedError:
            prem = 0.0

        sl_spot = (spot * (1 + self.cfg.stop_loss_pct) if side == 'short_call'
                   else spot * (1 - self.cfg.stop_loss_pct))

        self.position = {
            'side':        side,
            'strike':      strike,
            'expiry':      expiry,
            'symbol':      symbol,
            'order_id':    order_id,
            'entry_spot':  spot,
            'entry_prem':  prem,
            'stop_spot':   sl_spot,
        }
        msg = (f"📊 NIFTY SIGNAL\n"
               f"Action : SELL {opt}\n"
               f"Strike : {strike}\n"
               f"Spot   : {spot:.1f}\n"
               f"Premium: {prem:.1f}\n"
               f"SL-Spot: {sl_spot:.1f}\n"
               f"{'[PAPER]' if PAPER_TRADE else '[LIVE]'}")
        log.info(f"  Position open: {side} {symbol} @ ₹{prem:.1f}  SL-spot={sl_spot:.1f}")
        send_telegram(msg)

    def _close_position(self, reason: str):
        if self.position is None:
            return
        symbol = self.position['symbol']
        qty    = self.lots * self.cfg.lot_size
        log.info(f"  BUY {symbol}  qty={qty}  reason={reason}")
        self.broker.place_order(symbol, qty, "BUY")
        msg = (f"🔴 POSITION CLOSED\n"
               f"Symbol : {symbol}\n"
               f"Reason : {reason}\n"
               f"{'[PAPER]' if PAPER_TRADE else '[LIVE]'}")
        send_telegram(msg)
        self.position = None

    def _stop_loss_hit(self, spot: float) -> bool:
        if self.position is None:
            return False
        side     = self.position['side']
        sl_spot  = self.position['stop_spot']
        return (side == 'short_call' and spot >= sl_spot) or \
               (side == 'short_put'  and spot <= sl_spot)

    def _sleep_to_next_bar(self, bar_minutes: int = 5):
        now     = datetime.now(IST)
        elapsed = now.minute % bar_minutes * 60 + now.second
        wait    = bar_minutes * 60 - elapsed + 2   # 2-second buffer after bar close
        log.info(f"  Sleeping {wait}s until next bar close...")
        time.sleep(wait)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = StrategyConfig(
        entry_threshold = 0.25,
        exit_threshold  = 0.10,
        stop_loss_pct   = 0.008,
        premium_sl_mult = 2.0,
    )
    session = LiveSession(cfg=cfg, paper_trade=True)
    session.run()
