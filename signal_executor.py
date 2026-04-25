"""
Signal Executor — GitHub Actions stateful position manager
===========================================================

Called after live_trading_adapter.py generates results/current_signal.json.
Reads the previous position state from state/position.json (persisted between
workflow runs via actions/cache), compares with the new signal, and places
orders only when the signal changes.

This decouples signal generation from order execution, making it easy to
paper-trade or use with any broker without modifying the core strategy.

Usage:
    python signal_executor.py --state-dir state --signal-dir results
"""

import os
import json
import logging
import argparse
import requests
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("executor")

IST = ZoneInfo("Asia/Kolkata")

# Env vars
BROKER_API_KEY      = os.getenv("BROKER_API_KEY", "")
BROKER_ACCESS_TOKEN = os.getenv("BROKER_ACCESS_TOKEN", "")
BROKER_TYPE         = os.getenv("BROKER_TYPE", "zerodha").lower()
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
PAPER_TRADE         = os.getenv("PAPER_TRADE", "true").lower() == "true"
LOT_SIZE            = int(os.getenv("LOT_SIZE", "25"))
LOTS                = int(os.getenv("LOTS", "1"))
STOP_LOSS_PCT       = float(os.getenv("STOP_LOSS_PCT", "0.008"))
PREMIUM_SL_MULT     = float(os.getenv("PREMIUM_SL_MULT", "2.0"))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
                      timeout=5)
    except Exception as e:
        log.warning(f"Telegram failed: {e}")


def get_atm_strike(spot: float, step: int = 50) -> int:
    return int(round(spot / step) * step)


def nearest_expiry() -> date:
    today = date.today()
    days_to_thursday = (3 - today.weekday()) % 7
    expiry = today + timedelta(days=days_to_thursday)
    if expiry == today:
        expiry += timedelta(days=7)
    return expiry


def option_symbol_zerodha(strike: int, opt_type: str, expiry: date) -> str:
    return f"NIFTY{expiry.strftime('%y%b').upper()}{strike}{opt_type}"


# ─── Broker calls ─────────────────────────────────────────────────────────────

def place_order(symbol: str, qty: int, txn: str) -> str:
    tag = "[PAPER]" if PAPER_TRADE else "[LIVE]"
    log.info(f"{tag} {txn} {qty} × {symbol}")
    if PAPER_TRADE:
        return f"PAPER_{datetime.now().strftime('%H%M%S%f')}"

    if BROKER_TYPE == "zerodha":
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=BROKER_API_KEY)
        kite.set_access_token(BROKER_ACCESS_TOKEN)
        order_id = kite.place_order(
            variety           = KiteConnect.VARIETY_REGULAR,
            exchange          = "NFO",
            tradingsymbol     = symbol,
            transaction_type  = txn,
            quantity          = qty,
            product           = KiteConnect.PRODUCT_MIS,
            order_type        = KiteConnect.ORDER_TYPE_MARKET,
        )
        return str(order_id)
    elif BROKER_TYPE == "upstox":
        import upstox_client
        cfg = upstox_client.Configuration()
        cfg.access_token = BROKER_ACCESS_TOKEN
        api  = upstox_client.OrderApi(upstox_client.ApiClient(cfg))
        body = upstox_client.PlaceOrderRequest(
            quantity=qty, product="I", validity="DAY", price=0,
            instrument_token=symbol, order_type="MARKET",
            transaction_type=txn, disclosed_quantity=0,
            trigger_price=0, is_amo=False,
        )
        return api.place_order(body, "v2").data.order_id
    return "UNKNOWN"


# ─── State helpers ─────────────────────────────────────────────────────────────

def load_state(state_dir: str) -> dict:
    path = os.path.join(state_dir, "position.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"position": None}


def save_state(state_dir: str, state: dict):
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "position.json"), "w") as f:
        json.dump(state, f, indent=2)


# ─── Core logic ───────────────────────────────────────────────────────────────

def execute(signal_dir: str, state_dir: str):
    # Load current signal
    signal_path = os.path.join(signal_dir, "current_signal.json")
    if not os.path.exists(signal_path):
        log.error(f"Signal file not found: {signal_path}")
        return

    with open(signal_path) as f:
        signal = json.load(f)

    new_side   = signal.get("signal")          # short_call | short_put | flat
    spot       = signal.get("spot", 0.0)
    score      = signal.get("composite_score", 0.0)
    ts         = signal.get("timestamp", "")

    log.info(f"Signal  : {new_side}  |  Score: {score:+.4f}  |  Spot: {spot:.1f}")

    state       = load_state(state_dir)
    current_pos = state.get("position")       # None or dict

    # ── Mandatory square-off at 15:15 IST ────────────────────────────────────
    now_ist  = datetime.now(IST)
    if now_ist.hour > 15 or (now_ist.hour == 15 and now_ist.minute >= 15):
        if current_pos:
            log.info("15:15 mandatory square-off")
            _close_position(current_pos, spot, "squareoff")
            state["position"] = None
            save_state(state_dir, state)
        return

    # ── Stop-loss check on open position ─────────────────────────────────────
    if current_pos:
        sl_hit = False
        side   = current_pos["side"]
        sl_s   = current_pos.get("stop_spot", 0)
        entry_prem = current_pos.get("entry_prem", 0)

        if side == "short_call" and spot >= sl_s:
            sl_hit = True
        elif side == "short_put" and spot <= sl_s:
            sl_hit = True

        if sl_hit:
            log.warning(f"Stop-loss hit: {side} (spot={spot:.1f}, sl={sl_s:.1f})")
            _close_position(current_pos, spot, "stop_loss")
            state["position"] = None
            current_pos = None
            save_state(state_dir, state)

    # ── Act on signal ─────────────────────────────────────────────────────────
    if new_side in ("flat", "insufficient_data", None):
        if current_pos:
            log.info("Signal flat — closing position")
            _close_position(current_pos, spot, "signal_neutral")
            state["position"] = None
            save_state(state_dir, state)
        else:
            log.info("Flat signal, no position — nothing to do")
        return

    # Signal is short_call or short_put
    desired_side = new_side

    if current_pos is None:
        # Enter new position
        state["position"] = _open_position(desired_side, spot)
        save_state(state_dir, state)

    elif current_pos["side"] != desired_side:
        # Switch
        log.info(f"Switch: {current_pos['side']} → {desired_side}")
        _close_position(current_pos, spot, "signal_switch")
        state["position"] = _open_position(desired_side, spot)
        save_state(state_dir, state)

    else:
        log.info(f"Holding {current_pos['side']} — no change")


def _open_position(side: str, spot: float) -> dict:
    opt    = "CE" if side == "short_call" else "PE"
    strike = get_atm_strike(spot)
    expiry = nearest_expiry()
    symbol = option_symbol_zerodha(strike, opt, expiry)
    qty    = LOTS * LOT_SIZE

    order_id = place_order(symbol, qty, "SELL")

    sl_spot = (spot * (1 + STOP_LOSS_PCT) if side == "short_call"
               else spot * (1 - STOP_LOSS_PCT))

    pos = {
        "side":        side,
        "symbol":      symbol,
        "strike":      strike,
        "expiry":      expiry.isoformat(),
        "order_id":    order_id,
        "entry_spot":  spot,
        "entry_prem":  0.0,          # filled from broker quote if available
        "stop_spot":   round(sl_spot, 2),
        "entry_time":  datetime.now(IST).isoformat(),
    }

    msg = (f"{'📄 PAPER' if PAPER_TRADE else '🟢 LIVE'} SELL {opt}\n"
           f"Strike : {strike}  ({symbol})\n"
           f"Spot   : {spot:.1f}\n"
           f"SL-Spot: {sl_spot:.1f}")
    log.info(msg)
    send_telegram(msg)
    return pos


def _close_position(pos: dict, spot: float, reason: str):
    qty = LOTS * LOT_SIZE
    place_order(pos["symbol"], qty, "BUY")
    msg = (f"{'📄 PAPER' if PAPER_TRADE else '🔴 LIVE'} BUY (close)\n"
           f"Symbol : {pos['symbol']}\n"
           f"Spot   : {spot:.1f}\n"
           f"Reason : {reason}")
    log.info(msg)
    send_telegram(msg)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir",  default="state",   help="Dir for position.json")
    parser.add_argument("--signal-dir", default="results",  help="Dir with current_signal.json")
    args = parser.parse_args()

    execute(args.signal_dir, args.state_dir)
