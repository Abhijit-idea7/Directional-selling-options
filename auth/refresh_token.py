"""
Broker Token Refresh Script
============================

Called by the auth_refresh GitHub Actions workflow.
Outputs the new access token to stdout (just the token string, nothing else).

Reads credentials from environment variables (stored as GitHub Secrets):
  BROKER_API_KEY       — API key
  BROKER_API_SECRET    — API secret (for Zerodha request_token exchange)
  BROKER_TOTP_SECRET   — TOTP seed string (pyotp compatible)
  BROKER_USER_ID       — Login user ID
  BROKER_PASSWORD      — Login password
  BROKER_TYPE          — "zerodha" | "upstox"

Instructions per broker
------------------------

Zerodha (Kite Connect):
  The Kite Connect login flow requires a browser redirect to capture request_token.
  Two options:
    1. Use kiteconnect + Playwright headless browser (automated).
    2. Complete login manually in the morning and paste the request_token into a
       GitHub Secret called BROKER_REQUEST_TOKEN, then use the simple exchange below.

  For fully automated flow, set up kitelogin (pip install kite-login-helper) or
  use the Playwright approach in the commented-out section below.

Upstox:
  Upstox v2 supports access_token generation via TOTP + password without a redirect.
  Use the upstox_client SDK as shown below.
"""

import os
import sys
import pyotp

BROKER_TYPE       = os.getenv("BROKER_TYPE", "zerodha").lower()
API_KEY           = os.getenv("BROKER_API_KEY", "")
API_SECRET        = os.getenv("BROKER_API_SECRET", "")
TOTP_SECRET       = os.getenv("BROKER_TOTP_SECRET", "")
USER_ID           = os.getenv("BROKER_USER_ID", "")
PASSWORD          = os.getenv("BROKER_PASSWORD", "")
# For Zerodha manual exchange approach:
REQUEST_TOKEN     = os.getenv("BROKER_REQUEST_TOKEN", "")


def get_totp() -> str:
    return pyotp.TOTP(TOTP_SECRET).now()


# ─── Zerodha ──────────────────────────────────────────────────────────────────

def zerodha_token() -> str:
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=API_KEY)

    if REQUEST_TOKEN:
        # Simple approach: manually supply request_token as a secret each morning
        data = kite.generate_session(REQUEST_TOKEN, api_secret=API_SECRET)
        return data["access_token"]

    # Automated approach using Playwright (requires: pip install playwright && playwright install)
    # Uncomment and adapt if you want full automation:
    #
    # from playwright.sync_api import sync_playwright
    # import urllib.parse, re
    #
    # with sync_playwright() as p:
    #     browser = p.chromium.launch(headless=True)
    #     page    = browser.new_page()
    #     page.goto(kite.login_url())
    #     page.fill("#userid",   USER_ID)
    #     page.fill("#password", PASSWORD)
    #     page.click("button[type=submit]")
    #     page.wait_for_selector("#totp")
    #     page.fill("#totp", get_totp())
    #     page.click("button[type=submit]")
    #     page.wait_for_url("*request_token*")
    #     url = page.url
    #     browser.close()
    # request_token = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["request_token"][0]
    # data = kite.generate_session(request_token, api_secret=API_SECRET)
    # return data["access_token"]

    raise ValueError(
        "Set BROKER_REQUEST_TOKEN secret or enable the Playwright automated flow. "
        "See comments in auth/refresh_token.py."
    )


# ─── Upstox ───────────────────────────────────────────────────────────────────

def upstox_token() -> str:
    import requests as req

    totp = get_totp()
    session = req.Session()

    # Step 1: Login
    resp = session.post(
        "https://api-v2.upstox.com/login",
        json={"mobile_number": USER_ID, "password": PASSWORD},
    )
    resp.raise_for_status()

    # Step 2: TOTP verification
    resp = session.post(
        "https://api-v2.upstox.com/login/otp/verify",
        json={"otp": totp},
    )
    resp.raise_for_status()

    # Step 3: Authorize API key
    resp = session.get(
        f"https://api-v2.upstox.com/login/authorization/dialog"
        f"?response_type=code&client_id={API_KEY}"
        f"&redirect_uri=https://127.0.0.1",
        allow_redirects=False,
    )
    code = resp.headers.get("Location", "").split("code=")[-1].split("&")[0]

    # Step 4: Exchange code for access_token
    resp = req.post(
        "https://api-v2.upstox.com/login/authorization/token",
        data={
            "code":          code,
            "client_id":     API_KEY,
            "client_secret": API_SECRET,
            "redirect_uri":  "https://127.0.0.1",
            "grant_type":    "authorization_code",
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if BROKER_TYPE == "zerodha":
        token = zerodha_token()
    elif BROKER_TYPE == "upstox":
        token = upstox_token()
    else:
        print(f"Unknown BROKER_TYPE: {BROKER_TYPE}", file=sys.stderr)
        sys.exit(1)

    print(token, end="")   # stdout only — no newline, no extra text
