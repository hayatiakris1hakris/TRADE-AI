# external/whales/test_whale_service.py
import asyncio
import argparse
# external/whales/test_whale_service.py
import os
os.makedirs("sessions", exist_ok=True)
session_name = "sessions/whale_service_session"  # aynı isim/klasör
client = TelegramClient(session_name, TELEGRAM_API_ID, TELEGRAM_API_HASH)

from datetime import datetime, timezone

from telethon import TelegramClient

from config import (
    TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_CHANNELS,
    WHALE_ASSETS, WHALE_LOOKBACK_HOURS, WHALE_PER_CHANNEL_LIMIT, WHALE_INTERVAL_SEC,
    WHALE_MIN_USD, WHALE_MIN_BTC, WHALE_MIN_ETH, WHALE_MIN_USDT, WHALE_MIN_USDC,
    WHALE_LARGE_TX_USD
)
from external.whales.telegram_whale_service import (
    TelegramWhaleService, WhaleTracker, WhaleTelegramFetcher, parse_events_from_messages
)

def _print_asset_report(tracker: WhaleTracker, assets: list[str], window_min: int = 360):
    if not assets:
        return
    for a in assets:
        sym = f"{a}/USDT"
        mets = tracker.features(sym, window_min=window_min)
        cats = tracker.category_counts(sym, window_min=window_min)
        print(f"\n=== {sym} ===")
        print("features:", mets)
        print("categories:", cats)
        last = tracker.last_events(sym, n=3)
        for e in last:
            print(f"  - {e.ts} {e.asset} ${e.usd_value:,.0f} {e.from_label} -> {e.to_label}")

async def run_once():
    if TELEGRAM_API_ID == 0 or not TELEGRAM_API_HASH:
        print("config.py içinde TELEGRAM_API_ID / TELEGRAM_API_HASH doldurun (my.telegram.org/apps).")
        return
    session_name = "whale_service_session"
    client = TelegramClient(session_name, TELEGRAM_API_ID, TELEGRAM_API_HASH)
    await client.connect()
    if not await client.is_user_authorized():
        phone = input("Telefon numaranız (örn +90...): ")
        await client.send_code_request(phone)
        code = input("Telegram doğrulama kodu: ")
        await client.sign_in(phone=phone, code=code)

    fetcher = WhaleTelegramFetcher(client, TELEGRAM_CHANNELS)
    print(f"[TEST] Fetching telegram messages lookback={WHALE_LOOKBACK_HOURS}h limit/ch={WHALE_PER_CHANNEL_LIMIT}")
    msgs = await fetcher.fetch_recent_messages(lookback_hours=WHALE_LOOKBACK_HOURS, per_channel_limit=WHALE_PER_CHANNEL_LIMIT)

    events, skips, asset_counts = parse_events_from_messages(
        msgs, None, WHALE_MIN_USD, WHALE_MIN_BTC, WHALE_MIN_ETH, WHALE_MIN_USDT, WHALE_MIN_USDC
    )
    print(f"[TEST] Parsed events: {len(events)} | Skip reasons: {dict(skips)} | Asset hits: {dict(asset_counts)}")

    tracker = WhaleTracker(assets=WHALE_ASSETS, large_tx_usd=WHALE_LARGE_TX_USD)
    added = tracker.ingest(events)
    print(f"[TEST] Ingested: {added}")

    _print_asset_report(tracker, WHALE_ASSETS, window_min=360)

    await client.disconnect()

async def run_service(duration_sec: int):
    if TELEGRAM_API_ID == 0 or not TELEGRAM_API_HASH:
        print("config.py içinde TELEGRAM_API_ID / TELEGRAM_API_HASH doldurun (my.telegram.org/apps).")
        return
    tracker = WhaleTracker(assets=WHALE_ASSETS, large_tx_usd=WHALE_LARGE_TX_USD)
    thresholds = {
        "min_usd": WHALE_MIN_USD, "min_btc": WHALE_MIN_BTC, "min_eth": WHALE_MIN_ETH,
        "min_usdt": WHALE_MIN_USDT, "min_usdc": WHALE_MIN_USDC
    }
    service = TelegramWhaleService(
        api_id=TELEGRAM_API_ID, api_hash=TELEGRAM_API_HASH,
        channels=TELEGRAM_CHANNELS, tracker=tracker,
        lookback_hours=WHALE_LOOKBACK_HOURS, per_channel_limit=WHALE_PER_CHANNEL_LIMIT,
        interval_sec=WHALE_INTERVAL_SEC, thresholds=thresholds
    )
    print(f"[TEST] Starting service for {duration_sec}s...")
    await service.start()
    try:
        await asyncio.sleep(duration_sec)
        print("[TEST] Printing report before stopping service.")
        _print_asset_report(tracker, WHALE_ASSETS, window_min=360)
    finally:
        print("[TEST] Stopping service...")
        await service.stop()
        print("[TEST] Service stopped.")

def parse_args():
    p = argparse.ArgumentParser(description="Test TelegramWhaleService")
    p.add_argument("--mode", choices=["once", "service"], default="once", help="Test modu")
    p.add_argument("--duration", type=int, default=180, help="service modu süresi (sn)")
    return p.parse_args()

async def main():
    args = parse_args()
    if args.mode == "once":
        await run_once()
    else:
        await run_service(args.duration)

if __name__ == "__main__":
    asyncio.run(main())
