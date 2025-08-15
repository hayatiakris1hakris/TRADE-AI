import asyncio
import pandas as pd
from config import EXCHANGE_ID, TIMEFRAME, TRADING_SYMBOLS, TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_CHANNELS, WHALE_ASSETS, WHALE_LARGE_TX_USD, WHALE_LOOKBACK_HOURS, WHALE_PER_CHANNEL_LIMIT
from data.market_data import RealTimeMarketData
from external.whales.telegram_whale_service import TelegramWhaleService, WhaleTracker

async def main():
    # Market data
    md = RealTimeMarketData(exchange_id=EXCHANGE_ID, timeframe=TIMEFRAME)
    await md.initialize()
    for sym in TRADING_SYMBOLS:
        df = await md.fetch_ohlcv(sym, limit=500)
        print(f"[OK] OHLCV fetched: {sym} -> {len(df)} bars")
    await md.fetch_orderbook(TRADING_SYMBOLS[0])

    # Whale service (tek tur: fetch→parse→ingest)
    tracker = WhaleTracker(assets=WHALE_ASSETS, large_tx_usd=WHALE_LARGE_TX_USD)
    svc = TelegramWhaleService(
        api_id=TELEGRAM_API_ID, api_hash=TELEGRAM_API_HASH,
        channels=TELEGRAM_CHANNELS, tracker=tracker,
        lookback_hours=WHALE_LOOKBACK_HOURS, per_channel_limit=WHALE_PER_CHANNEL_LIMIT,
        interval_sec=180
    )
    await svc.start()
    # 10 sn bekle, ilk turu tamamlasın
    await asyncio.sleep(10)
    await svc.stop()

    # Son bar + whale feature’ları yazdır
    for sym in TRADING_SYMBOLS:
        df = md.data_buffer.get(sym)
        if df is None or df.empty:
            continue
        feats = tracker.features(sym, window_min=180)
        last_idx = df.index[-1]
        for k, v in feats.items():
            df.loc[last_idx, k] = v
        print(f"\n=== {sym} last bar ===")
        print(df.iloc[-1][['open','high','low','close','volume','rsi','macd_hist','bb_position','volume_ratio']].to_string())
        print("whale_features:", feats)

    await md.close()

if __name__ == "__main__":
    asyncio.run(main())
