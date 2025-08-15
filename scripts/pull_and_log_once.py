import asyncio
import pandas as pd
from datetime import datetime
from typing import Dict
from config import EXCHANGE_ID, TIMEFRAME, TRADING_SYMBOLS, TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_CHANNELS, WHALE_ASSETS, WHALE_LARGE_TX_USD, WHALE_LOOKBACK_HOURS, WHALE_PER_CHANNEL_LIMIT, LOG_DIR, LOG_FILE
from data.market_data import RealTimeMarketData
from external.whales.telegram_whale_service import TelegramWhaleService, WhaleTracker
from utils.csv_logger import CsvLogger
from telethon import TelegramClient

def build_log_row(sym, row, whale_feats):
    out = { 'timestamp': row.name.isoformat(), 'symbol': sym,
            'open': float(row.get('open', 0.0)), 'high': float(row.get('high', 0.0)),
            'low': float(row.get('low', 0.0)), 'close': float(row.get('close', 0.0)),
            'volume': float(row.get('volume', 0.0)), 'returns': float(row.get('returns', 0.0)),
            'rsi': float(row.get('rsi', 0.0)), 'macd_hist': float(row.get('macd_hist', 0.0)),
            'bb_position': float(row.get('bb_position', 0.0)), 'sma_20': float(row.get('sma_20', 0.0)),
            'sma_50': float(row.get('sma_50', 0.0)), 'ema_20': float(row.get('ema_20', 0.0)),
            'ema_50': float(row.get('ema_50', 0.0)), 'volume_ratio': float(row.get('volume_ratio', 0.0)),
            'atr': float(row.get('atr', 0.0)), 'spread_percentage': float(row.get('spread_percentage', 0.0)),
            'news_sentiment': float(row.get('news_sentiment', 0.0)),
            'policy_risk_score': float(row.get('policy_risk_score', 0.0)),
            'analyst_consensus': float(row.get('analyst_consensus', 0.0)) }
    for k, v in whale_feats.items():
        out[k] = float(v)
    return out

async def main():
    logger = CsvLogger(LOG_DIR, LOG_FILE)
    md = RealTimeMarketData(exchange_id=EXCHANGE_ID, timeframe=TIMEFRAME)
    await md.initialize()
    svc = None
    try:
        for sym in TRADING_SYMBOLS:
            df = await md.fetch_ohlcv(sym, limit=500)
            print(f'[OK] {sym} bars={len(df)}')
        ob = await md.fetch_orderbook(TRADING_SYMBOLS[0])
        if ob:
            sym0 = TRADING_SYMBOLS[0]
            df0 = md.data_buffer.get(sym0)
            if df0 is not None and (not df0.empty):
                df0.loc[df0.index[-1], 'spread_percentage'] = float(ob.get('spread_percentage', 0.0))
        tracker = WhaleTracker(assets=WHALE_ASSETS, large_tx_usd=WHALE_LARGE_TX_USD)
        # Opsiyonel interaktif login (ilk kez)
        # from telethon import TelegramClient
        # client = TelegramClient('sessions/whale_service_session', TELEGRAM_API_ID, TELEGRAM_API_HASH)
        # await client.connect()
        # if not await client.is_user_authorized():
        #     phone = input('Telefon numaranız (örn +90...): ').strip()
        #     await client.send_code_request(phone)
        #     code = input('OTP: ').strip()
        #     await client.sign_in(phone=phone, code=code)
        # await client.disconnect()

        svc = TelegramWhaleService(TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_CHANNELS, tracker, lookback_hours=WHALE_LOOKBACK_HOURS, per_channel_limit=WHALE_PER_CHANNEL_LIMIT, interval_sec=180)
        await svc.start()
        await asyncio.sleep(10)
        await svc.stop()
        svc = None

        for sym in TRADING_SYMBOLS:
            df = md.data_buffer.get(sym)
            if df is None or df.empty:
                continue
            feats = tracker.features(sym, window_min=180)
            row = df.iloc[-1]
            rec = build_log_row(sym, row, feats)
            logger.log_row(rec)
            print(f'[LOG] wrote last bar of {sym} to {LOG_DIR}/{LOG_FILE}')
    finally:
        if svc is not None:
            try:
                await svc.stop()
            except Exception:
                pass
        await md.close()

if __name__ == '__main__':
    asyncio.run(main())
