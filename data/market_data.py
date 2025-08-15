# file: data/market_data.py
import pandas as pd
import ccxt.async_support as ccxt
from typing import Dict, Optional
from datetime import datetime, timezone

# Safe import for followline
try:
    from core.indicators_falloween import followline
except Exception:
    def followline(df: pd.DataFrame, bb_period=20, bb_dev=2.0, atr_period=5):
        # No-op fallback to avoid import errors
        df['followline_long'] = 0.0
        df['followline_short'] = 0.0
        return df


class RealTimeMarketData:
    def __init__(self, exchange_id: str = 'binance', timeframe: str = '1m'):
        # Keep it simple; add binance options if needed
        self.exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        if exchange_id.lower() == 'binance':
            # Use USDT-m futures endpoints
            self.exchange.options.update({'defaultType': 'future', 'adjustForTimeDifference': True})
        self.timeframe = timeframe
        self.data_buffer: Dict[str, pd.DataFrame] = {}
        self.orderbook_cache: Dict[str, Dict] = {}

    async def initialize(self):
        await self.exchange.load_markets()

    async def close(self):
        try:
            await self.exchange.close()
        except Exception:
            pass

    async def fetch_ohlcv(self, symbol: str, limit: int = 500) -> Optional[pd.DataFrame]:
        try:
            raw = await self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df = self._indicators(df)
            self.data_buffer[symbol] = df
            return df
        except Exception as e:
            print(f'[MarketData] fetch_ohlcv error {symbol}: {e}')
            return None

    async def fetch_orderbook(self, symbol: str) -> Optional[Dict]:
        try:
            ob = await self.exchange.fetch_order_book(symbol)
            if ob['bids'] and ob['asks']:
                bid_vol = sum(b[1] for b in ob['bids'][:10])
                ask_vol = sum(a[1] for a in ob['asks'][:10])
                best_ask = ob['asks'][0][0]
                best_bid = ob['bids'][0][0]
                spread = best_ask - best_bid
                mid = (best_ask + best_bid) / 2
            else:
                bid_vol = ask_vol = spread = mid = 0.0
            info = {
                'bid_volume': bid_vol,
                'ask_volume': ask_vol,
                'ob_bid_ask_ratio': bid_vol / (ask_vol + 1e-10),
                'spread': spread,
                'spread_percentage': spread / mid * 100 if mid > 0 else 0.0,
                'mid_price': mid,
                'timestamp': datetime.now(timezone.utc),
            }
            self.orderbook_cache[symbol] = info
            return info
        except Exception as e:
            print(f'[MarketData] fetch_orderbook error {symbol}: {e}')
            return None

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['returns'] = df['close'].pct_change()
        for p in [20, 50, 100, 200]:
            df[f'sma_{p}'] = df['close'].rolling(p).mean()
            df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
        # RSI(14)
        d = df['close'].diff()
        gain = d.where(d > 0, 0).rolling(14).mean()
        loss = (-d.where(d < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - 100 / (1 + rs)
        # MACD hist
        e1 = df['close'].ewm(span=12, adjust=False).mean()
        e2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = e1 - e2
        sig = macd.ewm(span=9, adjust=False).mean()
        df['macd_hist'] = macd - sig
        # Bollinger(20, 2)
        mid = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        df['bb_position'] = (df['close'] - lower) / (upper - lower + 1e-10)
        # Volume ratio vs SMA(20)
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
        # ATR(14)
        high_low = (df['high'] - df['low']).abs()
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.ewm(span=14, adjust=False).mean()
        # Custom indicator (safe fallback if missing)
        df = followline(df, bb_period=20, bb_dev=2.0, atr_period=5)
        return df
