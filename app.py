import asyncio
from datetime import datetime, timezone
from collections import defaultdict
from typing import List, Optional, Dict
import pandas as pd

from config import (
    TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_CHANNELS,
    EXCHANGE_ID, TIMEFRAME, TRADING_SYMBOLS, WHALE_ASSETS,
    WHALE_LOOKBACK_HOURS, WHALE_PER_CHANNEL_LIMIT, WHALE_INTERVAL_SEC,
    WHALE_LARGE_TX_USD, WHALE_MIN_USD, WHALE_MIN_BTC, WHALE_MIN_ETH,
    WHALE_MIN_USDT, WHALE_MIN_USDC, LOG_DIR, LOG_FILE
)
from data.market_data import RealTimeMarketData
from core.fake_momentum import FakeMomentumDetector
from core.momentum_events import MomentumEventTracker
from core.regime_classifier import RegimeClassifier
from core.continuation_model import ContinuationModel
from core.strategy_selector import choose_strategy
from external.whales.telegram_whale_service import TelegramWhaleService, WhaleTracker
from signals.signals_hub import SignalsHub
from utils.csv_logger import CsvLogger


class TradingApp:

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.market = RealTimeMarketData(exchange_id=EXCHANGE_ID, timeframe=TIMEFRAME)
        self.detector = FakeMomentumDetector()
        self.momentum = MomentumEventTracker(look_ahead_minutes=30)
        self.whale_tracker = WhaleTracker(assets=WHALE_ASSETS, large_tx_usd=WHALE_LARGE_TX_USD)
        self.whale_service = TelegramWhaleService(
            api_id=TELEGRAM_API_ID,
            api_hash=TELEGRAM_API_HASH,
            channels=TELEGRAM_CHANNELS,
            tracker=self.whale_tracker,
            lookback_hours=WHALE_LOOKBACK_HOURS,
            per_channel_limit=WHALE_PER_CHANNEL_LIMIT,
            interval_sec=WHALE_INTERVAL_SEC,
            thresholds={
                'min_usd': WHALE_MIN_USD,
                'min_btc': WHALE_MIN_BTC,
                'min_eth': WHALE_MIN_ETH,
                'min_usdt': WHALE_MIN_USDT,
                'min_usdc': WHALE_MIN_USDC,
            },
        )
        self.signals = SignalsHub(self.whale_tracker)
        self.regime_clf = RegimeClassifier(model_path='models/regime_clf.joblib')
        self.cont_model = ContinuationModel('models/cont_cls.joblib', 'models/cont_reg.joblib')
        now_utc = pd.Timestamp.now(tz='UTC')
        self.last_fetch = defaultdict(lambda: {
            'ohlcv': now_utc - pd.Timedelta(seconds=10),
            'ob':    now_utc - pd.Timedelta(seconds=10),
        })
        self.min_interval = {'ohlcv': 2, 'ob': 3}
        self.running = False
        self.logger = CsvLogger(LOG_DIR, LOG_FILE)
        # Kapanmış bar log takibi (aynı kapanmış barı bir kez yazmak için)
        self.last_logged_closed_ts: Dict[str, pd.Timestamp] = {}

    async def initialize(self):
        if TELEGRAM_API_ID == 0 or not TELEGRAM_API_HASH:
            raise RuntimeError('TELEGRAM_API_ID / TELEGRAM_API_HASH doldurun (config.py).')
        await self.market.initialize()
        for s in self.symbols:
            await self.market.fetch_ohlcv(s)
        await self.whale_service.start()
        print('[App] Initialized.')

    async def loop(self):
        self.running = True
        try:
            while self.running:
                await asyncio.gather(*(self.process_symbol(s) for s in self.symbols), return_exceptions=True)
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def process_symbol(self, symbol: str):
        try:
            now = pd.Timestamp.now(tz='UTC')

            # 1) OHLCV/Orderbook fetch (rate limit’e saygı)
            df = self.market.data_buffer.get(symbol)
            if (now - self.last_fetch[symbol]['ohlcv']).total_seconds() >= self.min_interval['ohlcv']:
                df = await self.market.fetch_ohlcv(symbol)
                self.last_fetch[symbol]['ohlcv'] = now
            if (now - self.last_fetch[symbol]['ob']).total_seconds() >= self.min_interval['ob']:
                ob = await self.market.fetch_orderbook(symbol)
                self.last_fetch[symbol]['ob'] = now
            else:
                ob = self.market.orderbook_cache.get(symbol)

            if df is None or len(df) < 101:
                # Kapanmış barı yazmak için en az 2 bar gerekir ([-2] erişeceğiz). 100+ indikatör için de güvenlik.
                return

            # 2) Son bar’a dış sinyal özelliklerini ekle (whale/news/policy/analyst)
            ext = await self.signals.build_features(symbol, window_min=180)
            li_current = df.index[-1]              # forming (henüz kapanmamış) bar
            li_closed = df.index[-2]               # kapanmış bar

            # orderbook’tan spread% son bar’a yaz
            if ob:
                try:
                    df.loc[li_current, 'spread_percentage'] = float(ob.get('spread_percentage', 0.0))
                except Exception:
                    pass

            # sinyal modüllerinden gelen özellikleri son bar’a yaz
            for (k, v) in ext.items():
                df.loc[li_current, k] = v

            # 3) Sinyal/strateji (forming bar üzerinden)
            signal = self._basic_signal(df)
            feat_row_current = df.iloc[-1].to_dict()
            feat_vec = self.detector.build_features(feat_row_current)
            fake_p = self.detector.predict_proba(feat_vec)

            if signal and signal['type'] == 'OPEN_LONG' and (fake_p > 0.65):
                signal['size'] *= 0.3
                if signal['size'] < 0.02:
                    signal = None

            # Momentum event etiketleme
            direction = self.momentum.maybe_trigger(symbol, li_current, df.tail(10))
            if direction:
                self.momentum.open_event(symbol, li_current, float(df['close'].iloc[-1]), direction, feat_vec)
            self.momentum.resolve_due(datetime.now(timezone.utc), symbol, float(df['close'].iloc[-1]), self.detector)

            plan = None
            reg_probs: Dict[str, float] = {}
            cont: Dict[str, float] = {}
            if signal and signal['type'].startswith('OPEN_'):
                row = df.iloc[-1].to_dict()
                reg_probs = self.regime_clf.predict_proba(row)
                side = 'long' if signal['type'] == 'OPEN_LONG' else 'short'
                cont = self.cont_model.predict(row, side=side, regime_probs=reg_probs)
                plan = choose_strategy(reg_probs, cont, side)
                if plan['strategy'] is None:
                    signal = None
                else:
                    signal['size'] = max(0.02, min(0.2, signal['size'] * plan['size_mult']))
                    signal['meta'] = {
                        'strategy': plan['strategy'],
                        'regime_probs': reg_probs,
                        'cont': cont,
                        'risk': {'sl_atr': plan['sl_atr'], 'tp_atr': plan['tp_atr'], 'hold_min': plan['hold_min']}
                    }

            # 4) LOG: sadece kapanmış barı yaz
            # Aynı kapanmış bar’ı birden fazla yazmamak için guard
            if self.last_logged_closed_ts.get(symbol) != li_closed:
                # kapanmış bar satırını topla (forming bar’daki bazı alanları da taşıyacağız)
                self._log_closed_bar(df, symbol, ob, fake_p, reg_probs, cont, plan, direction, li_closed)
                self.last_logged_closed_ts[symbol] = li_closed

            # 5) Konsol sinyal çıktısı
            if signal and 'meta' in signal:
                print(f"[Signal] {symbol} -> {signal['meta']['strategy']} "
                      f"size={signal['size']:.3f} cont={signal['meta']['cont']['cont_prob']:.2f} fake_p={fake_p:.2f}")
            elif signal and signal['type'].startswith('CLOSE_'):
                print(f"[Signal] {symbol} -> {signal['type']}")

        except Exception as e:
            print(f'[process_symbol] {symbol} error: {e}')

    def _basic_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 100:
            return None
        last = df.iloc[-1]
        prev = df.iloc[-2]
        size = 0.1
        # Basit örnek kural: RSI düşük + SMA20 yukarı kesişim -> long aç
        if last['rsi'] < 30 and last['sma_20'] > last['sma_50'] and (prev['sma_20'] <= prev['sma_50']):
            return {'type': 'OPEN_LONG', 'size': size}
        # Kapanış: RSI yüksek + SMA20 aşağı kesişim -> close long
        if last['rsi'] > 70 and last['sma_20'] < last['sma_50'] and (prev['sma_20'] >= prev['sma_50']):
            return {'type': 'CLOSE_LONG', 'size': size}
        return None

    def _log_closed_bar(
        self,
        df: pd.DataFrame,
        symbol: str,
        ob: Optional[Dict],
        fake_p_current: float,
        reg_probs_current: Dict[str, float],
        cont_current: Dict[str, float],
        plan_current: Optional[Dict],
        direction_current: Optional[str],
        li_closed: pd.Timestamp,
    ):
        # Kapanmış bar satırı
        row = df.loc[li_closed]
        # Kapanmış bara yazılacak alanlar: teknik + whale/news/...; forming bar’dan gelen tanı (fake_p, reg_probs, cont) sadece teşhis amaçlı eklenir
        row_dict = {
            'timestamp': li_closed.isoformat(),
            'symbol': symbol,
            'open': float(row.get('open', 0.0)),
            'high': float(row.get('high', 0.0)),
            'low': float(row.get('low', 0.0)),
            'close': float(row.get('close', 0.0)),
            'volume': float(row.get('volume', 0.0)),
            # teknik
            'returns': float(row.get('returns', 0.0)),
            'rsi': float(row.get('rsi', 0.0)),
            'macd_hist': float(row.get('macd_hist', 0.0)),
            'bb_position': float(row.get('bb_position', 0.0)),
            'sma_20': float(row.get('sma_20', 0.0)),
            'sma_50': float(row.get('sma_50', 0.0)),
            'ema_20': float(row.get('ema_20', 0.0)),
            'ema_50': float(row.get('ema_50', 0.0)),
            'volume_ratio': float(row.get('volume_ratio', 0.0)),
            'atr': float(row.get('atr', 0.0)),
            'spread_percentage': float(row.get('spread_percentage', 0.0)),
            # whales/news/policy/analyst (son bilinen değerler kapanmış bar satırına)
            'whale_ex_inflow_usd': float(row.get('whale_ex_inflow_usd', 0.0)),
            'whale_ex_outflow_usd': float(row.get('whale_ex_outflow_usd', 0.0)),
            'whale_netflow_usd': float(row.get('whale_netflow_usd', 0.0)),
            'whale_netflow_log': float(row.get('whale_netflow_log', 0.0)),
            'whale_large_tx_count': float(row.get('whale_large_tx_count', 0.0)),
            'whale_recency_score': float(row.get('whale_recency_score', 0.0)),
            'whale_ex2ex_share': float(row.get('whale_ex2ex_share', 0.0)),
            'whale_unknown_share': float(row.get('whale_unknown_share', 0.0)),
            'news_sentiment': float(row.get('news_sentiment', 0.0)),
            'policy_risk_score': float(row.get('policy_risk_score', 0.0)),
            'analyst_consensus': float(row.get('analyst_consensus', 0.0)),
            # forming bar teşhisleri (bilgi amaçlı)
            'fake_p': float(fake_p_current),
            'reg_trend_up': float(reg_probs_current.get('trend_up', 0.0)),
            'reg_trend_down': float(reg_probs_current.get('trend_down', 0.0)),
            'reg_range': float(reg_probs_current.get('range', 0.0)),
            'reg_volatile': float(reg_probs_current.get('volatile', 0.0)),
            'cont_prob': float(cont_current.get('cont_prob', 0.0)),
            'remain_atr': float(cont_current.get('remain_atr', 0.0)),
            'expected_move_pct': float(cont_current.get('expected_move_pct', 0.0)),
            'momentum_event': direction_current or '',
        }

        # plan kolonları her zaman yazılsın (header stabil olsun)
        row_dict.setdefault('plan_strategy', '')
        row_dict.setdefault('plan_size_mult', 0.0)
        row_dict.setdefault('plan_hold_min', 0)
        row_dict.setdefault('plan_sl_atr', 0.0)
        row_dict.setdefault('plan_tp_atr', 0.0)

        if plan_current:
            row_dict.update({
                'plan_strategy': plan_current.get('strategy', ''),
                'plan_size_mult': float(plan_current.get('size_mult', 1.0)),
                'plan_hold_min': int(plan_current.get('hold_min', 0)),
                'plan_sl_atr': float(plan_current.get('sl_atr', 0.0)),
                'plan_tp_atr': float(plan_current.get('tp_atr', 0.0)),
            })

        try:
            self.logger.log_row(row_dict)
        except Exception as e:
            print(f'[Logger] error: {e}')

    async def shutdown(self):
        self.running = False
        await self.whale_service.stop()
        await self.market.close()
        print('[App] Shutdown complete.')


async def main():
    app = TradingApp(TRADING_SYMBOLS)
    try:
        await app.initialize()
        await app.loop()
    except KeyboardInterrupt:
        print('[Main] KeyboardInterrupt')
    finally:
        await app.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
