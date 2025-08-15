# Project Context (One-pager)

Goal
- Real-time crypto momentum: predict regime (trend_up/down/range/volatile) and continuation/remain move; choose strategy (trend/meanrev/breakout) and risk (size/SL/TP/hold).

Live flow
- app.py orchestrates:
  - data/market_data.py: OHLCV+orderbook (ccxt), indicators (returns, SMA/EMA, RSI, MACD, BB, ATR, volume_ratio, spread%)
  - external/whales/telegram_whale_service.py: Telegram whale messages → WhaleTracker features
  - signals/signals_hub.py: external features hub (whales + stubs for news/policy/analysts)
  - core/fake_momentum.py: online fake momentum gating (SGD, partial_fit)
  - core/regime_classifier.py: regime probabilities (model or fallback)
  - core/continuation_model.py: continuation/remain_atr (model or fallback)
  - core/strategy_selector.py: choose strategy and risk params
  - utils/csv_logger.py: per closed bar CSV logger (logs/market_features.csv)

Training
- scripts/train_regime_classifier.py → models/regime_clf.joblib
- scripts/train_continuation.py → models/cont_cls.joblib, models/cont_reg.joblib

Key params
- config.py: TELEGRAM_API_ID/HASH, TELEGRAM_CHANNELS, EXCHANGE_ID, TIMEFRAME, TRADING_SYMBOLS, WHALE thresholds
- continuation labels: lookahead & MFE/ATR thresholds
- regime labels: Sharpe/vol thresholds

Design notes
- External signals strong in training; light in live (latency risk).
- Online learning only for fake momentum.
- No data leakage; labels offline.

Run
- .env: secrets → TELEGRAM_API_ID, TELEGRAM_API_HASH, EXCHANGE_ID, TIMEFRAME, TRADING_SYMBOLS
- python app.py
- Whale test: python -m external.whales.test_whale_service --mode once

What to send in new chats
- Repo link + “Use ProjectContext.md and README.md”
- Only diffs/patches for changed files