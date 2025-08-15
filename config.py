# file: config.py
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def env_list(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if not val:
        return default
    try:
        import ast
        if val.strip().startswith('['):
            out = ast.literal_eval(val)
            return [str(x).strip() for x in out]
    except Exception:
        pass
    return [s.strip() for s in val.split(',') if s.strip()]

TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "")

# Kritik: Bu değişken tanımlı olmalı
TELEGRAM_CHANNELS = env_list("TELEGRAM_CHANNELS", [
    "whale_alert", "whale_alert_io", "whalealertio",
    "lookonchain", "whalechartorg"
])

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
TRADING_SYMBOLS = env_list("TRADING_SYMBOLS", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

WHALE_ASSETS = env_list("WHALE_ASSETS", ["BTC", "ETH", "USDT", "USDC", "SOL", "XRP", "DOGE"])
WHALE_LOOKBACK_HOURS = int(os.getenv("WHALE_LOOKBACK_HOURS", "72"))
WHALE_PER_CHANNEL_LIMIT = int(os.getenv("WHALE_PER_CHANNEL_LIMIT", "800"))
WHALE_INTERVAL_SEC = int(os.getenv("WHALE_INTERVAL_SEC", "180"))
WHALE_MIN_USD = float(os.getenv("WHALE_MIN_USD", "20000"))
WHALE_MIN_BTC = float(os.getenv("WHALE_MIN_BTC", "5"))
WHALE_MIN_ETH = float(os.getenv("WHALE_MIN_ETH", "100"))
WHALE_MIN_USDT = float(os.getenv("WHALE_MIN_USDT", "1000000"))
WHALE_MIN_USDC = float(os.getenv("WHALE_MIN_USDC", "1000000"))
WHALE_LARGE_TX_USD = float(os.getenv("WHALE_LARGE_TX_USD", "500000"))

LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "market_features.csv")
