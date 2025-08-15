# TRADE-AI

\# TRADE-AI



Setup

\- Python 3.10+

\- pip install -r requirements.txt

\- Create .env in project root:

&nbsp; TELEGRAM\_API\_ID=...

&nbsp; TELEGRAM\_API\_HASH=...

&nbsp; EXCHANGE\_ID=binance

&nbsp; TIMEFRAME=1m

&nbsp; TRADING\_SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT



Run

\- python app.py



Test whales

\- python -m external.whales.test\_whale\_service --mode once



Logs/Models

\- logs/market\_features.csv (closed-bar only)

\- models/\*.joblib (optional; fallback used if not present)









