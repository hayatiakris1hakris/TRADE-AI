import os
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, mean_absolute_error
import joblib

CONT_FEATURES = [
    'returns', 'volume_ratio', 'rsi', 'macd_hist', 'bb_position',
    'atr', 'spread_percentage',
    'whale_netflow_log', 'whale_large_tx_count', 'whale_recency_score', 'whale_ex2ex_share',
    'news_sentiment', 'policy_risk_score', 'analyst_consensus',
    'reg_trend_up', 'reg_trend_down', 'reg_range', 'reg_volatile',
    'side_code'
]

def ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if 'returns' not in df: df['returns'] = df['close'].pct_change()
    for p in [20, 50]:
        if f'sma_{p}' not in df: df[f'sma_{p}'] = df['close'].rolling(p).mean()
        if f'ema_{p}' not in df: df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
    if 'rsi' not in df:
        d = df['close'].diff(); gain = d.where(d > 0, 0).rolling(14).mean(); loss = (-d.where(d < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10); df['rsi'] = 100 - 100 / (1 + rs)
    if 'macd_hist' not in df:
        e1 = df['close'].ewm(span=12, adjust=False).mean(); e2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = e1 - e2; sig = macd.ewm(span=9, adjust=False).mean(); df['macd_hist'] = macd - sig
    if 'bb_position' not in df:
        mid = df['close'].rolling(20).mean(); std = df['close'].rolling(20).std()
        upper = mid + 2*std; lower = mid - 2*std
        df['bb_position'] = (df['close'] - lower) / (upper - lower + 1e-10)
    if 'volume_ratio' not in df:
        df['volume_sma'] = df['volume'].rolling(20).mean(); df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
    if 'atr' not in df:
        hl = (df['high'] - df['low']).abs()
        hc = (df['high'] - df['close'].shift(1)).abs()
        lc = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.ewm(span=14, adjust=False).mean()
    for c in ['whale_netflow_log','whale_large_tx_count','whale_recency_score','whale_ex2ex_share','news_sentiment','policy_risk_score','analyst_consensus','spread_percentage']:
        if c not in df: df[c] = 0.0
    for c in ['reg_trend_up','reg_trend_down','reg_range','reg_volatile']:
        if c not in df: df[c] = 0.25
    return df

def detect_events(df: pd.DataFrame, lookback_bars: int=5, ret_thr: float=0.015, volr_thr: float=1.3):
    ret_n = df['close'] / df['close'].shift(lookback_bars) - 1.0
    cond_up = (ret_n >= ret_thr) & (df['volume_ratio'] >= volr_thr)
    cond_dn = (ret_n <= -ret_thr) & (df['volume_ratio'] >= volr_thr)
    events = pd.Series(index=df.index, dtype='object')
    events[cond_up] = 'up'; events[cond_dn] = 'down'
    return events

def make_labels(df: pd.DataFrame, events: pd.Series, lookahead: int=30, mfe_atr_thr: float=0.5):
    idxs = events.dropna().index
    rows = []
    for t0 in idxs:
        side = events.loc[t0]
        i0 = df.index.get_loc(t0)
        i1 = i0 + 1; iL = min(i0 + lookahead, len(df) - 1)
        if i1 > iL: continue
        price0 = float(df['close'].iloc[i0]); atr0 = float(df['atr'].iloc[i0] or 0.0)
        if atr0 <= 0 or price0 <= 0: continue
        future = df.iloc[i1:iL+1]
        if side == 'up':
            mfe = float(future['close'].max() - price0)
        else:
            mfe = float(price0 - future['close'].min())
        mfe_atr = max(0.0, mfe) / atr0
        y_cont = 1 if mfe_atr >= mfe_atr_thr else 0
        row = {'timestamp': t0, 'side_code': 1.0 if side=='up' else -1.0, 'y_cont': y_cont, 'y_remain_atr': float(np.clip(mfe_atr, 0.0, 5.0))}
        for k in CONT_FEATURES:
            row[k] = float(df[k].iloc[i0]) if k in df.columns else (0.25 if k.startswith('reg_') else (row['side_code'] if k=='side_code' else 0.0))
        rows.append(row)
    return pd.DataFrame(rows).dropna()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--lookback', type=int, default=5)
    ap.add_argument('--ret_thr', type=float, default=0.015)
    ap.add_argument('--volr_thr', type=float, default=1.3)
    ap.add_argument('--lookahead', type=int, default=30)
    ap.add_argument('--mfe_atr_thr', type=float, default=0.5)
    ap.add_argument('--out_cls', default='models/cont_cls.joblib')
    ap.add_argument('--out_reg', default='models/cont_reg.joblib')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if 'timestamp' in df: df['timestamp'] = pd.to_datetime(df['timestamp']); df = df.sort_values('timestamp').set_index('timestamp')
    df = ensure_indicators(df)
    events = detect_events(df, args.lookback, args.ret_thr, args.volr_thr)
    ev = make_labels(df, events, args.lookahead, args.mfe_atr_thr)
    if ev.empty:
        print('No events; thresholds too strict?'); return
    X = ev[CONT_FEATURES].values; y_cls = ev['y_cont'].values.astype(int); y_reg = ev['y_remain_atr'].values.astype(float)

    base = HistGradientBoostingClassifier(max_depth=6, max_iter=400, learning_rate=0.05, validation_fraction=0.1)
    cls = CalibratedClassifierCV(base, method='isotonic', cv=3); cls.fit(X, y_cls)
    reg = HistGradientBoostingRegressor(max_depth=6, max_iter=400, learning_rate=0.05, validation_fraction=0.1); reg.fit(X, y_reg)

    os.makedirs('models', exist_ok=True)
    joblib.dump(cls, args.out_cls); joblib.dump(reg, args.out_reg)

    print('[Continuation] cls/reg saved.')
    print(classification_report(y_cls, cls.predict(X), digits=3))
    print('[REG] MAE:', mean_absolute_error(y_reg, reg.predict(X)))

if __name__ == '__main__':
    main()
