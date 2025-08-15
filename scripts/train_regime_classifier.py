import os
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

REGIME_LABELS = ['trend_up', 'trend_down', 'range', 'volatile']
FEATURES = [
    'returns', 'rsi', 'macd_hist', 'bb_position', 'volume_ratio',
    'sma_20', 'sma_50', 'ema_20', 'ema_50',
    'whale_netflow_log', 'whale_large_tx_count', 'whale_recency_score', 'whale_ex2ex_share',
    'news_sentiment', 'policy_risk_score', 'analyst_consensus'
]

def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
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
    for c in ['whale_netflow_log','whale_large_tx_count','whale_recency_score','whale_ex2ex_share','news_sentiment','policy_risk_score','analyst_consensus']:
        if c not in df: df[c] = 0.0
    return df

def make_labels(df: pd.DataFrame, lookahead: int=30, thr_sharpe: float=0.8) -> pd.Series:
    fwd_ret = df['close'].shift(-lookahead) / df['close'] - 1.0
    rv = df['close'].pct_change().rolling(lookahead).std()
    dir_sharpe = fwd_ret / (rv + 1e-10)
    vol_q = rv.rolling(200).quantile(0.7)
    is_volatile = rv > vol_q.fillna(rv.median())
    labels = pd.Series(index=df.index, dtype='object')
    labels[dir_sharpe >= thr_sharpe] = 'trend_up'
    labels[dir_sharpe <= -thr_sharpe] = 'trend_down'
    labels[labels.isna() & is_volatile.fillna(False)] = 'volatile'
    labels[labels.isna()] = 'range'
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='models/regime_clf.joblib')
    ap.add_argument('--lookahead', type=int, default=30)
    ap.add_argument('--thr_sharpe', type=float, default=0.8)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if 'timestamp' in df: df['timestamp'] = pd.to_datetime(df['timestamp']); df = df.sort_values('timestamp')
    df = ensure_features(df).dropna().copy()
    y = make_labels(df, args.lookahead, args.thr_sharpe).astype('category')
    X = df[FEATURES].fillna(0.0)

    base = HistGradientBoostingClassifier(max_depth=6, max_iter=400, learning_rate=0.05, validation_fraction=0.1)
    clf = CalibratedClassifierCV(base, method='sigmoid', cv=3)
    clf.fit(X, y)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    joblib.dump(clf, args.out)

    y_hat = clf.predict(X)
    print('[Regime] saved:', args.out)
    print(classification_report(y, y_hat, labels=REGIME_LABELS))
    print('CM (labels order):', REGIME_LABELS)
    print(confusion_matrix(y, y_hat, labels=REGIME_LABELS))

if __name__ == '__main__':
    main()
