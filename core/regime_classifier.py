import os
from typing import Dict, List
import numpy as np
import joblib

REGIME_LABELS = ['trend_up', 'trend_down', 'range', 'volatile']
FEATURES: List[str] = [
    'returns', 'rsi', 'macd_hist', 'bb_position', 'volume_ratio',
    'sma_20', 'sma_50', 'ema_20', 'ema_50',
    'whale_netflow_log', 'whale_large_tx_count', 'whale_recency_score', 'whale_ex2ex_share',
    'news_sentiment', 'policy_risk_score', 'analyst_consensus'
]

class RegimeClassifier:

    def __init__(self, model_path: str = 'models/regime_clf.joblib'):
        self.model_path = model_path
        self.model = joblib.load(model_path) if os.path.exists(model_path) else None


    def _vec(self, row: Dict) -> np.ndarray:
        v = np.array([row.get(k, 0.0) for k in FEATURES], dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v.reshape(1, -1)


    def predict_proba(self, row: Dict) -> Dict[str, float]:
        if self.model is None:
            ema20 = float(row.get('ema_20', 0.0) or 0.0)
            ema50 = float(row.get('ema_50', 0.0) or 0.0)
            rsi = float(row.get('rsi', 50.0) or 50.0)
            base = {'trend_up': 0.25, 'trend_down': 0.25, 'range': 0.35, 'volatile': 0.15}
            if ema20 > ema50 and rsi > 55:
                base['trend_up'] += 0.2; base['range'] -= 0.1
            elif ema20 < ema50 and rsi < 45:
                base['trend_down'] += 0.2; base['range'] -= 0.1
            s = sum(base.values())
            return {k: v / s for (k, v) in base.items()}
        prob = self.model.predict_proba(self._vec(row))[0]
        return {REGIME_LABELS[i]: float(prob[i]) for i in range(len(REGIME_LABELS))}

    def predict_label(self, row: Dict) -> str:
        p = self.predict_proba(row)
        return max(p, key=p.get)
