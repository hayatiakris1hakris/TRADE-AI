# core/continuation_model.py
import os
from typing import Dict, List, Optional
import numpy as np
import joblib

CONT_FEATURES: List[str] = [
    'returns', 'volume_ratio', 'rsi', 'macd_hist', 'bb_position',
    'atr', 'spread_percentage',
    'whale_netflow_log', 'whale_large_tx_count', 'whale_recency_score', 'whale_ex2ex_share',
    'news_sentiment', 'policy_risk_score', 'analyst_consensus'
]
REGIME_KEYS = ['trend_up', 'trend_down', 'range', 'volatile']

class ContinuationModel:
    """
    Tahminler:
    - cont_prob: hareketin (t0 yönünde) lookahead içinde devam olasılığı
    - remain_atr: beklenen kalan hareket (ATR birimi)
    - expected_move_pct: yaklaşık beklenen yüzde (remain_atr * atr / price)
    """

    def __init__(self,
                 cls_path: str = 'models/cont_cls.joblib',
                 reg_path: str = 'models/cont_reg.joblib'):
        self.cls_path = cls_path
        self.reg_path = reg_path
        self.cls = joblib.load(cls_path) if os.path.exists(cls_path) else None
        self.reg = joblib.load(reg_path) if os.path.exists(reg_path) else None


    def _vec(self, row: Dict, regime_probs: Optional[Dict[str, float]], side: str) -> np.ndarray:
        x = np.array([row.get(k, 0.0) for k in CONT_FEATURES], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).tolist()
        rp = regime_probs or {}
        x.extend([float(rp.get(k, 0.0)) for k in REGIME_KEYS])
        side_code = 1.0 if side.lower().startswith(('long', 'up')) else -1.0
        x.append(side_code)
        v = np.array(x, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v.reshape(1, -1)

    def predict(self, row: Dict, side: str, regime_probs: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        v = self._vec(row, regime_probs, side)
        # Olasılık (model varsa)
        if self.cls is not None:
            cont_prob = float(self.cls.predict_proba(v)[0][1])
        else:
            # Fallback: trend bias + momentum kesiti
            ru = (regime_probs or {}).get('trend_up', 0.25)
            rd = (regime_probs or {}).get('trend_down', 0.25)
            mom = float(row.get('rsi', 50.0) or 50.0)
            macd = float(row.get('macd_hist', 0.0) or 0.0)
            base = 0.55 if side.startswith('long') else 0.45
            trend_bias = (ru - rd) if side.startswith('long') else (rd - ru)
            cont_prob = max(0.05, min(0.95, base + 0.25 * trend_bias + 0.1 * np.tanh(macd) + 0.1 * ((mom - 50.0) / 50.0)))
        # Kalan ATR (model varsa)
        if self.reg is not None:
            remain_atr = max(0.0, float(self.reg.predict(v)[0]))
        else:
            remain_atr = max(0.0, 0.6 if cont_prob > 0.65 else 0.2 if cont_prob > 0.5 else 0.1)
        atr = float(row.get('atr', 0.0) or 0.0)
        price = float(row.get('close', 0.0) or 0.0)
        expected_move_pct = float(remain_atr * (atr / price)) if (atr > 0 and price > 0) else 0.0
        return {
            'cont_prob': cont_prob,
            'remain_atr': remain_atr,
            'expected_move_pct': expected_move_pct
        }
