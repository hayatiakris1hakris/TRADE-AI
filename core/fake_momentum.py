import numpy as np
from collections import deque
from sklearn.linear_model import SGDClassifier
from typing import Dict

class FakeMomentumDetector:
    def __init__(self):
        self.model = SGDClassifier(loss="log_loss", alpha=0.0005)
        self.is_warm = False
        self.buffer = deque(maxlen=5000)

    def build_features(self, row: Dict) -> np.ndarray:
        def sf(x):
            try:
                x = float(x)
            except Exception:
                return 0.0
            return 0.0 if not np.isfinite(x) else x

        return np.array([
            sf(row.get("returns", 0.0)),
            sf(row.get("volume_ratio", 0.0)),
            sf(row.get("rsi", 50.0)) / 100.0,
            sf(row.get("macd_hist", 0.0)),
            sf(row.get("bb_position", 0.0)),
            sf(row.get("whale_netflow_log", 0.0)),
            sf(row.get("whale_large_tx_count", 0.0)) / 10.0,
            sf(row.get("whale_recency_score", 0.0)),
            sf(row.get("whale_ex2ex_share", 0.0)),
            sf(row.get("whale_unknown_share", 0.0)),
        ], dtype=np.float32)

 
    def predict_proba(self, feat: np.ndarray) -> float:
        if not self.is_warm:
            return 0.5
        try:
            return float(self.model.predict_proba(feat.reshape(1, -1))[0][1])
        except Exception:
            return 0.5

    def observe_event(self, feat: np.ndarray, fake_label: int):
        self.buffer.append((feat, fake_label))
        X = np.array([x for (x, y) in self.buffer])
        y = np.array([y for (x, y) in self.buffer])
        if not self.is_warm:
            self.model.partial_fit(X, y, classes=np.array([0, 1]))
            self.is_warm = True
        else:
            self.model.partial_fit(X[-64:], y[-64:])
