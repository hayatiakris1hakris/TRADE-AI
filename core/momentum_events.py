# file: core/momentum_events.py
from dataclasses import dataclass
from typing import Dict, Optional, Any
import pandas as pd
from datetime import datetime, timedelta, timezone

@dataclass
class MomentumEvent:
    symbol: str
    opened_at: pd.Timestamp  # tz-aware
    direction: str
    open_price: float
    features: Any

class MomentumEventTracker:
    def __init__(self, look_ahead_minutes: int = 30):
        self.look_ahead = timedelta(minutes=look_ahead_minutes)
        self.active: Dict[str, MomentumEvent] = {}

    def maybe_trigger(self, symbol: str, ts: pd.Timestamp, df_tail: pd.DataFrame) -> Optional[str]:
        # ts: df.index[-1], tz-aware varsayımı
        if len(df_tail) < 3:
            return None
        # Basit tetikleyici örneği
        last = df_tail.iloc[-1]
        prev = df_tail.iloc[-2]
        if last.get('volume_ratio', 0) > 1.5 and last.get('returns', 0) > 0.004 and prev.get('returns', 0) > 0:
            return 'up'
        if last.get('volume_ratio', 0) > 1.5 and last.get('returns', 0) < -0.004 and prev.get('returns', 0) < 0:
            return 'down'
        return None

    def open_event(self, symbol: str, ts: pd.Timestamp, price: float, direction: str, features: Any):
        # ts tz-aware
        self.active[symbol] = MomentumEvent(symbol, ts, direction, price, features)

    def resolve_due(self, now_utc: datetime, symbol: str, last_price: float, observer) -> Optional[Dict]:
        # now_utc tz-aware datetime
        ev = self.active.get(symbol)
        if not ev:
            return None
        # ev.opened_at tz-aware Timestamp → .to_pydatetime()
        if now_utc >= ev.opened_at.to_pydatetime() + self.look_ahead:
            # etiketle ve observer'a gönder
            lbl = int((ev.direction == 'up' and last_price > ev.open_price) or
                      (ev.direction == 'down' and last_price < ev.open_price))
            try:
                observer.observe_event(ev.features, lbl)
            except Exception:
                pass
            del self.active[symbol]
            return {'label': lbl, 'held_min': self.look_ahead.total_seconds() / 60.0}
        return None
