from typing import Dict

class PolicyRegTracker:
    async def fetch(self, symbol: str) -> Dict[str, float]:
        return {'policy_risk_score': 0.0}
