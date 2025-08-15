from typing import Dict

class AnalystRecsTracker:
    async def fetch(self, symbol: str) -> Dict[str, float]:
        return {'analyst_consensus': 0.0}
