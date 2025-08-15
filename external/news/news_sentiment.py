# file: external/news/news_sentiment.py
from typing import Dict
class NewsSentimentAnalyzer:
    async def fetch(self, symbol: str) -> Dict[str, float]:
        return {'news_sentiment': 0.0}

# file: external/policy/policy_tracker.py
from typing import Dict
class PolicyRegTracker:
    async def fetch(self, symbol: str) -> Dict[str, float]:
        return {'policy_risk_score': 0.0}

# file: external/analysts/analyst_recs.py
from typing import Dict
class AnalystRecsTracker:
    async def fetch(self, symbol: str) -> Dict[str, float]:
        return {'analyst_consensus': 0.0}
