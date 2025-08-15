from typing import Dict
from external.whales.telegram_whale_service import WhaleTracker
from external.news.news_sentiment import NewsSentimentAnalyzer
from external.policy.policy_tracker import PolicyRegTracker
from external.analysts.analyst_recs import AnalystRecsTracker

class SignalsHub:
    def __init__(self, whale_tracker: WhaleTracker):
        self.whales = whale_tracker
        self.news = NewsSentimentAnalyzer()
        self.policy = PolicyRegTracker()
        self.analyst = AnalystRecsTracker()

    async def build_features(self, symbol: str, window_min: int=180) -> Dict[str, float]:
        wf = self.whales.features(symbol, window_min=window_min)
        nf = await self.news.fetch(symbol)
        pf = await self.policy.fetch(symbol)
        af = await self.analyst.fetch(symbol)
        merged = {}
        merged.update(wf); merged.update(nf); merged.update(pf); merged.update(af)
        return merged
