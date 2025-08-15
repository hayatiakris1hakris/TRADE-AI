import os
import re
import hashlib
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
from collections import deque, Counter
from telethon import TelegramClient
from telethon.tl.types import PeerChannel
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.functions.channels import JoinChannelRequest
from math import copysign, log1p, exp

# Borsa isimlerinin tespiti için liste ve regex
EXCHANGES = [
    'binance', 'coinbase', 'kraken', 'okx', 'okex', 'bybit', 'kucoin', 'huobi', 'htx',
    'gate', 'bitfinex', 'bitstamp', 'gemini', 'upbit', 'bithumb', 'mexc', 'poloniex',
    'bitget', 'okcoin', 'bingx'
]
EXCH_RE = re.compile(r'\b(' + '|'.join(EXCHANGES) + r')\b', re.I)

# Desteklenen varlık tickleri ve regex
ASSET_TICKERS = ['BTC', 'ETH', 'USDT', 'USDC', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'TRX', 'MATIC', 'DOT', 'AVAX', 'SHIB']
ASSET_RE = r'(?:' + '|'.join(ASSET_TICKERS) + r')'

@dataclass
class WhaleEvent:
    timestamp: datetime
    ts: datetime
    asset: str
    chain: str
    amount: float
    usd_value: float
    from_label: str
    to_label: str
    tx_id: str
    provider: str = 'telegram'
    note: str = 'msg'

def hash_tx(provider: str, raw: str) -> str:
    return hashlib.sha256(f'{provider}:{raw}'.encode()).hexdigest()[:24]

def normalize_text(text: str) -> str:
    t = text.replace('\xa0', ' ').replace('\u2009', ' ')
    return ' '.join(t.split())

def normalize_owner(s: str) -> str:
    s = (s or '').lower()
    m = EXCH_RE.search(s)
    return f'exchange:{m.group(1).lower()}' if m else 'unknown'

def asset_chain(asset: str) -> str:
    mapping = {'BTC': 'bitcoin', 'ETH': 'ethereum', 'USDT': 'multi', 'USDC': 'ethereum', 'BNB': 'bsc', 'SOL': 'solana'}
    return mapping.get(asset.upper(), asset.lower())

def asset_from_text(t: str) -> Optional[str]:
    # #BTC / $BTC etiketleri
    tags = re.findall(r'[#\$]([A-Za-z0-9]{2,10})\b', t)
    if tags and tags[0].upper() in ASSET_TICKERS:
        return tags[0].upper()
    # "1234 BTC" gibi miktar + varlık
    m = re.search(rf'\b[0-9][0-9,.\s]*\s*({ASSET_RE})\b', t, re.I)
    return m.group(1).upper() if m else None

def parse_num(s: str) -> Optional[float]:
    s = s.replace(',', '').strip().lower()
    mul = 1.0
    if s.endswith('k'):
        mul = 1_000.0
        s = s[:-1]
    elif s.endswith('m'):
        mul = 1_000_000.0
        s = s[:-1]
    elif s.endswith('b'):
        mul = 1_000_000_000.0
        s = s[:-1]
    try:
        return float(s) * mul
    except Exception:
        return None

def parse_amount_and_usd(text: str) -> Tuple[Optional[float], Optional[float]]:
    t = normalize_text(text)
    usd = None
    patterns = [
        r'\( \s*[~≈]?\s*\$?\s*([0-9][0-9,.\s]*\s*[KkMmBb]?)\s*(?:USD)?\s* \)',
        r'worth\s*\$?\s*([0-9][0-9,.\s]*\s*[KkMmBb]?)',
        r'\$?\s*([0-9][0-9,.\s]*\s*[KkMmBb]?)\s*USD\b',
        r'\$\s*([0-9][0-9,.\s]*\s*[KkMmBb]?)',
    ]
    for p in patterns:
        m = re.search(p, t, re.I)
        if m:
            usd = parse_num(m.group(1).strip())
            if usd is not None:
                break
    if usd is None:
        m2 = re.search(r'([0-9][0-9,.\s]*)\s*(million|billion)\s*USD', t, re.I)
        if m2:
            base = parse_num(m2.group(1).strip())
            if base is not None:
                usd = base * (1_000_000.0 if m2.group(2).lower() == 'million' else 1_000_000_000.0)

    amt = None
    m_amt = re.search(rf'\b([0-9][0-9,.\s]*\s*[KkMmBb]?)\s*({ASSET_RE})\b', t, re.I)
    if m_amt:
        amt = parse_num(m_amt.group(1).strip())
    return (amt, usd)

def detect_from_to_labels(text: str) -> Tuple[str, str]:
    t = normalize_text(text).lower()
    m_from_to = re.search(r'from\s+([A-Za-z0-9_\-\s#@]+?)\s+to\s+([A-Za-z0-9_\-\s#@]+)', t)
    from_label = normalize_owner(m_from_to.group(1) if m_from_to else '')
    to_label = normalize_owner(m_from_to.group(2) if m_from_to else '')
    if to_label == 'unknown':
        m_to = re.search(r'\sto\s+([A-Za-z0-9_\-\s#@]+)', t)
        to_label = normalize_owner(m_to.group(1) if m_to else '')
    if from_label == 'unknown':
        m_from = re.search(r'from\s+([A-Za-z0-9_\-\s#@]+)\s+', t)
        from_label = normalize_owner(m_from.group(1) if m_from else '')
    return (from_label or 'unknown', to_label or 'unknown')

class WhaleTracker:
    """
    Telegram'dan parse edilen WhaleEvent kayıtlarını tutar, metrik ve model özellikleri üretir.
    """

    def __init__(self, assets: Optional[List[str]], large_tx_usd: float = 1_000_000.0):
        self.assets = [a.upper() for a in assets] if assets else []
        self.events: Dict[str, deque] = {a: deque(maxlen=5000) for a in self.assets} if self.assets else {}
        self.seen: Dict[str, set] = {a: set() for a in self.assets} if self.assets else {}
        self.large_tx_usd = float(large_tx_usd)

    def ingest(self, events: List[WhaleEvent]) -> int:
        added = 0
        for e in events:
            key = e.asset.upper()
            if self.assets and key not in self.events:
                continue
            if key not in self.events:
                self.events[key] = deque(maxlen=5000)
                self.seen[key] = set()
            if e.tx_id in self.seen[key]:
                continue
            self.seen[key].add(e.tx_id)
            self.events[key].append(e)
            added += 1
        return added

    def _is_ex(self, label: str) -> bool:
        return isinstance(label, str) and label.startswith('exchange:')

    def _cutoff(self, base: str, window_min: int) -> List[WhaleEvent]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_min)
        return [e for e in self.events.get(base, []) if e.ts >= cutoff]

    def category_counts(self, symbol: str, window_min: int = 180) -> dict:
        base = symbol.split('/')[0].upper()
        evs = self._cutoff(base, window_min)
        counts = {
            'u2u': 0, 'u2ex': 0, 'ex2u': 0, 'ex2ex': 0,
            'u2u_usd': 0.0, 'u2ex_usd': 0.0, 'ex2u_usd': 0.0, 'ex2ex_usd': 0.0,
            'total': 0, 'last_event_age_min': float(window_min),
        }
        if not evs:
            return counts

        for e in evs:
            f_ex = self._is_ex(e.from_label)
            t_ex = self._is_ex(e.to_label)
            if not f_ex and not t_ex:
                counts['u2u'] += 1
                counts['u2u_usd'] += e.usd_value
            elif not f_ex and t_ex:
                counts['u2ex'] += 1
                counts['u2ex_usd'] += e.usd_value
            elif f_ex and not t_ex:
                counts['ex2u'] += 1
                counts['ex2u_usd'] += e.usd_value
            else:
                counts['ex2ex'] += 1
                counts['ex2ex_usd'] += e.usd_value

        counts['total'] = len(evs)
        last_age_min = (datetime.now(timezone.utc) - max(e.ts for e in evs)).total_seconds() / 60.0
        counts['last_event_age_min'] = float(last_age_min)
        return counts

    def features(self, symbol: str, window_min: int = 180) -> Dict[str, float]:
        cats = self.category_counts(symbol, window_min)
        inflow = cats['u2ex_usd']
        outflow = cats['ex2u_usd']
        net = inflow - outflow
        base = symbol.split('/')[0].upper()
        evs = self._cutoff(base, window_min)
        large = sum((1 for e in evs if e.usd_value >= self.large_tx_usd))
        net_log = 0.0 if net == 0 else copysign(log1p(abs(net)), net)
        recency = exp(-(cats['last_event_age_min'] / 60.0))
        total = max(1, cats['total'])
        ex2ex_share = cats['ex2ex'] / total
        u2u_share = cats['u2u'] / total
        return {
            'whale_ex_inflow_usd': float(inflow),
            'whale_ex_outflow_usd': float(outflow),
            'whale_netflow_usd': float(net),
            'whale_netflow_log': float(net_log),
            'whale_large_tx_count': float(large),
            'whale_recency_score': float(recency),
            'whale_ex2ex_share': float(ex2ex_share),
            'whale_unknown_share': float(u2u_share),
            'whale_last_event_age_min': float(cats['last_event_age_min']),
            'whale_events_in_window': float(cats['total']),
        }

    def last_events(self, symbol: str, n: int = 3) -> List[WhaleEvent]:
        base = symbol.split('/')[0].upper()
        evs = list(self.events.get(base, []))
        return evs[-n:] if evs else []

class TelegramWhaleService:
    """
    Telegram'dan whale mesajlarını periyodik çeken ve WhaleTracker'a işleyen servis.
    """

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        channels: List[str],
        tracker: WhaleTracker,
        lookback_hours: int = 72,
        per_channel_limit: int = 800,
        interval_sec: int = 180,
        thresholds: Dict[str, float] | None = None,
    ):
        self.api_id = api_id
        self.api_hash = api_hash
        self.channels = channels
        self.tracker = tracker
        self.lookback_hours = lookback_hours
        self.per_channel_limit = per_channel_limit
        self.interval_sec = interval_sec
        self.thresholds = thresholds or {
            'min_usd': 20000.0,
            'min_btc': 5.0,
            'min_eth': 100.0,
            'min_usdt': 1_000_000.0,
            'min_usdc': 1_000_000.0,
        }
        self.client: Optional[TelegramClient] = None
        self.running = False
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        os.makedirs('sessions', exist_ok=True)
        self.client = TelegramClient('sessions/whale_service_session', self.api_id, self.api_hash)
        await self.client.connect()
        # Yetkili değilse etkileşimli login
        try:
            if not await self.client.is_user_authorized():
                print('[TG] İlk giriş gerekiyor. Lütfen telefon ve OTP girin.')
                phone = input('Telefon numaranız (örn +90...): ').strip()
                await self.client.send_code_request(phone)
                code = input('Telegram doğrulama kodu: ').strip()
                await self.client.sign_in(phone=phone, code=code)
                print('[TG] Giriş başarılı.')
        except Exception as e:
            print(f'[TG] Login hata: {e}')
        self.running = True
        self.task = asyncio.create_task(self._loop())

    async def stop(self):
        self.running = False
        if self.task:
            try:
                await asyncio.wait([self.task], timeout=5)
            except Exception:
                pass
        if self.client:
            try:
                await self.client.disconnect()
            except Exception:
                pass
        self.task = None
        self.client = None

    async def _loop(self):
        fetcher = WhaleTelegramFetcher(self.client, self.channels)
        while self.running:
            try:
                msgs = await fetcher.fetch_recent_messages(self.lookback_hours, self.per_channel_limit)
                (events, _, _) = parse_events_from_messages(
                    msgs, None,
                    self.thresholds['min_usd'],
                    self.thresholds['min_btc'],
                    self.thresholds['min_eth'],
                    self.thresholds['min_usdt'],
                    self.thresholds['min_usdc'],
                )
                added = self.tracker.ingest(events)
                print(f'[WhaleService] added events: {added}')
            except Exception as e:
                print(f'[WhaleService] error: {e}')
            await asyncio.sleep(self.interval_sec)

class WhaleTelegramFetcher:
    """
    Telegram kanallarından mesaj çeker (Telethon).
    """

    def __init__(self, client: TelegramClient, channels: List[str]):
        self.client = client
        self.channels = channels

    async def ensure_join(self, username: str) -> Optional[PeerChannel]:
        try:
            entity = await self.client.get_entity(username)
            return entity
        except Exception:
            try:
                await self.client(JoinChannelRequest(username))
                entity = await self.client.get_entity(username)
                return entity
            except Exception:
                print(f'[WARN] Could not join/access channel: {username}')
                return None

    async def fetch_recent_messages(self, lookback_hours: int = 12, per_channel_limit: int = 400) -> List[Dict]:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=lookback_hours)
        all_msgs = []
        for ch_username in self.channels:
            entity = await self.ensure_join(ch_username)
            if not entity:
                continue
            offset_id, fetched = 0, 0
            while fetched < per_channel_limit:
                try:
                    hist = await self.client(GetHistoryRequest(
                        peer=entity, limit=100, offset_date=None, offset_id=offset_id,
                        max_id=0, min_id=0, add_offset=0, hash=0
                    ))
                except Exception as e:
                    print(f'[WARN] history fetch error for {ch_username}: {e}')
                    break
                if not hist.messages:
                    break
                batch = 0
                for msg in hist.messages:
                    txt = getattr(msg, 'message', None)
                    if not txt:
                        continue
                    # Telethon message.date genelde tz-aware (UTC). Yine de garantiye alalım.
                    dt = msg.date
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt < start_time:
                        continue
                    all_msgs.append({'date': dt, 'text': txt, 'channel': ch_username, 'id': msg.id})
                    batch += 1
                if batch == 0:
                    break
                offset_id = hist.messages[-1].id
                fetched += batch
        print(f'[INFO] Fetched {len(all_msgs)} messages from Telegram (lookback={lookback_hours}h)')
        return all_msgs

def parse_events_from_messages(
    msgs: List[Dict],
    assets_whitelist: Optional[List[str]],
    min_usd: float, min_btc: float, min_eth: float, min_usdt: float, min_usdc: float
):
    """
    Telegram mesajlarını WhaleEvent listesine çevirir. Ayrıca skip nedenleri ve varlık sayımlarını döndürür.
    """
    whitelist = set((a.upper() for a in assets_whitelist)) if assets_whitelist else None
    events: List[WhaleEvent] = []
    skip_reasons = Counter()
    asset_counts = Counter()

    for m in msgs:
        text_raw = m.get('text') or ''
        if not text_raw:
            skip_reasons['no_text'] += 1
            continue
        text = normalize_text(text_raw)
        a = asset_from_text(text)
        if not a:
            skip_reasons['no_asset'] += 1
            continue
        asset_counts[a] += 1

        if whitelist and a not in whitelist:
            skip_reasons['not_whitelisted'] += 1
            continue

        (amt, usd) = parse_amount_and_usd(text)
        # Stablecoin'de USD yoksa miktarı USD say
        if usd is None and a in ('USDT', 'USDC') and amt:
            usd = amt
        if usd is None and amt is None:
            skip_reasons['no_amount_usd'] += 1
            continue

        pass_th = (
            (usd is not None and usd >= min_usd) or
            (a == 'BTC' and (amt or 0.0) >= min_btc) or
            (a == 'ETH' and (amt or 0.0) >= min_eth) or
            (a == 'USDT' and (amt or 0.0) >= min_usdt) or
            (a == 'USDC' and (amt or 0.0) >= min_usdc)
        )
        if not pass_th:
            skip_reasons['below_threshold'] += 1
            continue

        (from_label, to_label) = detect_from_to_labels(text)
        tx_id = hash_tx('telegram', f"{m['channel']}:{m['id']}")
        # ÖNEMLİ: timestamp ve ts alanlarını UTC-aware tarih ile doldur
        dt = m['date']
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        events.append(WhaleEvent(
            timestamp=dt,
            ts=dt,
            asset=a,
            chain=asset_chain(a),
            amount=float(amt or 0.0),
            usd_value=float(usd or 0.0),
            from_label=from_label,
            to_label=to_label,
            tx_id=tx_id
        ))

    return (events, skip_reasons, asset_counts)
