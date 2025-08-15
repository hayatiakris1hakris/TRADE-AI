import pandas as pd
import numpy as np

def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder RMA (smoothed moving average)"""
    a = series.to_numpy(dtype=float)
    out = np.full_like(a, np.nan)
    if len(a) < period:
        return pd.Series(out, index=series.index)
    first = np.nanmean(a[:period])
    out[period - 1] = first
    alpha = 1.0 / period
    for i in range(period, len(a)):
        out[i] = out[i - 1] + alpha * (a[i] - out[i - 1])
    return pd.Series(out, index=series.index)

def atr(series_high: pd.Series, series_low: pd.Series, series_close: pd.Series, period: int=5) -> pd.Series:
    """Wilder ATR(period) = RMA(True Range, period)"""
    hl = (series_high - series_low).abs()
    hc = (series_high - series_close.shift(1)).abs()
    lc = (series_low - series_close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return rma(tr, period)

def followline(
    df: pd.DataFrame,
    bb_period: int=20,
    bb_dev: float=2.0,
    atr_period: int=5,
    col_open: str='open',
    col_high: str='high',
    col_low: str='low',
    col_close: str='close',
    out_col: str='followline'
) -> pd.DataFrame:
    """
    df: columns must include open/high/low/close
    Adds columns:
      - atr_flin (ATR(atr_period))
      - followline
      - followline_buy, followline_sell (boolean)
    Not: BB bileşenleri lokal hesaplanır (bb_mid/std/u/l).
    """
    # Giriş serileri
    c = df[col_close].astype(float)
    h = df[col_high].astype(float)
    l = df[col_low].astype(float)

    # Bollinger (SMA + sample std ddof=1)
    bb_mid = c.rolling(bb_period).mean()
    bb_std = c.rolling(bb_period).std(ddof=1)
    bb_upper = bb_mid + bb_std * bb_dev
    bb_lower = bb_mid - bb_std * bb_dev

    # ATR (Wilder-RMA)
    atr_flin = atr(h, l, c, period=atr_period)

    # FollowLine dizi
    fl = np.full(len(df), np.nan, dtype=float)

    # Başlangıç indeksi: BB ve ATR hazır olduktan sonraki ilk bar
    # bb/atr ilk geçerli indexlerine göre belirlemek daha güvenli
    def first_valid_idx(s: pd.Series):
        idx = s.first_valid_index()
        return df.index.get_loc(idx) if idx is not None else None

    candidates = [first_valid_idx(bb_upper), first_valid_idx(bb_lower), first_valid_idx(atr_flin)]
    candidates = [i for i in candidates if i is not None]
    if not candidates:
        df[out_col] = fl
        df['followline_buy'] = False
        df['followline_sell'] = False
        return df

    start_idx = max(candidates)

    # Seed: Close
    fl[start_idx] = float(c.iloc[start_idx])

    # İteratif güncelleme
    for i in range(start_idx + 1, len(df)):
        prev = fl[i - 1]
        ci = float(c.iloc[i])
        li = float(l.iloc[i])
        hi = float(h.iloc[i])
        bb_u = bb_upper.iloc[i]
        bb_l = bb_lower.iloc[i]
        ai = atr_flin.iloc[i]

        if (not np.isnan(bb_u)) and (not np.isnan(ai)) and (ci > bb_u) and (li - ai > prev):
            fl[i] = li - ai
        elif (not np.isnan(bb_l)) and (not np.isnan(ai)) and (ci < bb_l) and (hi + ai < prev):
            fl[i] = hi + ai
        else:
            fl[i] = prev

    df[out_col] = fl
    df['atr_flin'] = atr_flin

    # Sinyaller — mevcut dosyadaki yaklaşımı koruyoruz
    prev_fl = df[out_col].shift(1)
    buy = (prev_fl < prev_fl.shift(1)) & (df[out_col] >= prev_fl)
    sell = (prev_fl > prev_fl.shift(1)) & (df[out_col] <= prev_fl)

    df['followline_buy'] = buy.fillna(False).astype(bool)
    df['followline_sell'] = sell.fillna(False).astype(bool)
    return df
