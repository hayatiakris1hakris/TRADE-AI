from typing import Dict, Tuple

def choose_strategy(reg_probs: Dict[str, float], cont: Dict[str, float], side: str) -> Dict[str, object]:
    """
    Girdi:
      reg_probs: {'trend_up','trend_down','range','volatile'} olasılıkları
      cont: {'cont_prob','remain_atr','expected_move_pct'}
      side: 'long'|'short'
    Çıktı:
      {'strategy','size_mult','hold_min','sl_atr','tp_atr'}
    """
    trend_p = reg_probs.get('trend_up', 0.0) if side.startswith('long') else reg_probs.get('trend_down', 0.0)
    range_p = reg_probs.get('range', 0.0)
    vol_p = reg_probs.get('volatile', 0.0)
    cont_p = cont.get('cont_prob', 0.5)
    rem_atr = cont.get('remain_atr', 0.2)

    # Varsayılan yok
    strat = None
    size_mult = 1.0
    hold_min = 0
    sl_atr = 0.6
    tp_atr = max(0.6, min(2.0, rem_atr))

    # Trend koşulu
    if trend_p >= 0.45 and cont_p >= 0.6:
        strat = 'trend'
        size_mult = 1.0 + 0.5 * (cont_p - 0.6) + 0.5 * min(1.0, rem_atr)
        hold_min = int(45 + 45 * min(1.0, rem_atr))
        sl_atr = 0.7
        tp_atr = max(tp_atr, 1.0)

    # Range koşulu
    elif range_p >= 0.45 and cont_p <= 0.55:
        strat = 'meanrev'
        size_mult = 0.8
        hold_min = 20
        sl_atr = 0.5
        tp_atr = min(0.8, rem_atr if rem_atr > 0 else 0.4)

    # Volatilite-breakout
    elif vol_p >= 0.4 and cont_p >= 0.6:
        strat = 'breakout'
        size_mult = 1.0
        hold_min = 30
        sl_atr = 0.8
        tp_atr = max(1.2, rem_atr)

    # Boundaries
    size_mult = max(0.3, min(1.8, size_mult))
    return {'strategy': strat, 'size_mult': float(size_mult), 'hold_min': int(hold_min), 'sl_atr': float(sl_atr), 'tp_atr': float(tp_atr)}
