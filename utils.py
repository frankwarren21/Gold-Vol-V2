# utils.py
from datetime import datetime
from config import *

def is_trading_time(ts: datetime) -> bool:
    """Check if timestamp is within 7:30–15:30 EST trading window"""
    hour = ts.hour
    minute = ts.minute
    if hour < TRADING_START_HOUR or hour > TRADING_END_HOUR:
        return False
    if hour == TRADING_START_HOUR and minute < TRADING_START_MIN:
        return False
    if hour == TRADING_END_HOUR and minute > TRADING_END_MIN:
        return False
    return True

def prepare_state(row_h, current_price, pattern_strength):
    """Create RL state vector"""
    trend_diff = (current_price - row_h['sma_50']) / row_h['sma_50'] if row_h['sma_50'] != 0 else 0
    vol_ratio = row_h['volume'] / row_h['avg_vol'] if row_h['avg_vol'] != 0 else 1.0
    time_frac = row_h.name.hour + row_h.name.minute / 60.0
    return [trend_diff, vol_ratio, time_frac, pattern_strength]
