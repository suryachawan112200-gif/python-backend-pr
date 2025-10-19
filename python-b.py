import json
import random
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from uuid import uuid4
import time

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Input & Validation
class InputValidator:
    REQUIRED_FIELDS = {"coin", "market", "entry_price", "quantity", "position_type", "timeframe"}
    VALID_TIMEFRAMES = ["5m", "15m", "4h", "1d"]

    @staticmethod
    def validate_and_normalize(data):
        for field in InputValidator.REQUIRED_FIELDS:
            if field not in data:
                raise ValueError(f"Missing field: {field}")
        data["coin"] = data["coin"].upper().strip()
        data["market"] = data["market"].lower().strip()
        if data["market"] not in ["spot", "futures"]:
            raise ValueError("Market must be 'spot' or 'futures'")
        data["position_type"] = data["position_type"].lower().strip()
        data["timeframe"] = data["timeframe"].lower().strip()
        if data["timeframe"] not in InputValidator.VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe. Must be one of: {', '.join(InputValidator.VALID_TIMEFRAMES)}")
        data["risk_pct"] = data.get("risk_pct", 0.02)
        return data

# Indicator & Analysis Helpers
def get_interval_mapping(timeframe):
    mapping = {
        "5m": (5, 1.0, 10, 10, 20, 2.0),
        "15m": (15, 1.5, 14, 14, 20, 2.0),
        "4h": (240, 2.0, 14, 14, 20, 2.5),
        "1d": ("D", 3.0, 21, 21, 20, 2.5)
    }
    return mapping[timeframe]

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_ma(series, window):
    return series.rolling(window=window).mean()

def compute_atr(df, period=14):
    if len(df) < period:
        return 0.0
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1] if not tr.empty else 0.0

def compute_rsi(series, period=14, return_series=False):
    if len(series) < period:
        return pd.Series([50.0] * len(series), index=series.index) if return_series else 50.0
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - 100 / (1 + rs)
    return rsi if return_series else rsi.iloc[-1] if not rsi.empty else 50.0

def compute_adx(df, period=14):
    if len(df) < period:
        return 25.0
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff().where((high.diff() > low.diff()) & (high.diff() > 0), 0)
    minus_dm = -low.diff().where((low.diff() > high.diff()) & (low.diff() > 0), 0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean().replace(0, 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean().replace(0, 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.rolling(period).mean()
    return adx.iloc[-1] if not adx.empty else 25.0

def compute_macd(series, fast=12, slow=26, signal=9):
    if len(series) < max(fast, slow, signal):
        return 0.0, 0.0, 0.0
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = compute_ema(macd, signal)
    histogram = macd - signal_line
    return macd.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def compute_obv(df):
    if len(df) < 1:
        return 0.0
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv.iloc[-1] if not obv.empty else 0.0

def compute_hma(series, period=9):
    if len(series) < period:
        return series.iloc[-1]
    wma1 = series.rolling(period // 2).mean() * 2 - series.rolling(period).mean()
    hma = wma1.rolling(int(np.sqrt(period))).mean()
    return hma.iloc[-1] if not hma.empty else series.iloc[-1]

def compute_heikin_ashi(df):
    if len(df) < 1:
        return df
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = (df['open'].shift() + df['close'].shift()) / 2
    ha_open = ha_open.fillna((df['open'].iloc[0] + df['close'].iloc[0]) / 2)
    ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df['low'], ha_open, ha_close], axis=1).min(axis=1)
    ha_df = pd.DataFrame({
        'open': ha_open,
        'high': ha_high,
        'low': ha_low,
        'close': ha_close,
        'volume': df['volume']
    }, index=df.index)
    return ha_df

def compute_supertrend(df, period=7, multiplier=3.0):
    if len(df) < period:
        return df['close'].iloc[-1] if not df.empty else 0.0
    atr = compute_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    supertrend.iloc[0] = df['close'].iloc[0]
    direction.iloc[0] = 1
    
    for i in range(1, len(df)):
        if df['close'].iloc[i-1] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif df['close'].iloc[i-1] < supertrend.iloc[i-1]:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]
        
        if direction.iloc[i] == 1 and df['low'].iloc[i] < supertrend.iloc[i]:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        elif direction.iloc[i] == -1 and df['high'].iloc[i] > supertrend.iloc[i]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
    
    return supertrend.iloc[-1] if not supertrend.empty else df['close'].iloc[-1]

def detect_rsi_divergence(df, period=14):
    if len(df) < period + 2:
        return None
    close = df['close']
    rsi_series = compute_rsi(close, period, return_series=True)
    price_peaks = get_peak_indices(close)[-2:]
    price_troughs = get_trough_indices(close)[-2:]
    rsi_peaks = get_peak_indices(rsi_series)[-2:]
    rsi_troughs = get_trough_indices(rsi_series)[-2:]
    
    if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
        t1_idx = price_troughs[-2]
        t2_idx = price_troughs[-1]
        rt1_idx = rsi_troughs[-2]
        rt2_idx = rsi_troughs[-1]
        if abs(df.index.get_loc(t1_idx) - df.index.get_loc(rt1_idx)) <= 5 and abs(df.index.get_loc(t2_idx) - df.index.get_loc(rt2_idx)) <= 5:
            if close[t2_idx] < close[t1_idx] and rsi_series[rt2_idx] > rsi_series[rt1_idx]:
                return "Bullish Divergence"
    if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
        p1_idx = price_peaks[-2]
        p2_idx = price_peaks[-1]
        rp1_idx = rsi_peaks[-2]
        rp2_idx = rsi_peaks[-1]
        if abs(df.index.get_loc(p1_idx) - df.index.get_loc(rp1_idx)) <= 5 and abs(df.index.get_loc(p2_idx) - df.index.get_loc(rp2_idx)) <= 5:
            if close[p2_idx] > close[p1_idx] and rsi_series[rp2_idx] < rsi_series[rp1_idx]:
                return "Bearish Divergence"
    return None

def compute_pivot_points(df, period=50):
    if len(df) < 1:
        return [], [], df['close'].iloc[-1] if not df.empty else 0.0
    period = min(len(df), period)
    recent_df = df.tail(period)
    high = recent_df['high'].max()
    low = recent_df['low'].min()
    close = recent_df['close'].iloc[-1]
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return [float(s1), float(s2), float(s3)], [float(r1), float(r2), float(r3)], float(pivot)

def detect_support_resistance(df, current_price, window=3, lookback=100):
    if len(df) < window:
        return [current_price * 0.99], [current_price * 1.01]
    df_close = df['close']
    df_volume = df['volume']
    avg_volume = df_volume.rolling(20).mean().iloc[-1] if len(df_volume) >= 20 else df_volume.iloc[-1]
    lows = df['low'].rolling(window, center=True).min()
    highs = df['high'].rolling(window, center=True).max()
    candidate_supports = df['low'][(df['low'] == lows) & (df['volume'] > avg_volume * 1.0)].tail(lookback).unique().tolist()
    candidate_resistances = df['high'][(df['high'] == highs) & (df['volume'] > avg_volume * 1.0)].tail(lookback).unique().tolist()

    def count_touches(level, series, tol):
        return sum(abs(series - level) < level * tol)

    support_counts = {s: count_touches(s, df_close.tail(lookback), tol=0.005) for s in candidate_supports if candidate_supports}
    resistance_counts = {r: count_touches(r, df_close.tail(lookback), tol=0.005) for r in candidate_resistances if candidate_resistances}

    supports = [s for s in candidate_supports if support_counts.get(s, 0) >= 2] if candidate_supports else [current_price * 0.99]
    resistances = [r for r in candidate_resistances if resistance_counts.get(r, 0) >= 2] if candidate_resistances else [current_price * 1.01]

    pivot_supports, pivot_resistances, _ = compute_pivot_points(df)
    supports.extend(pivot_supports)
    resistances.extend(pivot_resistances)

    supports = sorted(list(set(map(float, supports))))
    resistances = sorted(list(set(map(float, resistances))))

    nearby_supports = sorted([s for s in supports if s < current_price], key=lambda s: abs(s - current_price))[:3]
    nearby_resistances = sorted([r for r in resistances if r > current_price], key=lambda r: abs(r - current_price))[:3]

    return nearby_supports or [current_price * 0.99], nearby_resistances or [current_price * 1.01]

def detect_candlestick_patterns(df):
    patterns = []
    if len(df) < 1:
        return patterns
    open_ = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    last_open = open_.iloc[-1]
    last_high = high.iloc[-1]
    last_low = low.iloc[-1]
    last_close = close.iloc[-1]
    last_volume = volume.iloc[-1]
    avg_volume = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else last_volume
    body = abs(last_open - last_close)
    upper_shadow = last_high - max(last_open, last_close)
    lower_shadow = min(last_open, last_close) - last_low
    total_range = last_high - last_low if last_high > last_low else 1e-6

    if body / total_range < 0.05 and last_volume > avg_volume * 1.2:
        patterns.append("Doji")
    if lower_shadow > 2 * body and upper_shadow < 0.1 * body and last_close > last_open and last_volume > avg_volume:
        patterns.append("Hammer")
    if upper_shadow > 2 * body and lower_shadow < 0.1 * body and last_close > last_open and last_volume > avg_volume:
        patterns.append("Inverted Hammer")
    if upper_shadow > 2 * body and lower_shadow < 0.1 * body and last_close < last_open and last_volume > avg_volume:
        patterns.append("Shooting Star")
    if lower_shadow > 2 * body and upper_shadow < 0.1 * body and last_close < last_open and last_volume > avg_volume:
        patterns.append("Hanging Man")
    if body / total_range < 0.3 and upper_shadow > body and lower_shadow > body and last_volume > avg_volume:
        patterns.append("Spinning Top")
    if upper_shadow < 0.05 * body and lower_shadow < 0.05 * body and last_close > last_open and last_volume > avg_volume:
        patterns.append("Bullish Marubozu")
    if upper_shadow < 0.05 * body and lower_shadow < 0.05 * body and last_close < last_open and last_volume > avg_volume:
        patterns.append("Bearish Marubozu")
    if lower_shadow < 0.05 * body and last_close > last_open and last_open <= last_low + 0.001 * total_range and last_volume > avg_volume:
        patterns.append("Bullish Belt Hold")
    if upper_shadow < 0.05 * body and last_close < last_open and last_open >= last_high - 0.001 * total_range and last_volume > avg_volume:
        patterns.append("Bearish Belt Hold")

    if len(df) >= 2:
        prev_open = open_.iloc[-2]
        prev_close = close.iloc[-2]
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        prev_volume = volume.iloc[-2]
        if prev_close < prev_open and last_close > prev_open and last_open < prev_close and last_close > last_open and last_volume > prev_volume * 1.1:
            patterns.append("Bullish Engulfing")
        if prev_close > prev_open and last_close < prev_open and last_open > prev_close and last_close < last_open and last_volume > prev_volume * 1.1:
            patterns.append("Bearish Engulfing")
        if prev_close < prev_open and last_open < prev_close and last_close > (prev_open + prev_close) / 2 and last_close < prev_open and last_volume > prev_volume:
            patterns.append("Piercing Line")
        if prev_close > prev_open and last_open > prev_close and last_close < (prev_open + prev_close) / 2 and last_close > prev_open and last_volume > prev_volume:
            patterns.append("Dark Cloud Cover")
        if prev_close < prev_open and last_open > prev_close and last_close < prev_open and last_close > last_open and last_volume > prev_volume:
            patterns.append("Bullish Harami")
        if prev_close > prev_open and last_open < prev_close and last_close > prev_open and last_close < last_open and last_volume > prev_volume:
            patterns.append("Bearish Harami")
        if abs(last_low - prev_low) < 0.001 * last_low and prev_close < prev_open and last_close > last_open and last_volume > avg_volume:
            patterns.append("Tweezer Bottom")
        if abs(last_high - prev_high) < 0.001 * last_high and prev_close > prev_open and last_close < last_open and last_volume > avg_volume:
            patterns.append("Tweezer Top")

    if len(df) >= 3:
        p2_open = open_.iloc[-3]
        p2_close = close.iloc[-3]
        p1_open = open_.iloc[-2]
        p1_close = close.iloc[-2]
        p1_volume = volume.iloc[-2]
        if p2_close < p2_open and abs(p1_close - p1_open) < 0.3 * abs(p2_close - p2_open) and last_close > last_open and last_close > (p2_open + p2_close) / 2 and last_volume > p1_volume * 1.1:
            patterns.append("Morning Star")
        if p2_close > p2_open and abs(p1_close - p1_open) < 0.3 * abs(p2_close - p2_open) and last_close < last_open and last_close < (p2_open + p2_close) / 2 and last_volume > p1_volume * 1.1:
            patterns.append("Evening Star")
        if all(close.iloc[-i] > open_.iloc[-i] for i in range(1, 4)) and close.iloc[-2] > close.iloc[-3] and close.iloc[-1] > close.iloc[-2] and last_volume > avg_volume:
            patterns.append("Three White Soldiers")
        if all(close.iloc[-i] < open_.iloc[-i] for i in range(1, 4)) and close.iloc[-2] < close.iloc[-3] and close.iloc[-1] < close.iloc[-2] and last_volume > avg_volume:
            patterns.append("Three Black Crows")

    return patterns

def get_peak_indices(series):
    return series.index[(series.shift(1) < series) & (series.shift(-1) < series)].tolist()

def get_trough_indices(series):
    return series.index[(series.shift(1) > series) & (series.shift(-1) > series)].tolist()

def detect_chart_patterns(df, timeframe):
    if len(df) < 10:
        return [], {}, {}
    df_close = df['close']
    patterns = []
    pattern_targets = {}
    pattern_sls = {}
    peak_indices = get_peak_indices(df_close)[-5:]
    trough_indices = get_trough_indices(df_close)[-5:]
    current = df_close.iloc[-1]
    index_positions = {idx: i for i, idx in enumerate(df.index)}

    if len(peak_indices) >= 2:
        p1_idx, p2_idx = peak_indices[-2], peak_indices[-1]
        p1, p2 = df_close[p1_idx], df_close[p2_idx]
        p1_pos, p2_pos = index_positions[p1_idx], index_positions[p2_idx]
        if abs(p1 - p2) / ((p1 + p2) / 2) < 0.015:
            patterns.append("Double Top")
            neckline = min(df_close.iloc[p1_pos:p2_pos]) if len(df_close.iloc[p1_pos:p2_pos]) > 0 else current
            height = p1 - neckline
            pattern_targets["Double Top"] = neckline - height
            pattern_sls["Double Top"] = max(p1, p2) + height * 0.1
            if not (current > pattern_targets["Double Top"] and current < pattern_sls["Double Top"]):
                patterns.remove("Double Top")
                pattern_targets.pop("Double Top", None)
                pattern_sls.pop("Double Top", None)

    if len(trough_indices) >= 2:
        t1_idx, t2_idx = trough_indices[-2], trough_indices[-1]
        t1, t2 = df_close[t1_idx], df_close[t2_idx]
        t1_pos, t2_pos = index_positions[t1_idx], index_positions[t2_idx]
        if abs(t1 - t2) / ((t1 + t2) / 2) < 0.015:
            patterns.append("Double Bottom")
            neckline = max(df_close.iloc[t1_pos:t2_pos]) if len(df_close.iloc[t1_pos:t2_pos]) > 0 else current
            height = neckline - t1
            pattern_targets["Double Bottom"] = neckline + height
            pattern_sls["Double Bottom"] = min(t1, t2) - height * 0.1
            if not (current < pattern_targets["Double Bottom"] and current > pattern_sls["Double Bottom"]):
                patterns.remove("Double Bottom")
                pattern_targets.pop("Double Bottom", None)
                pattern_sls.pop("Double Bottom", None)

    if len(peak_indices) >= 3:
        p1_idx, p2_idx, p3_idx = peak_indices[-3:]
        p1, p2, p3 = df_close[p1_idx], df_close[p2_idx], df_close[p3_idx]
        p1_pos, p2_pos, p3_pos = index_positions[p1_idx], index_positions[p2_idx], index_positions[p3_idx]
        if max(abs(p1 - p2), abs(p2 - p3), abs(p1 - p3)) / ((p1 + p2 + p3) / 3) < 0.015:
            patterns.append("Triple Top")
            neckline = df_close.iloc[p1_pos:p3_pos+1].min() if len(df_close.iloc[p1_pos:p3_pos+1]) > 0 else current
            height = (p1 + p2 + p3)/3 - neckline
            pattern_targets["Triple Top"] = neckline - height
            pattern_sls["Triple Top"] = max(p1, p2, p3) + height * 0.1
            if not (current > pattern_targets["Triple Top"] and current < pattern_sls["Triple Top"]):
                patterns.remove("Triple Top")
                pattern_targets.pop("Triple Top", None)
                pattern_sls.pop("Triple Top", None)
        if p2 > p1 and p2 > p3 and abs(p1 - p3) / p2 < 0.02:
            patterns.append("Head and Shoulders")
            neckline = (p1 + p3) / 2
            height = p2 - neckline
            pattern_targets["Head and Shoulders"] = neckline - height
            pattern_sls["Head and Shoulders"] = p2 + height * 0.1
            if not (current > pattern_targets["Head and Shoulders"] and current < pattern_sls["Head and Shoulders"]):
                patterns.remove("Head and Shoulders")
                pattern_targets.pop("Head and Shoulders", None)
                pattern_sls.pop("Head and Shoulders", None)

    if len(trough_indices) >= 3:
        t1_idx, t2_idx, t3_idx = trough_indices[-3:]
        t1, t2, t3 = df_close[t1_idx], df_close[t2_idx], df_close[t3_idx]
        t1_pos, t2_pos, t3_pos = index_positions[t1_idx], index_positions[t2_idx], index_positions[t3_idx]
        if max(abs(t1 - t2), abs(t2 - t3), abs(t1 - t3)) / ((t1 + t2 + t3) / 3) < 0.015:
            patterns.append("Triple Bottom")
            neckline = df_close.iloc[t1_pos:t3_pos+1].max() if len(df_close.iloc[t1_pos:t3_pos+1]) > 0 else current
            height = neckline - (t1 + t2 + t3)/3
            pattern_targets["Triple Bottom"] = neckline + height
            pattern_sls["Triple Bottom"] = min(t1, t2, t3) - height * 0.1
            if not (current < pattern_targets["Triple Bottom"] and current > pattern_sls["Triple Bottom"]):
                patterns.remove("Triple Bottom")
                pattern_targets.pop("Triple Bottom", None)
                pattern_sls.pop("Triple Bottom", None)
        if t2 < t1 and t2 < t3 and abs(t1 - t3) / abs(t2) < 0.02:
            patterns.append("Inverse Head and Shoulders")
            neckline = (t1 + t3) / 2
            height = neckline - t2
            pattern_targets["Inverse Head and Shoulders"] = neckline + height
            pattern_sls["Inverse Head and Shoulders"] = t2 - height * 0.1
            if not (current < pattern_targets["Inverse Head and Shoulders"] and current > pattern_sls["Inverse Head and Shoulders"]):
                patterns.remove("Inverse Head and Shoulders")
                pattern_targets.pop("Inverse Head and Shoulders", None)
                pattern_sls.pop("Inverse Head and Shoulders", None)

    if len(peak_indices) >= 2 and len(trough_indices) >= 2:
        p1_idx, p2_idx = peak_indices[-2], peak_indices[-1]
        t1_idx, t2_idx = trough_indices[-2], trough_indices[-1]
        p1, p2 = df_close[p1_idx], df_close[p2_idx]
        t1, t2 = df_close[t1_idx], df_close[t2_idx]
        p1_pos, p2_pos = index_positions[p1_idx], index_positions[p2_idx]
        t1_pos, t2_pos = index_positions[t1_idx], index_positions[t2_idx]
        slope_peak = (p2 - p1) / (p2_pos - p1_pos + 1e-6)
        slope_trough = (t2 - t1) / (t2_pos - t1_pos + 1e-6)

        if abs(p2 - p1) / df_close.mean() < 0.01 and t2 > t1:
            patterns.append("Ascending Triangle")
            height = p1 - t1
            pattern_targets["Ascending Triangle"] = p2 + height
            pattern_sls["Ascending Triangle"] = min(t1, t2) - height * 0.1
            if not (current < pattern_targets["Ascending Triangle"]):
                patterns.remove("Ascending Triangle")
                pattern_targets.pop("Ascending Triangle", None)
                pattern_sls.pop("Ascending Triangle", None)

        if abs(t2 - t1) / df_close.mean() < 0.01 and p2 < p1:
            patterns.append("Descending Triangle")
            height = p1 - t1
            pattern_targets["Descending Triangle"] = t2 - height
            pattern_sls["Descending Triangle"] = max(p1, p2) + height * 0.1
            if not (current > pattern_targets["Descending Triangle"]):
                patterns.remove("Descending Triangle")
                pattern_targets.pop("Descending Triangle", None)
                pattern_sls.pop("Descending Triangle", None)

        if p2 < p1 and t2 > t1:
            patterns.append("Symmetrical Triangle")
        if p2 < p1 and t2 < t1:
            if abs(slope_peak - slope_trough) < 0.001:
                patterns.append("Descending Channel")
            elif p1 - t1 > p2 - t2:
                patterns.append("Falling Wedge")
                height = p1 - t1
                pattern_targets["Falling Wedge"] = df_close.iloc[-1] + height
                pattern_sls["Falling Wedge"] = min(t1, t2) - height * 0.1
                if not (current < pattern_targets["Falling Wedge"] and current > pattern_sls["Falling Wedge"]):
                    patterns.remove("Falling Wedge")
                    pattern_targets.pop("Falling Wedge", None)
                    pattern_sls.pop("Falling Wedge", None)
        if p2 > p1 and t2 > t1:
            if abs(slope_peak - slope_trough) < 0.001:
                patterns.append("Ascending Channel")
            elif p2 - t2 < p1 - t1:
                patterns.append("Rising Wedge")
                height = p1 - t1
                pattern_targets["Rising Wedge"] = df_close.iloc[-1] - height
                pattern_sls["Rising Wedge"] = max(p1, p2) + height * 0.1
                if not (current > pattern_targets["Rising Wedge"] and current < pattern_sls["Rising Wedge"]):
                    patterns.remove("Rising Wedge")
                    pattern_targets.pop("Rising Wedge", None)
                    pattern_sls.pop("Rising Wedge", None)

    return patterns, pattern_targets, pattern_sls

# Analysis Engine
class AnalysisEngine:
    bullish_candles = ["Hammer", "Inverted Hammer", "Bullish Engulfing", "Piercing Line", "Morning Star", "Bullish Harami", "Three White Soldiers", "Bullish Belt Hold", "Bullish Marubozu"]
    bearish_candles = ["Shooting Star", "Hanging Man", "Bearish Engulfing", "Dark Cloud Cover", "Evening Star", "Bearish Harami", "Three Black Crows", "Bearish Belt Hold", "Bearish Marubozu"]
    bullish_charts = ["Inverse Head and Shoulders", "Double Bottom", "Triple Bottom", "Ascending Triangle", "Falling Wedge", "Ascending Channel"]
    bearish_charts = ["Head and Shoulders", "Double Top", "Triple Top", "Descending Triangle", "Rising Wedge", "Descending Channel"]
    high_prob_patterns = ["Triple Bottom", "Inverse Head and Shoulders", "Triple Top", "Head and Shoulders"]

    @staticmethod
    def profitability(position_type, entry_price, current_price, quantity):
        pl = (current_price - entry_price) * quantity if position_type == "long" else (entry_price - current_price) * quantity
        pl_pct = (pl / (entry_price * quantity)) * 100 if entry_price * quantity != 0 else 0
        comment = "Profit above avg move." if pl_pct > 0.5 else "Loss above avg move." if pl_pct < -0.5 else "Standard profit/loss."
        return f"{pl:+.4f} ({pl_pct:.2f}%)", comment

    @staticmethod
    def market_trend(df_ohlcv, df_higher_tf, current_price, timeframe, market_type):
        if len(df_ohlcv) < 1:
            return "sideways", 60, "Insufficient data for trend analysis.", 0.0
        close = df_ohlcv['close']
        _, multiplier, rsi_period, atr_period, _, _ = get_interval_mapping(timeframe)
        ema_short = compute_ema(close, 50).iloc[-1]
        ema_long = compute_ema(close, 200).iloc[-1]
        atr = compute_atr(df_ohlcv, atr_period)
        hma = compute_hma(close, 9)

        price_24h_ago = df_higher_tf['close'].iloc[-1] if not df_higher_tf.empty else close.iloc[-2] if len(close) > 1 else close.iloc[-1]
        daily_move_pct = ((current_price - price_24h_ago) / price_24h_ago) * 100 if price_24h_ago != 0 else 0.0

        trend = "sideways"
        confidence = 60
        comment = "Market moving sideways."

        move_threshold = 1.5 if market_type == "futures" else 1.0
        low_move_threshold = 0.5 if market_type == "spot" else 1.5

        if daily_move_pct > move_threshold and current_price > hma:
            trend = "bullish"
            confidence = 90 if daily_move_pct > 2.0 else 80
            comment = f"Strong 24h up ({daily_move_pct:.2f}%) with bullish HMA."
        elif daily_move_pct < -move_threshold and current_price < hma:
            trend = "bearish"
            confidence = 90 if daily_move_pct < -2.0 else 80
            comment = f"Strong 24h down ({daily_move_pct:.2f}%) with bearish HMA."
        elif abs(daily_move_pct) < low_move_threshold:
            trend = "sideways"
            confidence = 50
            comment = f"Low volatility ({daily_move_pct:.2f}%), potential reversal or consolidation."
        elif current_price > ema_short > ema_long:
            trend = "bullish"
            confidence = 70
            comment = "Price above 50/200 EMA, mild bullish bias."
        elif current_price < ema_short < ema_long:
            trend = "bearish"
            confidence = 70
            comment = "Price below 50/200 EMA, mild bearish bias."

        return trend, confidence, comment, daily_move_pct

    @staticmethod
    def determine_dominant_trend(df_ohlcv, df_higher_tf, detected_candles, detected_charts, current_price, timeframe, market_type):
        if len(df_ohlcv) < 10:
            return "neutral", 50, 50, "Insufficient data for trend analysis."
        trend, trend_conf, trend_comment, daily_move_pct = AnalysisEngine.market_trend(df_ohlcv, df_higher_tf, current_price, timeframe, market_type)
        rsi = compute_rsi(df_ohlcv['close'], get_interval_mapping(timeframe)[2])
        macd, signal, hist = compute_macd(df_ohlcv['close'])
        adx = compute_adx(df_ohlcv, 14)
        supertrend = compute_supertrend(df_ohlcv, 7, 3.0)
        obv = compute_obv(df_ohlcv)
        obv_prev = compute_obv(df_ohlcv.iloc[:-1]) if len(df_ohlcv) > 1 else obv
        volume = df_ohlcv['volume'].iloc[-1]
        volume_ma = compute_ma(df_ohlcv['volume'], 20).iloc[-1]
        ema_50 = compute_ema(df_ohlcv['close'], 50).iloc[-1]
        ema_200 = compute_ema(df_ohlcv['close'], 200).iloc[-1]
        ha_df = compute_heikin_ashi(df_ohlcv)
        ha_bullish = (ha_df['close'].iloc[-5:] > ha_df['open'].iloc[-5:]).sum() >= 4
        trend_candles = (df_ohlcv['close'].iloc[-5:] > df_ohlcv['open'].iloc[-5:]).sum()
        rsi_divergence = detect_rsi_divergence(df_ohlcv, get_interval_mapping(timeframe)[2])

        bullish_score = 0
        bearish_score = 0
        comments = []

        if adx > 25 and current_price > supertrend:
            bullish_score += 1
            comments.append("Bullish: ADX > 25 and price above SuperTrend")
            if adx > 35:
                bullish_score += 0.5
                comments.append("Strong trend: ADX > 35")
        elif adx > 25 and current_price < supertrend:
            bearish_score += 1
            comments.append("Bearish: ADX > 25 and price below SuperTrend")
            if adx > 35:
                bearish_score += 0.5
                comments.append("Strong trend: ADX > 35")

        if rsi > 55 and macd > signal and hist > 0:
            bullish_score += 1
            comments.append("Bullish: RSI > 55 and MACD bullish")
            if rsi_divergence == "Bullish Divergence":
                bullish_score += 0.5
                comments.append("Bullish RSI Divergence")
        elif rsi < 45 and macd < signal and hist < 0:
            bearish_score += 1
            comments.append("Bearish: RSI < 45 and MACD bearish")
            if rsi_divergence == "Bearish Divergence":
                bearish_score += 0.5
                comments.append("Bearish RSI Divergence")

        if obv > obv_prev and volume > volume_ma:
            bullish_score += 1
            comments.append("Bullish: OBV rising and volume > 20MA")
            if volume > volume_ma * 2:
                bullish_score += 0.5
                comments.append("Volume spike: volume > 2x 20MA")
        elif obv < obv_prev and volume > volume_ma:
            bearish_score += 1
            comments.append("Bearish: OBV falling and volume > 20MA")
            if volume > volume_ma * 2:
                bearish_score += 0.5
                comments.append("Volume spike: volume > 2x 20MA")

        if current_price > ema_50 and current_price > ema_200:
            bullish_score += 1
            comments.append("Bullish: Price above 50/200 EMA")
            if ema_50 > ema_200:
                bullish_score += 0.5
                comments.append("Golden Cross: 50 EMA > 200 EMA")
        elif current_price < ema_50 and current_price < ema_200:
            bearish_score += 1
            comments.append("Bearish: Price below 50/200 EMA")
            if ema_50 < ema_200:
                bearish_score += 0.5
                comments.append("Death Cross: 50 EMA < 200 EMA")

        if ha_bullish and trend_candles >= 4:
            bullish_score += 1
            comments.append("Bullish: Heikin Ashi and 4/5 trend candles bullish")
            if trend_candles == 5:
                bullish_score += 0.5
                comments.append("Strong candle trend: 5/5 bullish")
        elif (ha_df['close'].iloc[-5:] < ha_df['open'].iloc[-5:]).sum() >= 4 and trend_candles <= 1:
            bearish_score += 1
            comments.append("Bearish: Heikin Ashi and 4/5 trend candles bearish")
            if trend_candles == 0:
                bearish_score += 0.5
                comments.append("Strong candle trend: 5/5 bearish")

        for p in detected_charts:
            if p in AnalysisEngine.high_prob_patterns:
                if p in AnalysisEngine.bullish_charts:
                    bullish_score += 1
                    comments.append(f"Bullish high-prob chart pattern: {p}")
                elif p in AnalysisEngine.bearish_charts:
                    bearish_score += 1
                    comments.append(f"Bearish high-prob chart pattern: {p}")
            else:
                if p in AnalysisEngine.bullish_charts:
                    bullish_score += 0.5
                    comments.append(f"Bullish chart pattern: {p}")
                elif p in AnalysisEngine.bearish_charts:
                    bearish_score += 0.5
                    comments.append(f"Bearish chart pattern: {p}")
        for p in detected_candles:
            if p in AnalysisEngine.bullish_candles:
                bullish_score += 0.5
                comments.append(f"Bullish candlestick: {p}")
            elif p in AnalysisEngine.bearish_candles:
                bearish_score += 0.5
                comments.append(f"Bearish candlestick: {p}")

        if bullish_score > bearish_score:
            for p in detected_charts + detected_candles:
                if p in AnalysisEngine.bearish_charts or p in AnalysisEngine.bearish_candles:
                    bullish_score -= 0.5
                    comments.append(f"Contradictory bearish pattern: {p}")
        elif bearish_score > bullish_score:
            for p in detected_charts + detected_candles:
                if p in AnalysisEngine.bullish_charts or p in AnalysisEngine.bullish_candles:
                    bearish_score -= 0.5
                    comments.append(f"Contradictory bullish pattern: {p}")

        total_score = max(bullish_score + bearish_score, 1)
        long_conf = min(int((bullish_score / total_score) * 100), 95)
        short_conf = min(int((bearish_score / total_score) * 100), 95)

        move_threshold = 1.5 if market_type == "futures" else 1.0
        low_move_threshold = 0.5 if market_type == "spot" else 1.5

        if bullish_score >= 4:
            dominant_bias = "long"
            long_conf = 95 if bullish_score >= 5 else 80 if bullish_score >= 4 else 70
            if abs(daily_move_pct) < low_move_threshold:
                long_conf = max(40, long_conf - 10)
                comments.append("Low market move, potential reversal or sideways")
            elif daily_move_pct > move_threshold:
                long_conf = min(95, long_conf + 10)
                comments.append(f"Strong market move ({daily_move_pct:.2f}%) boosts confidence")
        elif bearish_score >= 4:
            dominant_bias = "short"
            short_conf = 95 if bearish_score >= 5 else 80 if bearish_score >= 4 else 70
            if abs(daily_move_pct) < low_move_threshold:
                short_conf = max(40, short_conf - 10)
                comments.append("Low market move, potential reversal or sideways")
            elif daily_move_pct < -move_threshold:
                short_conf = min(95, short_conf + 10)
                comments.append(f"Strong market move ({daily_move_pct:.2f}%) boosts confidence")
        else:
            dominant_bias = "neutral"
            long_conf = min(long_conf, 60)
            short_conf = min(short_conf, 60)
            comments.append("Mixed signals; market is sideways")

        if long_conf < 40 and short_conf > long_conf:
            dominant_bias = "short"
            short_conf = max(60, short_conf)
            comments.append("Low bullish confidence, flipping to bearish for reversal")
        elif short_conf < 40 and long_conf > short_conf:
            dominant_bias = "long"
            long_conf = max(60, long_conf)
            comments.append("Low bearish confidence, flipping to bullish for reversal")

        pattern_comment = "; ".join(comments) + f" (Bullish Score: {bullish_score:.2f}, Bearish Score: {bearish_score:.2f})"
        return dominant_bias, long_conf, short_conf, pattern_comment

    @staticmethod
    def target_stoploss(dominant_bias, supports, resistances, entry_price, pattern_targets, pattern_sls, atr, current_price, timeframe, df_ohlcv):
        _, timeframe_multiplier, _, atr_period, _, _ = get_interval_mapping(timeframe)
        targets = []
        stoplosses = []
        min_separation = atr * 0.5
        atr = compute_atr(df_ohlcv, atr_period)

        if dominant_bias == "long":
            potential_tgts = [r for r in resistances if r > current_price + min_separation]
            potential_sls = [s for s in supports if s < current_price - min_separation]
            if potential_tgts:
                targets.append(min(potential_tgts))
                for t in potential_tgts[1:]:
                    if t - targets[-1] < atr * 2 * timeframe_multiplier and len(targets) < 2:
                        targets.append(t)
            else:
                targets.append(current_price + atr * 4.0 * timeframe_multiplier)
            if potential_sls:
                stoplosses.append(max(potential_sls))
            else:
                stoplosses.append(current_price - atr * 2.0 * timeframe_multiplier)
            for pat, ptgt in pattern_targets.items():
                if pat in AnalysisEngine.bullish_charts and ptgt > current_price + min_separation and len(targets) < 3:
                    targets.append(ptgt)
            for pat, psl in pattern_sls.items():
                if pat in AnalysisEngine.bullish_charts and psl < current_price - min_separation and len(stoplosses) < 2:
                    stoplosses.append(psl)
        elif dominant_bias == "short":
            potential_tgts = [s for s in supports if s < current_price - min_separation]
            potential_sls = [r for r in resistances if r > current_price + min_separation]
            if potential_tgts:
                targets.append(max(potential_tgts))
                for t in potential_tgts[1:]:
                    if targets[-1] - t < atr * 2 * timeframe_multiplier and len(targets) < 2:
                        targets.append(t)
            else:
                targets.append(current_price - atr * 4.0 * timeframe_multiplier)
            if potential_sls:
                stoplosses.append(min(potential_sls))
            else:
                stoplosses.append(current_price + atr * 2.0 * timeframe_multiplier)
            for pat, ptgt in pattern_targets.items():
                if pat in AnalysisEngine.bearish_charts and ptgt < current_price - min_separation and len(targets) < 3:
                    targets.append(ptgt)
            for pat, psl in pattern_sls.items():
                if pat in AnalysisEngine.bearish_charts and psl > current_price + min_separation and len(stoplosses) < 2:
                    stoplosses.append(psl)
        else:
            targets.append(current_price)
            stoplosses.append(current_price)

        for tgt in targets[:]:
            sl = stoplosses[0] if stoplosses else entry_price
            rr = abs(tgt - entry_price) / abs(entry_price - sl) if abs(entry_price - sl) > 0 else 0
            if rr < 1.5:
                targets.remove(tgt)
                if dominant_bias == "long":
                    targets.append(entry_price + atr * 4.0 * timeframe_multiplier)
                elif dominant_bias == "short":
                    targets.append(entry_price - atr * 4.0 * timeframe_multiplier)

        return sorted(targets), sorted(stoplosses, reverse=True)

    @staticmethod
    def calculate_user_sl(dominant_bias, entry_price, supports, resistances, atr, risk_pct, timeframe):
        _, _, _, atr_period, _, _ = get_interval_mapping(timeframe)
        tol = atr * 0.5
        if dominant_bias == "long":
            user_sl_raw = entry_price * (1 - risk_pct)
            close_levels = [s for s in supports if abs(s - user_sl_raw) < tol and s < entry_price]
            user_sl = max(close_levels, default=user_sl_raw)
            return max(user_sl, entry_price - atr * 2.0)
        elif dominant_bias == "short":
            user_sl_raw = entry_price * (1 + risk_pct)
            close_levels = [r for r in resistances if abs(r - user_sl_raw) < tol and r > entry_price]
            user_sl = min(close_levels, default=user_sl_raw)
            return min(user_sl, entry_price + atr * 2.0)
        else:
            return entry_price * (1 - 0.01)

def backtest(client, timeframe="15m", investment_per_trade=50.0, risk_pct=0.02, market_type="futures"):
    start_time = time.time()
    try:
        csv_file = 'ETHUSDT_Futures_15m_1Year.csv'
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        logger.debug(f"Loading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        logger.debug(f"CSV loaded. Rows: {len(df)}, Columns: {df.columns.tolist()}")
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing column {col} in CSV file")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df[required_columns[1:]].dropna()

        if len(df) < 20:
            logger.warning("Insufficient data in CSV file. Need at least 20 rows for indicators.")
            return

        end_date = df.index.max()
        start_date = end_date - timedelta(days=30)
        df = df.loc[start_date:end_date]
        logger.debug(f"Filtered data from {start_date} to {end_date}, Rows: {len(df)}")

        higher_tf = "4h" if timeframe in ["5m", "15m"] else "1d"
        df_higher_tf = df.resample(higher_tf).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        logger.debug(f"Higher timeframe ({higher_tf}) data rows: {len(df_higher_tf)}")

        account_size = 10000
        fee_rate = 0.0004
        slippage = 0.0005
        open_trades = []
        trades = []
        trade_interval = 4  # Every hour (4 * 15m)

        for i in range(len(df)):
            if time.time() - start_time > 600:
                logger.error("Backtest timeout after 10 minutes")
                return
            if i < 20:
                continue
            if i % 100 == 0:
                logger.debug(f"Processing row {i}/{len(df)} ({i/len(df)*100:.1f}%)")
            current_time = df.index[i]

            current_df = df.iloc[:i+1]
            current_price = float(current_df['close'].iloc[-1])
            last_volume = current_df['volume'].iloc[-1]
            avg_volume = current_df['volume'].rolling(20).mean().iloc[-1] if len(current_df) >= 20 else last_volume

            detected_candles = detect_candlestick_patterns(current_df)
            detected_charts, pattern_targets, pattern_sls = detect_chart_patterns(current_df, timeframe)
            dominant_bias, long_conf, short_conf, pattern_comment = AnalysisEngine.determine_dominant_trend(
                current_df, df_higher_tf, detected_candles, detected_charts, current_price, timeframe, market_type
            )
            supports, resistances = detect_support_resistance(current_df, current_price, window=3, lookback=100)
            atr = compute_atr(current_df, get_interval_mapping(timeframe)[3])
            adx = compute_adx(current_df, 14)

            higher_trend, _, _, _ = AnalysisEngine.market_trend(df_higher_tf, pd.DataFrame(), current_price, "4h", market_type)
            logger.debug(f"Row {i}: Bias={dominant_bias}, LongConf={long_conf}, ShortConf={short_conf}, ADX={adx:.2f}, VolumeRatio={last_volume/avg_volume if avg_volume else 0:.2f}, HigherTrend={higher_trend}, Candles={detected_candles}, Charts={detected_charts}")

            if (i % trade_interval == 0 and dominant_bias != "neutral" and adx > 20 and last_volume > avg_volume * 1.2 and
                ((long_conf >= 60 and dominant_bias == "long" and higher_trend in ["bullish", "sideways"]) or 
                 (short_conf >= 60 and dominant_bias == "short" and higher_trend in ["bearish", "sideways"]))):
                user_sl = AnalysisEngine.calculate_user_sl(dominant_bias, current_price, supports, resistances, atr, risk_pct, timeframe)
                targets, stoplosses = AnalysisEngine.target_stoploss(
                    dominant_bias, supports, resistances, current_price, pattern_targets, pattern_sls, atr, current_price, timeframe, current_df
                )
                target = targets[0] if targets else current_price
                sl = stoplosses[0] if stoplosses else user_sl
                rr_ratios = [abs(target - current_price) / abs(current_price - sl)] if abs(current_price - sl) > 0 else [0]

                if dominant_bias == "long" and (sl >= current_price or target <= current_price or rr_ratios[0] < 1.5):
                    logger.warning(f"Invalid trade setup: SL {sl:.4f} >= Entry {current_price:.4f} or Target {target:.4f} <= Entry or RR {rr_ratios[0]:.2f} < 1.5")
                    continue
                if dominant_bias == "short" and (sl <= current_price or target >= current_price or rr_ratios[0] < 1.5):
                    logger.warning(f"Invalid trade setup: SL {sl:.4f} <= Entry {current_price:.4f} or Target {target:.4f} >= Entry or RR {rr_ratios[0]:.2f} < 1.5")
                    continue

                volume = investment_per_trade / current_price if current_price != 0 else 1.0
                entry_price = current_price * (1 + slippage if dominant_bias == "long" else 1 - slippage)
                risk_per_trade = abs(entry_price - sl) * volume
                if risk_per_trade > investment_per_trade:
                    logger.warning(f"Risk {risk_per_trade:.2f} exceeds investment {investment_per_trade:.2f}, skipping trade")
                    continue

                trailing_distance = atr * 1.5

                open_trades.append({
                    'entry_time': current_time,
                    'bias': dominant_bias,
                    'entry_price': entry_price,
                    'target': target,
                    'sl': sl,
                    'trailing_distance': trailing_distance,
                    'rr': rr_ratios[0] if rr_ratios else 0,
                    'volume': volume,
                    'confidence': long_conf if dominant_bias == "long" else short_conf,
                    'pattern': detected_charts + detected_candles,
                    'exit_time': None,
                    'result': None,
                    'pl': 0.0
                })
                logger.info(f"Trade Entry: Time={current_time}, Bias={dominant_bias}, EntryPrice={entry_price:.4f}, Target={target:.4f}, SL={sl:.4f}, RR={rr_ratios[0]:.2f}, Volume={volume:.6f}, Confidence={long_conf if dominant_bias == 'long' else short_conf}")

            for trade in open_trades[:]:
                if trade['exit_time'] is not None:
                    continue
                high = float(current_df['high'].iloc[-1])
                low = float(current_df['low'].iloc[-1])

                if trade['bias'] == "long" and high > trade['entry_price'] + trade['trailing_distance']:
                    new_sl = high - trade['trailing_distance']
                    trade['sl'] = max(trade['sl'], new_sl)
                elif trade['bias'] == "short" and low < trade['entry_price'] - trade['trailing_distance']:
                    new_sl = low + trade['trailing_distance']
                    trade['sl'] = min(trade['sl'], new_sl)

                if trade['bias'] == "long":
                    if low <= trade['sl']:
                        trade['result'] = 'LOSS'
                        trade['pl'] = (trade['sl'] - trade['entry_price']) * trade['volume'] - (fee_rate * trade['entry_price'] * trade['volume'] * 2)
                    elif high >= trade['target']:
                        trade['result'] = 'WIN'
                        trade['pl'] = (trade['target'] - trade['entry_price']) * trade['volume'] - (fee_rate * trade['entry_price'] * trade['volume'] * 2)
                else:
                    if high >= trade['sl']:
                        trade['result'] = 'LOSS'
                        trade['pl'] = (trade['entry_price'] - trade['sl']) * trade['volume'] - (fee_rate * trade['entry_price'] * trade['volume'] * 2)
                    elif low <= trade['target']:
                        trade['result'] = 'WIN'
                        trade['pl'] = (trade['entry_price'] - trade['target']) * trade['volume'] - (fee_rate * trade['entry_price'] * trade['volume'] * 2)
                if trade['result'] is not None:
                    trade['exit_time'] = current_time
                    trades.append(trade)
                    open_trades.remove(trade)
                    logger.info(f"Trade Outcome: Time={current_time}, Outcome={trade['result']}, Bias={trade['bias']}, P&L={trade['pl']:.2f}, Risk={abs(trade['entry_price'] - trade['sl']) * trade['volume']:.2f}")

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df.to_csv('trades.csv', index=False)
            logger.info("Trades saved to trades.csv")
        else:
            logger.warning("No trades executed during backtest.")

        total_trades = len(trades_df)
        if total_trades == 0:
            logger.info("No trades to analyze.")
            return

        wins = (trades_df['result'] == 'WIN').sum()
        losses = (trades_df['result'] == 'LOSS').sum()
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        avg_profit = trades_df[trades_df['result'] == 'WIN']['pl'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['result'] == 'LOSS']['pl'].mean() if losses > 0 else 0
        expectancy = (win_rate / 100 * avg_profit) - ((1 - win_rate / 100) * abs(avg_loss)) if total_trades > 0 else 0
        account_balance = account_size + trades_df['pl'].cumsum()
        peak = account_balance.max() if not account_balance.empty else account_size
        drawdown = (account_balance - peak).min() if not account_balance.empty else 0
        max_drawdown = (drawdown / peak * 100) if peak != 0 else 0
        bias_accuracy = ((trades_df['bias'] == 'long') & (trades_df['pl'] > 0) | (trades_df['bias'] == 'short') & (trades_df['pl'] > 0)).mean() * 100 if total_trades > 0 else 0

        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Average RR: {trades_df['rr'].mean():.2f}" if not trades_df.empty else "Average RR: N/A")
        logger.info(f"Expectancy: {expectancy:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"Bias Accuracy: {bias_accuracy:.2f}%")

    except pd.errors.EmptyDataError:
        logger.error(f"CSV file '{csv_file}' is empty or improperly formatted.")
        raise
    except pd.errors.ParserError:
        logger.error(f"Failed to parse CSV file '{csv_file}'. Ensure it has the correct format and columns: {required_columns}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise
    finally:
        logger.debug(f"Backtest completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    backtest(client=None, timeframe="15m", investment_per_trade=50.0, risk_pct=0.02, market_type="futures")