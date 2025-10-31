import pandas as pd
import numpy as np
from os import listdir
import time

# Paste your detect functions (full from main.py – all 30+ candlestick/chart)
def detect_candlestick_patterns(df, vol_mult=1.5):
    patterns = []
    if len(df) < 1:
        return patterns, 0.0
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
    body_to_range = body / total_range if total_range > 0 else 0

    # Stricter volume check
    if last_volume > avg_volume * vol_mult:
        if body_to_range < 0.05:
            patterns.append("Doji")
        if lower_shadow > 2 * body and upper_shadow < 0.1 * body and last_close > last_open:
            patterns.append("Hammer")
        if upper_shadow > 2 * body and lower_shadow < 0.1 * body and last_close > last_open:
            patterns.append("Inverted Hammer")
        if upper_shadow > 2 * body and lower_shadow < 0.1 * body and last_close < last_open:
            patterns.append("Shooting Star")
        if lower_shadow > 2 * body and upper_shadow < 0.1 * body and last_close < last_open:
            patterns.append("Hanging Man")
        if body_to_range < 0.3 and upper_shadow > body and lower_shadow > body:
            patterns.append("Spinning Top")
        if upper_shadow < 0.05 * body and lower_shadow < 0.05 * body and last_close > last_open:
            patterns.append("Bullish Marubozu")
        if upper_shadow < 0.05 * body and lower_shadow < 0.05 * body and last_close < last_open:
            patterns.append("Bearish Marubozu")
        if lower_shadow < 0.05 * body and last_close > last_open and last_open <= last_low + 0.001 * total_range:
            patterns.append("Bullish Belt Hold")
        if upper_shadow < 0.05 * body and last_close < last_open and last_open >= last_high - 0.001 * total_range:
            patterns.append("Bearish Belt Hold")

    if len(df) >= 2 and last_volume > avg_volume * vol_mult:
        prev_open = open_.iloc[-2]
        prev_close = close.iloc[-2]
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        prev_volume = volume.iloc[-2]
        if prev_close < prev_open and last_close > prev_open and last_open < prev_close and last_close > last_open and last_volume > prev_volume * 1.2:
            patterns.append("Bullish Engulfing")
        if prev_close > prev_open and last_close < prev_open and last_open > prev_close and last_close < last_open and last_volume > prev_volume * 1.2:
            patterns.append("Bearish Engulfing")
        mid_prev = (prev_open + prev_close) / 2
        if prev_close < prev_open and last_open < prev_close and last_close > mid_prev and last_close < prev_open and last_volume > prev_volume:
            patterns.append("Piercing Line")
        if prev_close > prev_open and last_open > prev_close and last_close < mid_prev and last_close > prev_open and last_volume > prev_volume:
            patterns.append("Dark Cloud Cover")
        if prev_close < prev_open and last_open > prev_close and last_close < prev_open and last_close > last_open and last_volume > prev_volume:
            patterns.append("Bullish Harami")
        if prev_close > prev_open and last_open < prev_close and last_close > prev_open and last_close < last_open and last_volume > prev_volume:
            patterns.append("Bearish Harami")
        if abs(last_low - prev_low) < 0.001 * last_low and prev_close < prev_open and last_close > last_open:
            patterns.append("Tweezer Bottom")
        if abs(last_high - prev_high) < 0.001 * last_high and prev_close > prev_open and last_close < last_open:
            patterns.append("Tweezer Top")

    if len(df) >= 3 and last_volume > avg_volume * vol_mult:
        p2_open = open_.iloc[-3]
        p2_close = close.iloc[-3]
        p1_open = open_.iloc[-2]
        p1_close = close.iloc[-2]
        p1_volume = volume.iloc[-2]
        if p2_close < p2_open and abs(p1_close - p1_open) < 0.3 * abs(p2_close - p2_open) and last_close > last_open and last_close > (p2_open + p2_close) / 2 and last_volume > p1_volume * 1.2:
            patterns.append("Morning Star")
        if p2_close > p2_open and abs(p1_close - p1_open) < 0.3 * abs(p2_close - p2_open) and last_close < last_open and last_close < (p2_open + p2_close) / 2 and last_volume > p1_volume * 1.2:
            patterns.append("Evening Star")
        if all(close.iloc[-i] > open_.iloc[-i] for i in range(1, 4)) and close.iloc[-2] > close.iloc[-3] and close.iloc[-1] > close.iloc[-2]:
            patterns.append("Three White Soldiers")
        if all(close.iloc[-i] < open_.iloc[-i] for i in range(1, 4)) and close.iloc[-2] < close.iloc[-3] and close.iloc[-1] < close.iloc[-2]:
            patterns.append("Three Black Crows")

    logger.info(f"Detected candlestick patterns: {patterns}, Body-to-range: {body_to_range:.2f}")
    return patterns, body_to_range

def detect_chart_patterns(df, tol=0.01):
    if df.empty:
        logger.warning("Empty DataFrame in detect_chart_patterns, returning empty patterns")
        return [], {}, {}

    df_close = df['close']
    patterns = []
    pattern_targets = {}
    pattern_sls = {}
    peak_indices = get_peak_indices(df_close)[-15:]  # More for complex patterns
    trough_indices = get_trough_indices(df_close)[-15:]
    current = df_close.iloc[-1]

    # Volume filter for confirmation (boost accuracy)
    avg_vol = df['volume'].rolling(20).mean().iloc[-1]
    current_vol = df['volume'].iloc[-1]
    vol_confirm = current_vol > avg_vol * 1.2  # High vol for validity

    # Double Top/Bottom (fixed logic – remove redundant, add vol)
    if len(peak_indices) >= 2 and vol_confirm:
        p1_idx = peak_indices[-2]
        p2_idx = peak_indices[-1]
        p1 = df_close[p1_idx]
        p2 = df_close[p2_idx]
        if abs(p1 - p2) / ((p1 + p2) / 2) < tol:
            slice_data = df_close[p1_idx:p2_idx]
            if not slice_data.empty:
                patterns.append("Double Top")
                neckline = slice_data.min()
                height = p1 - neckline
                pattern_targets["Double Top"] = neckline - height
                pattern_sls["Double Top"] = max(p1, p2) + height * 0.1
                # Fixed: Confirm break neckline (bearish for Double Top)
                if current < neckline:  # Break down = valid
                    pass
                else:
                    patterns.remove("Double Top")
                    del pattern_targets["Double Top"]
                    del pattern_sls["Double Top"]

    # Triple Top/Bottom
    if len(peak_indices) >= 3 and vol_confirm:
        p1 = df_close[peak_indices[-3]]
        p2 = df_close[peak_indices[-2]]
        p3 = df_close[peak_indices[-1]]
        if abs(p1 - p2) < tol * p1 and abs(p2 - p3) < tol * p2:
            patterns.append("Triple Top")
            neckline = df_close[peak_indices[-3]:peak_indices[-1]].min()
            height = p1 - neckline
            pattern_targets["Triple Top"] = neckline - height
            pattern_sls["Triple Top"] = max(p1, p2, p3) + height * 0.1

    # Head and Shoulders / Inverse
    if len(peak_indices) >= 3 and vol_confirm:
        p1 = df_close[peak_indices[-3]]
        p2 = df_close[peak_indices[-2]]
        p3 = df_close[peak_indices[-1]]
        if p2 > p1 and p2 > p3 and abs(p1 - p3) < tol * p2:
            patterns.append("Head and Shoulders")
            neckline = df_close[peak_indices[-3]:peak_indices[-1]].min()
            height = p2 - neckline
            pattern_targets["Head and Shoulders"] = neckline - height
            pattern_sls["Head and Shoulders"] = p2 + height * 0.1

    # Flag/Pennant (continuation)
    if len(peak_indices) >= 3 and len(trough_indices) >= 3 and vol_confirm:
        highs = df_close[peak_indices[-3:]]
        lows = df_close[trough_indices[-3:]]
        if highs.std() < highs.mean() * 0.02 and lows.std() < lows.mean() * 0.02:
            patterns.append("Flag" if len(highs) < 5 else "Pennant")
            height = highs.mean() - lows.mean()
            pattern_targets["Flag"] = current + height * (1 if highs.iloc[-1] > highs.iloc[-2] else -1)
            pattern_sls["Flag"] = current - height * 0.5

    # Cup & Handle (bullish)
    if len(peak_indices) >= 4 and len(trough_indices) >= 4 and vol_confirm:
        troughs = df_close[trough_indices[-4:]]
        peaks = df_close[peak_indices[-4:]]
        if np.isclose(troughs.iloc[0], troughs.iloc[2], rtol=0.02) and np.isclose(peaks.iloc[0], peaks.iloc[3], rtol=0.02) and troughs.iloc[3] > troughs.iloc[1]:
            patterns.append("Cup & Handle")
            pattern_targets["Cup & Handle"] = peaks.iloc[3] + (peaks.iloc[3] - troughs.iloc[1])
            pattern_sls["Cup & Handle"] = troughs.iloc[1] - (peaks.iloc[3] - troughs.iloc[1]) * 0.5

    # Ascending/Descending Channel
    if len(peak_indices) >= 4 and len(trough_indices) >= 4 and vol_confirm:
        peak_slope = (df_close[peak_indices[-1]] - df_close[peak_indices[-4]]) / 3
        trough_slope = (df_close[trough_indices[-1]] - df_close[trough_indices[-4]]) / 3
        if abs(peak_slope - trough_slope) < 0.01:
            patterns.append("Ascending Channel" if peak_slope > 0 else "Descending Channel")
            height = df_close[peak_indices[-1]] - df_close[trough_indices[-1]]
            pattern_targets["Ascending Channel"] = current + height * 1.5
            pattern_sls["Ascending Channel"] = current - height * 0.5

    logger.info(f"Detected chart patterns: {patterns}")
    return patterns, pattern_targets, pattern_sls
# Data folder
data_folder = "data/"
csv_files = [f for f in listdir(data_folder) if f.endswith('_enriched.csv')]  # All your enriched files

for csv_file in csv_files:  # Loop all coins
    try:
        # Load
        df = pd.read_csv(f"{data_folder}{csv_file}")
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.sort_values('open_time').reset_index(drop=True)
        print(f"Prepping {csv_file} – {len(df)} rows")

        # Labels for patterns (scan all, mark 1/0 by outcome)
        df['pattern_label'] = 0
        for i in range(len(df)-3):
            patterns, _ = detect_candlestick_patterns(df.iloc[i:i+3])
            if patterns:  # Any pattern found
                future_return = (df['close'].iloc[i+3] - df['close'].iloc[i]) / df['close'].iloc[i]
                if future_return > 0.01:  # Up 1% = correct
                    df.loc[df.index[i], 'pattern_label'] = 1

        # Labels for bias (mark 1/0)
        df['bias_label'] = 0
        for i in range(len(df)-1):
            if df['ema9'].iloc[i] > df['ma50'].iloc[i] and df['rsi'].iloc[i] > 50:
                future = (df['close'].iloc[i+1] - df['close'].iloc[i]) / df['close'].iloc[i]
                if future > 0.005:  # Up 0.5% = correct
                    df.loc[df.index[i], 'bias_label'] = 1

        # Save prepped (different files)
        patterns_df = df[['body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio', 'vol_ratio', 'rsi', 'ema9', 'ma50', 'atr', 'pattern_label']].dropna()
        bias_df = df[['ema9', 'ma50', 'rsi', 'atr', 'adx', 'bb_width', 'vol_ratio', 'bias_label']].dropna()
        patterns_df.to_csv(f"data/{csv_file.replace('.csv', '_patterns_data.csv')}", index=False)
        bias_df.to_csv(f"data/{csv_file.replace('.csv', '_bias_data.csv')}", index=False)
        print(f"Prepped {csv_file}: {len(patterns_df)} patterns rows, {len(bias_df)} bias rows. Files saved!")

    except Exception as e:
        print(f"Error prepping {csv_file}: {e}")

print("All prepped! Check new *_patterns_data.csv and *_bias_data.csv files in data/.")