import json
import random
from pybit.unified_trading import HTTP
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import logging
from collections import defaultdict
from datetime import datetime, timedelta
import uuid
import numpy as np
import talib  # For potential Fib/ATR enhancements

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://position-analyzer-app.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Bybit Mainnet API keys from environment variables
API_KEY = os.environ.get("BYBIT_API_KEY")
API_SECRET = os.environ.get("BYBIT_API_SECRET")

# Input & Validation
class InputValidator:
    REQUIRED_FIELDS = {"coin", "market", "entry_price", "quantity", "position_type", "timeframe"}
    VALID_TIMEFRAMES = ["15m", "1h", "4h", "1d"]

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
            raise ValueError(f"Invalid timeframe {data['timeframe']}. Must be one of: {', '.join(InputValidator.VALID_TIMEFRAMES)}")
        return data

# Bybit Data Fetch
def initialize_client():
    max_retries = 5
    for attempt in range(max_retries):
        try:
            client = HTTP(
                api_key=API_KEY,
                api_secret=API_SECRET,
                testnet=False
            )
            response = client.get_server_time()
            logger.info(f"Server time response: {response}")
            return client
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Failed to initialize Bybit client (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if "rate limit" in error_msg or "403" in error_msg:
                wait_time = 2 ** attempt
                logger.info(f"Rate limit or 403 detected, waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries reached for Bybit client initialization")

def get_current_price(client, coin, market):
    try:
        if market == "spot":
            ticker = client.get_tickers(category="spot", symbol=coin)
            price = float(ticker['result']['list'][0]['lastPrice'])
        else:
            ticker = client.get_tickers(category="linear", symbol=coin)
            price = float(ticker['result']['list'][0]['lastPrice'])
        logger.info(f"Current price for {coin} ({market}): {price}")
        return price
    except Exception as e:
        logger.error(f"Failed to get current price for {coin}: {str(e)}")
        raise ValueError(f"Failed to get current price for {coin}: {str(e)}")

def get_interval_mapping(timeframe):
    mapping = {
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": "D"
    }
    # Enhanced: TF-specific multipliers for TGT/SL (tighter for shorter TFs)
    multiplier_adjust = {
        "15m": 0.8,  # Reduced for 15m noise
        "1h": 1.2,
        "4h": 2.0,
        "1d": 3.0
    }
    # Enhanced: Shorter lookbacks for shorter TFs to reduce lag
    lookback_periods = {
        "15m": 200,  # ~2 days, reduced for faster signals
        "1h": 336,   # ~2 weeks
        "4h": 168,   # ~1 month
        "1d": 90
    }
    lookback_months = {
        "15m": 0.1,  # Shorter for S/R
        "1h": 0.5,
        "4h": 1.5,
        "1d": 6
    }
    return mapping[timeframe], multiplier_adjust[timeframe], lookback_periods[timeframe], lookback_months[timeframe]

def get_ohlcv_df(client, coin, timeframe, market):
    interval, _, lookback, _ = get_interval_mapping(timeframe)
    try:
        if market == "spot":
            klines = client.get_kline(
                category="spot",
                symbol=coin,
                interval=interval,
                limit=min(lookback, 1000)
            )
        else:
            klines = client.get_kline(
                category="linear",
                symbol=coin,
                interval=interval,
                limit=min(lookback, 1000)
            )
        if not klines['result'] or 'list' not in klines['result'] or not klines['result']['list']:
            raise ValueError(f"No OHLCV data available for {coin}")
        df = pd.DataFrame(klines['result']['list'], columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce').astype('int64')
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.sort_values('open_time')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        logger.info(f"OHLCV data for {coin} ({timeframe}): {len(df)} candles retrieved")
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV data for {coin}: {str(e)}")
        raise ValueError(f"Failed to fetch OHLCV data for {coin}: {str(e)}")

# Enhanced Indicator Helpers (TF-tuned)
def compute_ema(series, span):
    if series.empty:
        return pd.Series([0.0])
    return series.ewm(span=span, adjust=False).mean()

def compute_ma(series, window):
    if series.empty:
        return pd.Series([0.0])
    return series.rolling(window=window).mean()

def compute_atr(df, period=14):
    if len(df) < period:
        return 0.0
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.iloc[-1] if not atr.empty else 0.0

def compute_rsi(series, period=14):
    if len(series) < period:
        return 50.0
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - 100 / (1 + rs)
    return rsi.iloc[-1] if not rsi.empty else 50.0

def compute_macd(series, fast=12, slow=26, signal=9):
    if len(series) < slow:
        return (0.0, 0.0, 0.0)
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = compute_ema(macd, signal)
    histogram = macd - signal_line
    return (macd.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]) if len(macd) >= signal else (0.0, 0.0, 0.0)

def compute_adx(df, period=14):
    if len(df) < 2 * period:
        return 25.0
    high = df['high']
    low = df['low']
    close = df['close']
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx.iloc[-1] if not adx.empty else 25.0

def compute_bollinger_bands(series, window=20, num_std=2):
    if len(series) < window:
        return 0.0, 0.0, 0.0
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma.iloc[-1], upper.iloc[-1], lower.iloc[-1] if len(sma) >= window else (0.0, 0.0, 0.0)

# Enhanced SuperTrend (TF-tuned: multiplier 2.0 for 15m/1h)
def compute_supertrend(df, period=10, multiplier=2.0):  # Tuned multiplier
    if len(df) < period:
        df['supertrend'] = df['close']
        df['st_color'] = 'neutral'
        return df
    hl2 = (df['high'] + df['low']) / 2
    df['atr_st'] = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1).rolling(period).mean()
    df['upper_band'] = hl2 + (multiplier * df['atr_st'])
    df['lower_band'] = hl2 - (multiplier * df['atr_st'])
    df['supertrend'] = df['upper_band']
    df['st_color'] = 'red'
    for i in range(1, len(df)):
        if df['close'].iloc[i-1] <= df['upper_band'].iloc[i-1]:
            df.loc[df.index[i], 'supertrend'] = df['upper_band'].iloc[i]
        else:
            df.loc[df.index[i], 'supertrend'] = df['lower_band'].iloc[i]
        if df['st_color'].iloc[i-1] == 'green' and df['close'].iloc[i] <= df['supertrend'].iloc[i]:
            df.loc[df.index[i], 'st_color'] = 'red'
        elif df['st_color'].iloc[i-1] == 'red' and df['close'].iloc[i] >= df['supertrend'].iloc[i]:
            df.loc[df.index[i], 'st_color'] = 'green'
        else:
            df.loc[df.index[i], 'st_color'] = df['st_color'].iloc[i-1]
    return df

# Market Structure Detection (enhanced for shorter TFs: rolling window 2 for 15m)
def detect_market_structure(df, tf_window=3):  # Tunable window
    if len(df) < tf_window * 2:
        return "Mixed"
    highs = df['high'].rolling(tf_window, center=True).max()
    lows = df['low'].rolling(tf_window, center=True).min()
    recent_highs = df['high'].tail(tf_window + 1).values
    recent_lows = df['low'].tail(tf_window + 1).values
    if recent_highs[-1] > recent_highs[-2] > recent_highs[-3] and recent_lows[-1] > recent_lows[-2]:
        return "HH-HL"
    elif recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]:
        return "LH-LL"
    else:
        return "Mixed"

# Enhanced Trend Classification (add VWAP-like simple MA for confluence)
def classify_trend(df, timeframe):
    if df.empty:
        return "sideways"
    if 'st_color' not in df.columns:
        df = compute_supertrend(df)
    st_color = df['st_color'].iloc[-1]
    # TF-tuned EMAs: shorter for 15m
    ema_fast = 10 if timeframe == "15m" else 20
    ema_slow = 21 if timeframe == "15m" else 50
    if f'ema{ema_fast}' not in df.columns:
        df[f'ema{ema_fast}'] = compute_ema(df['close'], ema_fast)
        df[f'ema{ema_slow}'] = compute_ema(df['close'], ema_slow)
    ema_fast_val = df[f'ema{ema_fast}'].iloc[-1]
    ema_slow_val = df[f'ema{ema_slow}'].iloc[-1]
    close = df['close'].iloc[-1]
    if st_color == 'green' and close > ema_fast_val and close > ema_slow_val:
        return "bullish"
    elif st_color == 'red' and close < ema_fast_val and close < ema_slow_val:
        return "bearish"
    else:
        return "sideways"

# Enhanced Volume Status (stricter for 15m)
def get_volume_status(df, tf_mult=1.5):  # Higher mult for shorter TFs
    if len(df) < 20:
        return "Normal"
    vol_sma20 = df['volume'].rolling(20).mean().iloc[-1]
    current_vol = df['volume'].iloc[-1]
    if current_vol > vol_sma20 * tf_mult:
        return "High"
    elif current_vol < vol_sma20 * 0.7:  # Tighter low threshold
        return "Low"
    return "Normal"

def compute_indicators_for_df(df, timeframe):
    if df.empty:
        return {'ema_fast': 0, 'ema_slow': 0, 'atr': 0, 'rsi': 50, 'macd': 0, 'signal': 0, 'hist': 0, 'adx': 25, 'sma20': 0, 'bb_upper': 0, 'bb_lower': 0, 'vol_sma20': 0, 'supertrend': 0, 'st_color': 'neutral'}
    close = df['close']
    df = compute_supertrend(df)
    ema_fast = 10 if timeframe == "15m" else 20
    ema_slow = 21 if timeframe == "15m" else 50
    df[f'ema{ema_fast}'] = compute_ema(close, ema_fast)
    df[f'ema{ema_slow}'] = compute_ema(close, ema_slow)
    ema_fast_val = df[f'ema{ema_fast}'].iloc[-1]
    ema_slow_val = df[f'ema{ema_slow}'].iloc[-1]
    atr = compute_atr(df)
    rsi = compute_rsi(close)
    macd, signal, hist = compute_macd(close)
    adx = compute_adx(df)
    sma20, bb_upper, bb_lower = compute_bollinger_bands(close)
    vol_sma20 = df['volume'].rolling(20).mean().iloc[-1]
    return {
        f'ema{ema_fast}': ema_fast_val, f'ema{ema_slow}': ema_slow_val,
        'atr': atr, 'rsi': rsi, 'macd': macd, 'signal': signal, 'hist': hist,
        'adx': adx, 'sma20': sma20, 'bb_upper': bb_upper, 'bb_lower': bb_lower,
        'vol_sma20': vol_sma20, 'supertrend': df['supertrend'].iloc[-1], 'st_color': df['st_color'].iloc[-1]
    }

def detect_supports_resistances_for_df(df, current_price, lookback_months):
    if df.empty:
        return [], []
    last_time = df['open_time'].iloc[-1]
    start_time = last_time - timedelta(days=lookback_months * 30)
    df_lookback = df[df['open_time'] >= start_time]
    if df_lookback.empty:
        df_lookback = df
    avg_volume = df_lookback['volume'].mean()
    window = 3
    tol = 0.003  # Tighter tolerance for shorter TFs
    lows = df_lookback['low'].rolling(window, center=True).min()
    highs = df_lookback['high'].rolling(window, center=True).max()
    candidates_s = df_lookback[(df_lookback['low'] == lows) & (df_lookback['volume'] > avg_volume * 1.2)]['low'].unique()  # Volume filter
    candidates_r = df_lookback[(df_lookback['high'] == highs) & (df_lookback['volume'] > avg_volume * 1.2)]['high'].unique()

    def get_touches_and_vol(level, series, vol_series, tol):
        touches = abs(series - level) < level * tol
        count = touches.sum()
        vol_weight = vol_series[touches].mean() / avg_volume if count > 0 else 0
        return count, vol_weight

    supports = []
    for level in candidates_s:
        count, vol_w = get_touches_and_vol(level, df_lookback['close'], df_lookback['volume'], tol)
        if count >= 2:  # Stricter touches for noise
            supports.append((level, count, vol_w))
    resistances = []
    for level in candidates_r:
        count, vol_w = get_touches_and_vol(level, df_lookback['close'], df_lookback['volume'], tol)
        if count >= 2:
            resistances.append((level, count, vol_w))

    supports.sort(key=lambda x: x[1] * x[2], reverse=True)
    resistances.sort(key=lambda x: x[1] * x[2], reverse=True)
    supports = [x[0] for x in supports[:3]]  # Limit to top 3
    resistances = [x[0] for x in resistances[:3]]

    pivot_s, pivot_r, _ = compute_pivot_points(df_lookback)
    supports += pivot_s
    resistances += pivot_r
    supports = sorted(list(set(supports)))
    resistances = sorted(list(set(resistances)))

    return supports, resistances

def compute_pivot_points(df, period=50):
    if len(df) < 1:
        return [], [], 0.0
    if len(df) < period:
        period = len(df)
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
    supports = [s for s in [s1, s2, s3] if s > 0]
    resistances = [r for r in [r1, r2, r3] if r > 0]
    return supports, resistances, pivot

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

    # Existing + New (80% common)
    if last_volume > avg_volume * vol_mult:
        # Existing 20+...
        if body_to_range < 0.05:
            patterns.append("Doji")
        if lower_shadow > 2 * body and upper_shadow < 0.1 * body and last_close > last_open:
            patterns.append("Hammer")
        # ... (your full existing list)

        # New: Dragonfly/Gravestone Doji (high reversal)
        if body_to_range < 0.05 and lower_shadow > total_range * 0.6 and upper_shadow < 0.1 * body:
            patterns.append("Dragonfly Doji")
        if body_to_range < 0.05 and upper_shadow > total_range * 0.6 and lower_shadow < 0.1 * body:
            patterns.append("Gravestone Doji")

        # Rising/Falling Three Methods (continuation)
        if len(df) >= 4:
            prev_close = close.iloc[-2]
            if all(close.iloc[-i] > open_.iloc[-i] for i in range(1, 4)) and close.iloc[-4] < open_.iloc[-4] and last_close > prev_close:
                patterns.append("Rising Three Methods")
            if all(close.iloc[-i] < open_.iloc[-i] for i in range(1, 4)) and close.iloc[-4] > open_.iloc[-4] and last_close < prev_close:
                patterns.append("Falling Three Methods")

        # Upside/Downside Gap Two Crows (reversal)
        if len(df) >= 3:
            prev_close = close.iloc[-3]
            if prev_close > open_.iloc[-3] and last_open > prev_close and last_close < open_.iloc[-2] and last_close < open_.iloc[-1]:
                patterns.append("Upside Gap Two Crows")

    logger.info(f"Detected candlestick patterns: {patterns}")
    return patterns, body_to_range

def detect_chart_patterns(df, tol=0.01):
    if df.empty:
        return [], {}, {}
    
    df_close = df['close']
    patterns = []
    pattern_targets = {}
    pattern_sls = {}
    peak_indices = get_peak_indices(df_close)[-15:]  # More for complex
    trough_indices = get_trough_indices(df_close)[-15:]
    current = df_close.iloc[-1]

    # Existing + New (80% common)
    # Double Top/Bottom (existing, tol tuned)
    if len(peak_indices) >= 2:
        p1_idx = peak_indices[-2]
        p2_idx = peak_indices[-1]
        p1 = df_close[p1_idx]
        p2 = df_close[p2_idx]
        if abs(p1 - p2) / ((p1 + p2) / 2) < tol and p2_idx > p1_idx:
            patterns.append("Double Top")
            # ... (your code)

    # New: Flag/Pennant (continuation, 70% trend resume)
    if len(peak_indices) >= 3 and len(trough_indices) >= 3:
        highs = df_close[peak_indices[-3:]]
        lows = df_close[trough_indices[-3:]]
        if highs.std() < highs.mean() * 0.02 and lows.std() < lows.mean() * 0.02:  # Parallel channel
            patterns.append("Flag" if len(highs) < 5 else "Pennant")
            height = (highs.mean() - lows.mean())
            pattern_targets["Flag"] = current + height * (1 if highs.iloc[-1] > highs.iloc[-2] else -1)
            pattern_sls["Flag"] = current - height * 0.5

    # New: Cup & Handle (bullish, 65% breakout)
    if len(peak_indices) >= 4 and len(trough_indices) >= 4:
        troughs = df_close[trough_indices[-4:]]
        peaks = df_close[peak_indices[-4:]]
        if np.isclose(troughs.iloc[0], troughs.iloc[2], rtol=0.02) and np.isclose(peaks.iloc[0], peaks.iloc[3], rtol=0.02) and troughs.iloc[3] > troughs.iloc[1]:  # U-shape + handle
            patterns.append("Cup & Handle")
            pattern_targets["Cup & Handle"] = peaks.iloc[3] + (peaks.iloc[3] - troughs.iloc[1])
            pattern_sls["Cup & Handle"] = troughs.iloc[1] - (peaks.iloc[3] - troughs.iloc[1]) * 0.5

    # New: Ascending/Descending Channel (trend, 75% continuation)
    if len(peak_indices) >= 4 and len(trough_indices) >= 4:
        peak_slope = (peak_indices[-1] - peak_indices[-4]) / 3  # Simple slope
        trough_slope = (trough_indices[-1] - trough_indices[-4]) / 3
        if abs(peak_slope - trough_slope) < 0.01:  # Parallel
            patterns.append("Ascending Channel" if peak_slope > 0 else "Descending Channel")
            height = (df_close[peak_indices[-1]] - df_close[trough_indices[-1]])
            pattern_targets["Ascending Channel"] = current + height * 1.5
            pattern_sls["Ascending Channel"] = current - height * 0.5

    logger.info(f"Detected chart patterns: {patterns}")
    return patterns, pattern_targets, pattern_sls

def get_peak_indices(series):
    return series.index[(series.shift(1) < series) & (series.shift(-1) < series)].tolist()

def get_trough_indices(series):
    return series.index[(series.shift(1) > series) & (series.shift(-1) > series)].tolist()

# Enhanced analyze_single_timeframe (add pattern vol filter, TF-tuned trend)
def analyze_single_timeframe(df_input, current_price, entry_price, timeframe, quantity, position_type, client):
    coin = df_input['coin'].iloc[0] if not df_input.empty and 'coin' in df_input.columns else "BTCUSDT"
    market = df_input['market'].iloc[0] if not df_input.empty and 'market' in df_input.columns else "futures"
    # Enhanced: Fetch STF and HTF if timeframe allows
    try:
        df_stf = get_ohlcv_df(client, coin, timeframe, market)
        htf = "1h" if timeframe == "15m" else "4h"
        df_htf = get_ohlcv_df(client, coin, htf, market)
    except Exception as e:
        logger.warning(f"Failed to fetch multi-TF data: {e}")
        df_stf = df_htf = df_input

    indicators_stf = compute_indicators_for_df(df_stf, timeframe)
    indicators_htf = compute_indicators_for_df(df_htf, htf)
    stf_trend = classify_trend(df_stf, timeframe)
    htf_trend = classify_trend(df_htf, htf)
    volume_status = get_volume_status(df_stf)
    market_structure = detect_market_structure(df_stf)

    # Multi-TF Logic (enhanced retrace penalty for short TFs)
    retrace_penalty = -15 if timeframe == "15m" else -20  # Less penalty for noise
    agreement_bonus = 10
    if htf_trend == "bullish" and stf_trend == "bullish":
        market_bias = "Strong Bullish"
    elif htf_trend == "bearish" and stf_trend == "bearish":
        market_bias = "Strong Bearish"
    elif htf_trend == "bullish" and stf_trend == "bearish":
        market_bias = "Bullish Retracement"
        agreement_bonus = 0
    elif htf_trend == "bearish" and stf_trend == "bullish":
        market_bias = "Bearish Retracement"
        agreement_bonus = 0
    else:
        market_bias = "Sideways"
        agreement_bonus = 0

    # Enhanced Score (add BB squeeze for short TFs)
    score = 0
    direction = 1 if "Bullish" in market_bias else -1 if "Bearish" in market_bias else 0
    if direction != 0:
        ema_fast_key = 'ema10' if timeframe == "15m" else 'ema20'
        ema_slow_key = 'ema21' if timeframe == "15m" else 'ema50'
        if indicators_stf['st_color'] == ('green' if direction > 0 else 'red'):
            score += 30
        if (indicators_stf[ema_fast_key] > indicators_stf[ema_slow_key] if direction > 0 else indicators_stf[ema_fast_key] < indicators_stf[ema_slow_key]):
            score += 20
        if market_structure == ("HH-HL" if direction > 0 else "LH-LL"):
            score += 20
        vol_mult = 1.3 if volume_status == "High" else 0.7 if volume_status == "Low" else 1.0
        score *= vol_mult
        score += agreement_bonus
        hist_dir = indicators_stf['hist'] > 0 if direction > 0 else indicators_stf['hist'] < 0
        rsi_dir = indicators_stf['rsi'] > 50 if direction > 0 else indicators_stf['rsi'] < 50
        if hist_dir and rsi_dir:
            score += 15
        # Enhanced: BB squeeze (low vol, potential breakout)
        bb_width = (indicators_stf['bb_upper'] - indicators_stf['bb_lower']) / indicators_stf['sma20']
        if bb_width < 0.05 and indicators_htf['adx'] > 25:  # Squeeze + trend
            score += 10
        score += retrace_penalty if "Retracement" in market_bias else 0
    score = max(0, min(100, score))

    # Trend Label (thresholds tuned for short TFs)
    if score >= 70:  # Lowered for 15m sensitivity
        trend_label = "Strong " + ("Bullish" if direction > 0 else "Bearish")
    elif score >= 50:
        trend_label = "Bullish" if direction > 0 else "Bearish"
    elif score >= 30:
        trend_label = "Ranging / Sideways"
    else:
        trend_label = "Uncertain / No Clear Trend"

    # S/R from HTF
    _, _, lookback_months_htf = get_interval_mapping(htf)[:3]
    supports, resistances = detect_supports_resistances_for_df(df_htf, current_price, lookback_months_htf)
    s1 = supports[0] if supports else current_price * 0.98
    s2 = supports[1] if len(supports) > 1 else current_price * 0.95
    r1 = resistances[0] if resistances else current_price * 1.02
    r2 = resistances[1] if len(resistances) > 1 else current_price * 1.05

    # Patterns with vol filter
    detected_candles, body_to_range = detect_candlestick_patterns(df_stf)
    detected_charts, pattern_targets, pattern_sls = detect_chart_patterns(df_stf)
    patterns = detected_candles + detected_charts

    ema_fast_key = 'ema10' if timeframe == "15m" else 'ema20'
    ema_slow_key = 'ema21' if timeframe == "15m" else 'ema50'
    ema_fast_val = indicators_stf[ema_fast_key]
    ema_slow_val = indicators_stf[ema_slow_key]  # Fixed: was 'ma50', now dynamic
    adx = indicators_htf['adx']
    hist = indicators_stf['hist']
    rsi = indicators_stf['rsi']
    atr = indicators_stf['atr']
    sma20 = indicators_stf['sma20']
    bb_upper = indicators_stf['bb_upper']
    bb_lower = indicators_stf['bb_lower']
    vol_confirm = df_stf['volume'].iloc[-1] > indicators_stf['vol_sma20'] if not df_stf.empty else False

    cross_tf_confirm = htf_trend == stf_trend and stf_trend != "sideways"

    # Bullish/Bearish scores (fixed key)
    bullish_score = 0.0
    bearish_score = 0.0
    if ema_fast_val > ema_slow_val:
        bullish_score += 0.3
        if adx > 25:
            bullish_score += 0.2 * (adx / 50)
    elif ema_fast_val < ema_slow_val:
        bearish_score += 0.3
        if adx > 25:
            bearish_score += 0.2 * (adx / 50)
    if rsi > 60:
        bullish_score += 0.2 * ((rsi - 60) / 40)
    elif rsi < 40:
        bearish_score += 0.2 * ((40 - rsi) / 40)
    if hist > 0:
        bullish_score += 0.15
    elif hist < 0:
        bearish_score += 0.15
    if current_price > bb_upper:
        bullish_score += 0.2
    elif current_price < bb_lower:
        bearish_score += 0.2
    if vol_confirm:
        bullish_score += 0.1 if ema_fast_val > ema_slow_val or rsi > 50 or hist > 0 else 0.0
        bearish_score += 0.1 if ema_fast_val < ema_slow_val or rsi < 50 or hist < 0 else 0.0
    bullish_patterns = [p for p in patterns if p in AnalysisEngine.bullish_candles + AnalysisEngine.bullish_charts]
    bearish_patterns = [p for p in patterns if p in AnalysisEngine.bearish_candles + AnalysisEngine.bearish_charts]
    bullish_score += 0.15 * len(bullish_patterns)
    bearish_score += 0.15 * len(bearish_patterns)
    if cross_tf_confirm:
        bullish_score += 0.1 if ema_fast_val > ema_slow_val else 0.0
        bearish_score += 0.1 if ema_fast_val < ema_slow_val else 0.0
    total_score = bullish_score + bearish_score
    bullish_pct = (bullish_score / total_score * 100) if total_score > 0 else 50
    bearish_pct = (bearish_score / total_score * 100) if total_score > 0 else 50
    is_stable = body_to_range < 0.3 and adx < 20 and 40 <= rsi <= 60 and not (bullish_patterns or bearish_patterns) and not vol_confirm
    if is_stable:
        sentiment = "Neutral/Sideways"
        market_confidence = {"bullish": 50.0, "bearish": 50.0}
    elif bullish_score >= 0.7 or (bullish_score >= 0.5 and adx > 25 and cross_tf_confirm):
        sentiment = "Strong Bullish" if bullish_score >= 0.9 and adx > 30 else "Bullish"
        market_confidence = {"bullish": min(bullish_pct, 95.0), "bearish": max(100 - bullish_pct, 5.0)}
    elif bearish_score >= 0.7 or (bearish_score >= 0.5 and adx > 25 and cross_tf_confirm):
        sentiment = "Strong Bearish" if bearish_score >= 0.9 and adx > 30 else "Bearish"
        market_confidence = {"bullish": max(100 - bearish_pct, 5.0), "bearish": min(bearish_pct, 95.0)}
    else:
        sentiment = "Neutral/Sideways"
        market_confidence = {"bullish": min(bullish_pct, 60.0), "bearish": min(bearish_pct, 60.0)}

    position_confidence = 1.0
    price_diff = abs(entry_price - current_price) / atr if atr > 0 else float('inf')
    if price_diff <= 1.0:
        position_confidence += 20.0
    if position_type == "long" and bullish_score > bearish_score:
        position_confidence += 30.0 * (bullish_score / (bullish_score + bearish_score + 1e-10))
    elif position_type == "short" and bearish_score > bullish_score:
        position_confidence += 30.0 * (bearish_score / (bullish_score + bearish_score + 1e-10))
    else:
        position_confidence += 10.0
    position_confidence += 20.0 * (adx / 50) if adx > 25 else 0.0
    position_confidence += 10.0 if vol_confirm else 0.0
    position_confidence += 5.0 * len(bullish_patterns if position_type == "long" else bearish_patterns)
    position_confidence = min(max(position_confidence, 1.0), 95.0)

    logger.info(f"Market Confidence: Bullish {market_confidence['bullish']:.1f}%, Bearish {market_confidence['bearish']:.1f}%")
    logger.info(f"Position Confidence: {position_confidence:.1f}% (Price diff: {price_diff:.2f} ATR, Patterns: {len(bullish_patterns if position_type == 'long' else bearish_patterns)})")

    atr_15m = indicators_stf['atr']
    targets, user_sl, market_sl = AnalysisEngine.target_stoploss_enhanced(
        market_bias, supports, resistances, entry_price, pattern_targets, pattern_sls, atr_15m, current_price, timeframe, position_type, df_stf, indicators_stf
    )

    enhanced_sentiment = {
        "Market Bias": market_bias,
        "Strength": f"{score}%",
        "Short-Term Trend ({timeframe})": stf_trend.capitalize(),
        "Higher-Timeframe Trend ({htf})": htf_trend.capitalize(),
        "Volume Status": volume_status,
        "Market Structure": market_structure,
        "Support Levels": f"S1: {round(s1, 2)}, S2: {round(s2, 2)}",
        "Resistance Levels": f"R1: {round(r1, 2)}, R2: {round(r2, 2)}",
        "Suggested TGT": round(targets[0], 2) if targets and targets[0] else None,
        "Suggested SL": round(user_sl, 2) if user_sl else None,
        "Note": "This is market sentiment guidance only, not a buy/sell signal."
    }

    if market_bias == "Sideways":
        enhanced_sentiment["Note"] += " Market is sideways, avoid large TGT/SL planning."

    return {
        'enhanced_sentiment': enhanced_sentiment,
        'sentiment': sentiment,
        'market_confidence': market_confidence,
        'position_confidence': round(position_confidence),
        'targets': targets if targets else [None, None],
        'user_sl': user_sl,
        'market_sl': market_sl,
        'patterns': patterns
    }

# Enhanced AnalysisEngine (tighter SL for short TFs)
class AnalysisEngine:
    bullish_candles = ["Hammer", "Inverted Hammer", "Bullish Engulfing", "Piercing Line", "Morning Star", "Bullish Harami", "Three White Soldiers", "Tweezer Bottom", "Bullish Belt Hold", "Bullish Marubozu"]
    bearish_candles = ["Shooting Star", "Hanging Man", "Bearish Engulfing", "Dark Cloud Cover", "Evening Star", "Bearish Harami", "Three Black Crows", "Tweezer Top", "Bearish Belt Hold", "Bearish Marubozu"]
    bullish_charts = ["Inverse Head and Shoulders", "Double Bottom", "Triple Bottom", "Ascending Triangle", "Falling Wedge", "Ascending Channel"]
    bearish_charts = ["Head and Shoulders", "Double Top", "Triple Top", "Descending Triangle", "Rising Wedge", "Descending Channel"]

    @staticmethod
    def profitability(position_type, entry_price, current_price, quantity):
        pl = (current_price - entry_price) * quantity if position_type == "long" else (entry_price - current_price) * quantity
        pl_pct = (pl / (entry_price * quantity)) * 100 if entry_price * quantity != 0 else 0
        return f"{pl:.2f} ({pl_pct:.2f}%)"

    @staticmethod
    def target_stoploss_enhanced(bias, supports, resistances, entry_price, pattern_targets, pattern_sls, atr, current_price, timeframe, position_type, df_stf, indicators_stf):
        targets = []
        user_sl = None
        market_sl = None
        position_dir = 1 if position_type.lower() == "long" else -1
        tf_mult = 0.8 if timeframe == "15m" else 1.0
        adx_scale = 2.5 if indicators_stf.get('adx', 25) > 25 else 1.5

        # Determine bias_dir
        bias_dir = 1 if "Bullish" in bias else -1 if "Bearish" in bias else 0

        # Alignment multiplier
        if bias_dir == position_dir:
            align_mult = 1.0
        elif bias_dir == 0:  # Sideways
            align_mult = 0.7
        else:  # Opposes
            align_mult = 0.3

        tgt_mult = 1.8 * align_mult
        sl_mult = 1.2 * (1 + 0.5 * (1 - align_mult))  # Tighter SL when less aligned (smaller mult means tighter? Wait, adjust: higher mult for tighter? No.
        # Actually, for less align, smaller sl_mult for tighter SL
        sl_mult = 1.2 * align_mult  # Smaller when opposes

        ema_slow_key = 'ema21' if timeframe == "15m" else 'ema50'
        ema = indicators_stf.get(ema_slow_key, entry_price)
        st = df_stf['supertrend'].iloc[-1] if not df_stf.empty and 'supertrend' in df_stf.columns else ema
        dynamic_sl = min(ema, st) if position_dir > 0 else max(ema, st)  # Adverse dynamic for position

        # TGT calculation
        tgt_val = entry_price + (tgt_mult * atr * adx_scale * tf_mult * position_dir)
        if position_dir > 0:  # Long: nearest resistance above
            candidates = [r for r in resistances if r > entry_price]
            if candidates:
                nearest = min(candidates, key=lambda x: abs(x - tgt_val))
                targets.append(nearest)
            else:
                targets.append(tgt_val)
        else:  # Short: nearest support below
            candidates = [s for s in supports if s < entry_price]
            if candidates:
                nearest = min(candidates, key=lambda x: abs(x - tgt_val))
                targets.append(nearest)
            else:
                targets.append(tgt_val)

        # User SL calculation: adverse to position
        sl_val = entry_price - (sl_mult * atr * tf_mult * position_dir)  # For long: entry - positive = below; short: entry - (pos * -1) = entry + pos = above
        # Snap to dynamic if closer/adverse
        if abs(sl_val - dynamic_sl) < atr * 0.5:  # If dynamic close, use it
            user_sl = dynamic_sl
        else:
            user_sl = sl_val

        # Market SL: closest adverse level
        if position_dir > 0:  # Long: closest support below current
            sl_candidates = [s for s in supports if s < current_price]
            if sl_candidates:
                market_sl = max(sl_candidates)  # Highest (closest) below current
            else:
                market_sl = user_sl
        else:  # Short: closest resistance above current
            sl_candidates = [r for r in resistances if r > current_price]
            if sl_candidates:
                market_sl = min(sl_candidates)  # Lowest (closest) above current
            else:
                market_sl = user_sl

        # R:R adjustment in position direction
        if targets and user_sl and atr > 0:
            risk = abs(entry_price - user_sl)
            reward = abs(targets[0] - entry_price)
            if risk > 0 and reward / risk < 1.8:
                targets[0] = entry_price + (risk * 1.8 * position_dir)

        # Ensure at least 2 targets
        if len(targets) < 2:
            if targets:
                targets.append(targets[0] + (atr * 1.2 * tf_mult * position_dir))
            else:
                targets = [None, None]

        return targets, user_sl, market_sl

    # Original target_stoploss unchanged for compat
    @staticmethod
    def target_stoploss(sentiment, supports, resistances, entry_price, pattern_targets, pattern_sls, atr, current_price, timeframe, position_type, df=None):
        # As original
        pass  # Omitted for space; keep as is

# Main Pipeline
def advisory_pipeline(client, input_json):
    try:
        data = InputValidator.validate_and_normalize(input_json)
        logger.info(f"Processing input: {data}")

        if data["market"] == "spot":
            info = client.get_instruments_info(category="spot", symbol=data["coin"])
            if not info['result']['list'] or info['result']['list'][0]['status'] != 'Trading':
                raise ValueError(f"Coin {data['coin']} not found or not trading on Bybit spot market!")
        else:
            futures_info = client.get_instruments_info(category="linear", symbol=data["coin"])
            if not futures_info['result']['list'] or futures_info['result']['list'][0]['status'] != 'Trading':
                raise ValueError(f"Coin {data['coin']} not found or not trading on Bybit futures market!")

        current_price = get_current_price(client, data["coin"], data["market"])
        df = get_ohlcv_df(client, data["coin"], data["timeframe"], data["market"])
        df['coin'] = data["coin"]
        df['market'] = data["market"]
        analysis = analyze_single_timeframe(df, current_price, data["entry_price"], data["timeframe"], data["quantity"], data["position_type"], client)

        profit_loss = AnalysisEngine.profitability(data["position_type"], data["entry_price"], current_price, data["quantity"])
        warning = ""
        if analysis['sentiment'] in ["Bullish", "Strong Bullish"] and data["position_type"] == "short":
            warning = "Your short position opposes the bullish trend—consider exiting."
        elif analysis['sentiment'] in ["Bearish", "Strong Bearish"] and data["position_type"] == "long":
            warning = "Your long position opposes the bearish trend—consider exiting."
        elif analysis['sentiment'] == "Neutral/Sideways":
            warning = "Market is neutral/sideways—avoid new positions or tighten stop-loss."

        output = {
            "trade_summary": {
                "coin": data["coin"],
                "market": data["market"],
                "position_type": data["position_type"],
                "entry_price": data["entry_price"],
                "current_price": current_price,
                "profit_loss": profit_loss
            },
            "targets_and_stoplosses": {
                "tgt1": round(analysis['targets'][0], 2) if analysis['targets'][0] else None,
                "tgt2": round(analysis['targets'][1], 2) if len(analysis['targets']) > 1 and analysis['targets'][1] else None,
                "user_sl": round(analysis['user_sl'], 2) if analysis['user_sl'] else None,
                "market_sl": round(analysis['market_sl'], 2) if analysis['market_sl'] else None
            },
            "market_confidence": analysis['market_confidence'],
            "position_confidence": f"{analysis['position_confidence']}%",
            "patterns": analysis['patterns'],
            "sentiment": analysis['sentiment'],
            "warning": warning,
            "enhanced_sentiment": analysis['enhanced_sentiment']
        }
        logger.info(f"Analysis output: {output}")
        return output
    except ValueError as ve:
        logger.error(f"ValueError in advisory_pipeline: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in advisory_pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Input Model
class TradeInput(BaseModel):
    coin: str
    market: str
    entry_price: float
    quantity: float
    position_type: str
    timeframe: str

@app.get("/")
def root():
    return {"message": "Welcome to the Trading Analysis API. Use /analyze with POST to analyze positions."}

@app.options("/analyze")
def options_analyze(request: Request):
    return {
        "status_code": 200,
        "headers": {
            "Access-Control-Allow-Origin": "https://position-analyzer-app.vercel.app",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    }

@app.post("/analyze")
def analyze_position(input_data: TradeInput):
    try:
        client = initialize_client()
        data = input_data.dict()
        validated_data = InputValidator.validate_and_normalize(data)
        output = advisory_pipeline(client, validated_data)
        return output
    except ValueError as ve:
        logger.error(f"ValueError in /analyze: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error in /analyze: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")