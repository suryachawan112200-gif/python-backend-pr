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
    VALID_TIMEFRAMES = ["1h", "4h", "1d"]

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
        "1h": 60,
        "4h": 240,
        "1d": "D"
    }
    multiplier_adjust = {
        "1h": 2.0,
        "4h": 2.5,
        "1d": 3.0
    }
    lookback_periods = {
        "1h": 2160,
        "4h": 540,
        "1d": 90
    }
    lookback_months = {
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

# Indicator & Analysis Helpers
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

def compute_indicators_for_df(df):
    if df.empty:
        return {'ema20': 0, 'ma50': 0, 'atr': 0, 'rsi': 50, 'macd': 0, 'signal': 0, 'hist': 0, 'adx': 25, 'sma20': 0, 'bb_upper': 0, 'bb_lower': 0}
    close = df['close']
    ema20 = compute_ema(close, 20).iloc[-1]
    ma50 = compute_ma(close, 50).iloc[-1]
    atr = compute_atr(df)
    rsi = compute_rsi(close)
    macd, signal, hist = compute_macd(close)
    adx = compute_adx(df)
    sma20, bb_upper, bb_lower = compute_bollinger_bands(close)
    return {
        'ema20': ema20,
        'ma50': ma50,
        'atr': atr,
        'rsi': rsi,
        'macd': macd,
        'signal': signal,
        'hist': hist,
        'adx': adx,
        'sma20': sma20,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower
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
    tol = 0.005
    lows = df_lookback['low'].rolling(window, center=True).min()
    highs = df_lookback['high'].rolling(window, center=True).max()
    candidates_s = df_lookback[(df_lookback['low'] == lows) & (df_lookback['volume'] > avg_volume)]['low'].unique()
    candidates_r = df_lookback[(df_lookback['high'] == highs) & (df_lookback['volume'] > avg_volume)]['high'].unique()

    def get_touches_and_vol(level, series, vol_series, tol):
        touches = abs(series - level) < level * tol
        count = touches.sum()
        vol_weight = vol_series[touches].mean() / avg_volume if count > 0 else 0
        return count, vol_weight

    supports = []
    for level in candidates_s:
        count, vol_w = get_touches_and_vol(level, df_lookback['close'], df_lookback['volume'], tol)
        if count >= 1:
            supports.append((level, count, vol_w))
    resistances = []
    for level in candidates_r:
        count, vol_w = get_touches_and_vol(level, df_lookback['close'], df_lookback['volume'], tol)
        if count >= 1:
            resistances.append((level, count, vol_w))

    supports.sort(key=lambda x: x[1] * x[2], reverse=True)
    resistances.sort(key=lambda x: x[1] * x[2], reverse=True)
    supports = [x[0] for x in supports]
    resistances = [x[0] for x in resistances]

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

def detect_candlestick_patterns(df):
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

    if body_to_range < 0.05 and last_volume > avg_volume * 1.5:
        patterns.append("Doji")
    if lower_shadow > 2 * body and upper_shadow < 0.1 * body and last_close > last_open and last_volume > avg_volume:
        patterns.append("Hammer")
    if upper_shadow > 2 * body and lower_shadow < 0.1 * body and last_close > last_open and last_volume > avg_volume:
        patterns.append("Inverted Hammer")
    if upper_shadow > 2 * body and lower_shadow < 0.1 * body and last_close < last_open and last_volume > avg_volume:
        patterns.append("Shooting Star")
    if lower_shadow > 2 * body and upper_shadow < 0.1 * body and last_close < last_open and last_volume > avg_volume:
        patterns.append("Hanging Man")
    if body_to_range < 0.3 and upper_shadow > body and lower_shadow > body and last_volume > avg_volume:
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
        if p2_close < p2_open and abs(p1_close - p1_open) < 0.3 * abs(p2_close - p2_open) and last_close > last_open and last_close > (p2_open + p2_close) / 2 and last_volume > p1_volume * 1.2:
            patterns.append("Morning Star")
        if p2_close > p2_open and abs(p1_close - p1_open) < 0.3 * abs(p2_close - p2_open) and last_close < last_open and last_close < (p2_open + p2_close) / 2 and last_volume > p1_volume * 1.2:
            patterns.append("Evening Star")
        if all(close.iloc[-i] > open_.iloc[-i] for i in range(1, 4)) and close.iloc[-2] > close.iloc[-3] and close.iloc[-1] > close.iloc[-2] and last_volume > avg_volume:
            patterns.append("Three White Soldiers")
        if all(close.iloc[-i] < open_.iloc[-i] for i in range(1, 4)) and close.iloc[-2] < close.iloc[-3] and close.iloc[-1] < close.iloc[-2] and last_volume > avg_volume:
            patterns.append("Three Black Crows")

    logger.info(f"Detected candlestick patterns: {patterns}, Body-to-range: {body_to_range:.2f}")
    return patterns, body_to_range

def get_peak_indices(series):
    return series.index[(series.shift(1) < series) & (series.shift(-1) < series)].tolist()

def get_trough_indices(series):
    return series.index[(series.shift(1) > series) & (series.shift(-1) > series)].tolist()

def detect_chart_patterns(df):
    if df.empty:
        logger.warning("Empty DataFrame in detect_chart_patterns, returning empty patterns")
        return [], {}, {}
    
    df_close = df['close']
    patterns = []
    pattern_targets = {}
    pattern_sls = {}
    peak_indices = get_peak_indices(df_close)[-10:]
    trough_indices = get_trough_indices(df_close)[-10:]
    current = df_close.iloc[-1]

    if len(peak_indices) >= 2:
        p1_idx = peak_indices[-2]
        p2_idx = peak_indices[-1]
        p1 = df_close[p1_idx]
        p2 = df_close[p2_idx]
        if abs(p1 - p2) / ((p1 + p2) / 2) < 0.015 and p2_idx > p1_idx:
            slice_data = df_close[p1_idx:p2_idx]
            if not slice_data.empty:
                patterns.append("Double Top")
                neckline = slice_data.min()
                height = p1 - neckline
                pattern_targets["Double Top"] = neckline - height
                pattern_sls["Double Top"] = max(p1, p2) + height * 0.1
                tgt = pattern_targets["Double Top"]
                sl = pattern_sls["Double Top"]
                if not (current > tgt and current < sl):
                    patterns.remove("Double Top")
                    del pattern_targets["Double Top"]
                    del pattern_sls["Double Top"]
            else:
                logger.warning(f"Empty slice for Double Top between indices {p1_idx}:{p2_idx}")

    if len(trough_indices) >= 2:
        t1_idx = trough_indices[-2]
        t2_idx = trough_indices[-1]
        t1 = df_close[t1_idx]
        t2 = df_close[t2_idx]
        if abs(t1 - t2) / ((t1 + t2) / 2) < 0.015 and t2_idx > t1_idx:
            slice_data = df_close[t1_idx:t2_idx]
            if not slice_data.empty:
                patterns.append("Double Bottom")
                neckline = slice_data.max()
                height = neckline - t1
                pattern_targets["Double Bottom"] = neckline + height
                pattern_sls["Double Bottom"] = min(t1, t2) - height * 0.1
                tgt = pattern_targets["Double Bottom"]
                sl = pattern_sls["Double Bottom"]
                if not (current < tgt and current > sl):
                    patterns.remove("Double Bottom")
                    del pattern_targets["Double Bottom"]
                    del pattern_sls["Double Bottom"]
            else:
                logger.warning(f"Empty slice for Double Bottom between indices {t1_idx}:{t2_idx}")

    if len(peak_indices) >= 3:
        p1_idx = peak_indices[-3]
        p2_idx = peak_indices[-2]
        p3_idx = peak_indices[-1]
        p1 = df_close[p1_idx]
        p2 = df_close[p2_idx]
        p3 = df_close[p3_idx]
        if max(abs(p1 - p2), abs(p2 - p3), abs(p1 - p3)) / ((p1 + p2 + p3) / 3) < 0.015 and p3_idx > p2_idx > p1_idx:
            slice_data = df_close[p1_idx:p3_idx+1]
            if not slice_data.empty:
                patterns.append("Triple Top")
                neckline = slice_data.min()
                height = (p1 + p2 + p3)/3 - neckline
                pattern_targets["Triple Top"] = neckline - height
                pattern_sls["Triple Top"] = max(p1, p2, p3) + height * 0.1
                tgt = pattern_targets["Triple Top"]
                sl = pattern_sls["Triple Top"]
                if not (current > tgt and current < sl):
                    patterns.remove("Triple Top")
                    del pattern_targets["Triple Top"]
                    del pattern_sls["Triple Top"]
            else:
                logger.warning(f"Empty slice for Triple Top between indices {p1_idx}:{p3_idx+1}")
        if p2 > p1 and p2 > p3 and abs(p1 - p3) / p2 < 0.02 and p3_idx > p2_idx > p1_idx:
            slice_data = df_close[p1_idx:p3_idx+1]
            if not slice_data.empty:
                patterns.append("Head and Shoulders")
                neckline = (p1 + p3) / 2
                height = p2 - neckline
                pattern_targets["Head and Shoulders"] = neckline - height
                pattern_sls["Head and Shoulders"] = p2 + height * 0.1
                tgt = pattern_targets["Head and Shoulders"]
                sl = pattern_sls["Head and Shoulders"]
                if not (current > tgt and current < sl):
                    patterns.remove("Head and Shoulders")
                    del pattern_targets["Head and Shoulders"]
                    del pattern_sls["Head and Shoulders"]
            else:
                logger.warning(f"Empty slice for Head and Shoulders between indices {p1_idx}:{p3_idx+1}")

    if len(trough_indices) >= 3:
        t1_idx = trough_indices[-3]
        t2_idx = trough_indices[-2]
        t3_idx = trough_indices[-1]
        t1 = df_close[t1_idx]
        t2 = df_close[t2_idx]
        t3 = df_close[t3_idx]
        if max(abs(t1 - t2), abs(t2 - t3), abs(t1 - t3)) / ((t1 + t2 + t3) / 3) < 0.015 and t3_idx > t2_idx > t1_idx:
            slice_data = df_close[t1_idx:t3_idx+1]
            if not slice_data.empty:
                patterns.append("Triple Bottom")
                neckline = slice_data.max()
                height = neckline - (t1 + t2 + t3)/3
                pattern_targets["Triple Bottom"] = neckline + height
                pattern_sls["Triple Bottom"] = min(t1, t2, t3) - height * 0.1
                tgt = pattern_targets["Triple Bottom"]
                sl = pattern_sls["Triple Bottom"]
                if not (current < tgt and current > sl):
                    patterns.remove("Triple Bottom")
                    del pattern_targets["Triple Bottom"]
                    del pattern_sls["Triple Bottom"]
            else:
                logger.warning(f"Empty slice for Triple Bottom between indices {t1_idx}:{t3_idx+1}")
        if t2 < t1 and t2 < t3 and abs(t1 - t3) / abs(t2) < 0.02 and t3_idx > t2_idx > t1_idx:
            slice_data = df_close[t1_idx:t3_idx+1]
            if not slice_data.empty:
                patterns.append("Inverse Head and Shoulders")
                neckline = (t1 + t3) / 2
                height = neckline - t2
                pattern_targets["Inverse Head and Shoulders"] = neckline + height
                pattern_sls["Inverse Head and Shoulders"] = t2 - height * 0.1
                tgt = pattern_targets["Inverse Head and Shoulders"]
                sl = pattern_sls["Inverse Head and Shoulders"]
                if not (current < tgt and current > sl):
                    patterns.remove("Inverse Head and Shoulders")
                    del pattern_targets["Inverse Head and Shoulders"]
                    del pattern_sls["Inverse Head and Shoulders"]
            else:
                logger.warning(f"Empty slice for Inverse Head and Shoulders between indices {t1_idx}:{t3_idx+1}")

    if len(peak_indices) >= 2 and len(trough_indices) >= 2:
        p1_idx = peak_indices[-2]
        p2_idx = peak_indices[-1]
        t1_idx = trough_indices[-2]
        t2_idx = trough_indices[-1]
        p1 = df_close[p1_idx]
        p2 = df_close[p2_idx]
        t1 = df_close[t1_idx]
        t2 = df_close[t2_idx]

        slope_peak = (p2 - p1) / (p2_idx - p1_idx + 1e-6)
        slope_trough = (t2 - t1) / (t2_idx - t1_idx + 1e-6)

        if abs(p2 - p1) / df_close.mean() < 0.01 and t2 > t1 and p2_idx > p1_idx and t2_idx > t1_idx:
            patterns.append("Ascending Triangle")
            height = p1 - t1
            pattern_targets["Ascending Triangle"] = p2 + height
            pattern_sls["Ascending Triangle"] = min(t1, t2) - height * 0.1
            tgt = pattern_targets["Ascending Triangle"]
            if not (current < tgt):
                patterns.remove("Ascending Triangle")
                del pattern_targets["Ascending Triangle"]
                del pattern_sls["Ascending Triangle"]

        if abs(t2 - t1) / df_close.mean() < 0.01 and p2 < p1 and p2_idx > p1_idx and t2_idx > t1_idx:
            patterns.append("Descending Triangle")
            height = p1 - t1
            pattern_targets["Descending Triangle"] = t2 - height
            pattern_sls["Descending Triangle"] = max(p1, p2) + height * 0.1
            tgt = pattern_targets["Descending Triangle"]
            if not (current > tgt):
                patterns.remove("Descending Triangle")
                del pattern_targets["Descending Triangle"]
                del pattern_sls["Descending Triangle"]

        if p2 < p1 and t2 > t1 and p2_idx > p1_idx and t2_idx > t1_idx:
            patterns.append("Symmetrical Triangle")
            height = max(p1, p2) - min(t1, t2)
            pattern_targets["Symmetrical Triangle"] = current + height if current > (p1 + t1) / 2 else current - height
            pattern_sls["Symmetrical Triangle"] = min(t1, t2) - height * 0.1
            tgt = pattern_targets["Symmetrical Triangle"]
            sl = pattern_sls["Symmetrical Triangle"]
            if not (current > sl and (tgt > current or tgt < current)):
                patterns.remove("Symmetrical Triangle")
                del pattern_targets["Symmetrical Triangle"]
                del pattern_sls["Symmetrical Triangle"]

        if p2 < p1 and t2 < t1 and p2_idx > p1_idx and t2_idx > t1_idx:
            if abs(slope_peak - slope_trough) < 0.001:
                patterns.append("Descending Channel")
                height = p1 - t1
                pattern_targets["Descending Channel"] = current - height
                pattern_sls["Descending Channel"] = max(p1, p2) + height * 0.1
                tgt = pattern_targets["Descending Channel"]
                sl = pattern_sls["Descending Channel"]
                if not (current > tgt):
                    patterns.remove("Descending Channel")
                    del pattern_targets["Descending Channel"]
                    del pattern_sls["Descending Channel"]
            elif p1 - t1 > p2 - t2:
                patterns.append("Falling Wedge")
                height = p1 - t1
                pattern_targets["Falling Wedge"] = current + height
                pattern_sls["Falling Wedge"] = min(t1, t2) - height * 0.1
                tgt = pattern_targets["Falling Wedge"]
                sl = pattern_sls["Falling Wedge"]
                if not (current < tgt and current > sl):
                    patterns.remove("Falling Wedge")
                    del pattern_targets["Falling Wedge"]
                    del pattern_sls["Falling Wedge"]

        if p2 > p1 and t2 > t1 and p2_idx > p1_idx and t2_idx > t1_idx:
            if abs(slope_peak - slope_trough) < 0.001:
                patterns.append("Ascending Channel")
                height = p1 - t1
                pattern_targets["Ascending Channel"] = current + height
                pattern_sls["Ascending Channel"] = min(t1, t2) - height * 0.1
                tgt = pattern_targets["Ascending Channel"]
                sl = pattern_sls["Ascending Channel"]
                if not (current < tgt):
                    patterns.remove("Ascending Channel")
                    del pattern_targets["Ascending Channel"]
                    del pattern_sls["Ascending Channel"]
            elif p2 - t2 < p1 - t1:
                patterns.append("Rising Wedge")
                height = p1 - t1
                pattern_targets["Rising Wedge"] = current - height
                pattern_sls["Rising Wedge"] = max(p1, p2) + height * 0.1
                tgt = pattern_targets["Rising Wedge"]
                sl = pattern_sls["Rising Wedge"]
                if not (current > tgt and current < sl):
                    patterns.remove("Rising Wedge")
                    del pattern_targets["Rising Wedge"]
                    del pattern_sls["Rising Wedge"]

    logger.info(f"Detected chart patterns: {patterns}")
    return patterns, pattern_targets, pattern_sls

def analyze_single_timeframe(df, current_price, entry_price, timeframe, quantity, position_type, client):
    indicators = compute_indicators_for_df(df)
    _, _, _, lookback_months = get_interval_mapping(timeframe)
    supports, resistances = detect_supports_resistances_for_df(df, current_price, lookback_months)
    detected_candles, body_to_range = detect_candlestick_patterns(df)
    detected_charts, pattern_targets, pattern_sls = detect_chart_patterns(df)
    patterns = detected_candles + detected_charts

    ema20 = indicators['ema20']
    ma50 = indicators['ma50']
    adx = indicators['adx']
    hist = indicators['hist']
    rsi = indicators['rsi']
    atr = indicators['atr']
    sma20 = indicators['sma20']
    bb_upper = indicators['bb_upper']
    bb_lower = indicators['bb_lower']
    vol_confirm = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] if not df.empty and len(df) >= 20 else False

    # Cross-timeframe validation for market confidence
    cross_tf_confirm = False
    if timeframe in ["1h", "4h"]:
        higher_tf = "4h" if timeframe == "1h" else "1d"
        try:
            higher_df = get_ohlcv_df(client, df['coin'].iloc[0] if not df.empty else "BTCUSDT", higher_tf, df['market'].iloc[0] if not df.empty else "futures")
            higher_indicators = compute_indicators_for_df(higher_df)
            cross_tf_confirm = (higher_indicators['ema20'] > higher_indicators['ma50'] and ema20 > ma50) or \
                              (higher_indicators['ema20'] < higher_indicators['ma50'] and ema20 < ma50)
        except Exception as e:
            logger.warning(f"Cross-timeframe validation failed: {str(e)}")
            cross_tf_confirm = False

    # Compute bullish and bearish scores for market confidence
    bullish_score = 0.0
    bearish_score = 0.0

    # Trend indicators
    if ema20 > ma50:
        bullish_score += 0.3
        if adx > 25:
            bullish_score += 0.2 * (adx / 50)  # Scale with ADX strength
    elif ema20 < ma50:
        bearish_score += 0.3
        if adx > 25:
            bearish_score += 0.2 * (adx / 50)

    # Momentum indicators
    if rsi > 60:
        bullish_score += 0.2 * ((rsi - 60) / 40)  # Scale with RSI strength
    elif rsi < 40:
        bearish_score += 0.2 * ((40 - rsi) / 40)
    if hist > 0:
        bullish_score += 0.15
    elif hist < 0:
        bearish_score += 0.15

    # Bollinger Bands: Price near upper/lower band indicates trend
    if current_price > bb_upper:
        bullish_score += 0.2
    elif current_price < bb_lower:
        bearish_score += 0.2

    # Volume confirmation
    if vol_confirm:
        bullish_score += 0.1 if ema20 > ma50 or rsi > 50 or hist > 0 else 0.0
        bearish_score += 0.1 if ema20 < ma50 or rsi < 50 or hist < 0 else 0.0

    # Pattern confirmation
    bullish_patterns = [p for p in patterns if p in AnalysisEngine.bullish_candles + AnalysisEngine.bullish_charts]
    bearish_patterns = [p for p in patterns if p in AnalysisEngine.bearish_candles + AnalysisEngine.bearish_charts]
    bullish_score += 0.15 * len(bullish_patterns)
    bearish_score += 0.15 * len(bearish_patterns)

    # Cross-timeframe confirmation
    if cross_tf_confirm:
        bullish_score += 0.1 if ema20 > ma50 else 0.0
        bearish_score += 0.1 if ema20 < ma50 else 0.0

    # Normalize market confidence to sum to 100%
    total_score = bullish_score + bearish_score
    if total_score > 0:
        bullish_pct = (bullish_score / total_score) * 100
        bearish_pct = (bearish_score / total_score) * 100
    else:
        bullish_pct = 50.0
        bearish_pct = 50.0

    # Determine sentiment
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

    # Calculate position confidence
    position_confidence = 1.0  # Minimum 1%
    price_diff = abs(entry_price - current_price) / atr if atr > 0 else float('inf')
    if price_diff <= 1.0:  # Entry price within 1 ATR of current price
        position_confidence += 20.0
    if position_type == "long" and bullish_score > bearish_score:
        position_confidence += 30.0 * (bullish_score / (bullish_score + bearish_score + 1e-10))
    elif position_type == "short" and bearish_score > bullish_score:
        position_confidence += 30.0 * (bearish_score / (bullish_score + bearish_score + 1e-10))
    else:
        position_confidence += 10.0  # Lower boost if position opposes market bias
    position_confidence += 20.0 * (adx / 50) if adx > 25 else 0.0  # Trend strength boost
    position_confidence += 10.0 if vol_confirm else 0.0  # Volume confirmation
    position_confidence += 5.0 * len(bullish_patterns if position_type == "long" else bearish_patterns)
    position_confidence = min(max(position_confidence, 1.0), 95.0)  # Cap between 1% and 95%

    # Log indicator contributions
    logger.info(f"Market Confidence: Bullish {market_confidence['bullish']:.1f}%, Bearish {market_confidence['bearish']:.1f}%")
    logger.info(f"Position Confidence: {position_confidence:.1f}% (Price diff: {price_diff:.2f} ATR, Patterns: {len(bullish_patterns if position_type == 'long' else bearish_patterns)})")
    logger.info(f"Indicators: EMA20={ema20:.2f}, MA50={ma50:.2f}, ADX={adx:.1f}, RSI={rsi:.1f}, MACD Hist={hist:.4f}, Vol Confirm={vol_confirm}, Cross TF={cross_tf_confirm}")

    # Targets and Stop-losses
    targets, user_sl, market_sl = AnalysisEngine.target_stoploss(
        sentiment, supports, resistances, entry_price, pattern_targets, pattern_sls, atr, current_price, timeframe, position_type, df
    )

    return {
        'sentiment': sentiment,
        'market_confidence': market_confidence,
        'position_confidence': round(position_confidence),
        'targets': targets,
        'user_sl': user_sl,
        'market_sl': market_sl,
        'patterns': patterns
    }

# Analysis Engine
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
    def target_stoploss(sentiment, supports, resistances, entry_price, pattern_targets, pattern_sls, atr, current_price, timeframe, position_type, df=None):
        targets = []
        user_sl = None
        market_sl = None
        trend_strength = abs((compute_ema(df['close'], 20).iloc[-1] - compute_ma(df['close'], 50).iloc[-1]) / compute_ma(df['close'], 50).iloc[-1]) if df is not None and not df.empty and compute_ma(df['close'], 50).iloc[-1] != 0 else 0.0
        adjustment = 1.0 + trend_strength * 2
        _, timeframe_multiplier, _, _ = get_interval_mapping(timeframe)
        atr_buffer = atr * 0.5
        max_diff = atr * 10

        logger.info(f"Calculating targets/stoplosses: ATR = {atr:.6f}, Trend strength = {trend_strength:.2f}, Timeframe multiplier = {timeframe_multiplier}, Max diff = {max_diff:.6f}")

        # User stop-loss based on position and ATR
        if position_type == "long":
            user_sl = entry_price - atr * 2.0 * timeframe_multiplier
        else:
            user_sl = entry_price + atr * 2.0 * timeframe_multiplier

        if sentiment in ["Bullish", "Strong Bullish"]:
            # Targets: Use resistances and bullish pattern targets
            potential_tgts = sorted([r for r in resistances if r > current_price])
            pattern_tgts = [t for p, t in pattern_targets.items() if p in AnalysisEngine.bullish_charts and t > current_price]
            combined_tgts = sorted(list(set(potential_tgts + pattern_tgts)))
            if combined_tgts:
                tgt1 = combined_tgts[0] + atr_buffer
                targets.append(tgt1)
                if len(combined_tgts) > 1:
                    tgt2 = combined_tgts[1] + atr_buffer
                    if tgt2 - tgt1 <= max_diff * timeframe_multiplier * adjustment:
                        targets.append(tgt2)
                if len(targets) < 2:
                    targets.append(current_price + atr * 6.0 * timeframe_multiplier * adjustment)
            else:
                targets.append(current_price + atr * 3.5 * timeframe_multiplier * adjustment)
                targets.append(current_price + atr * 6.0 * timeframe_multiplier * adjustment)
            # Market stop-loss: Nearest support or bearish pattern stop-loss below current price
            potential_sls = sorted([s for s in supports if s < current_price], reverse=True)
            pattern_sls_list = [s for p, s in pattern_sls.items() if p in AnalysisEngine.bearish_charts and s < current_price]
            combined_sls = sorted(list(set(potential_sls + pattern_sls_list)), reverse=True)
            market_sl = combined_sls[0] - atr_buffer if combined_sls else current_price - atr * 2.0 * timeframe_multiplier
            logger.info(f"{sentiment}: Targets = {targets}, User SL = {user_sl}, Market SL = {market_sl}")
        elif sentiment in ["Bearish", "Strong Bearish"]:
            # Targets: Use supports and bearish pattern targets
            potential_tgts = sorted([s for s in supports if s < current_price], reverse=True)
            pattern_tgts = [t for p, t in pattern_targets.items() if p in AnalysisEngine.bearish_charts and t < current_price]
            combined_tgts = sorted(list(set(potential_tgts + pattern_tgts)), reverse=True)
            if combined_tgts:
                tgt1 = combined_tgts[0] - atr_buffer
                targets.append(tgt1)
                if len(combined_tgts) > 1:
                    tgt2 = combined_tgts[1] - atr_buffer
                    if tgt1 - tgt2 <= max_diff * timeframe_multiplier * adjustment:
                        targets.append(tgt2)
                if len(targets) < 2:
                    targets.append(current_price - atr * 6.0 * timeframe_multiplier * adjustment)
            else:
                targets.append(current_price - atr * 3.5 * timeframe_multiplier * adjustment)
                targets.append(current_price - atr * 6.0 * timeframe_multiplier * adjustment)
            # Market stop-loss: Nearest resistance or bullish pattern stop-loss above current price
            potential_sls = sorted([r for r in resistances if r > current_price])
            pattern_sls_list = [s for p, s in pattern_sls.items() if p in AnalysisEngine.bullish_charts and s > current_price]
            combined_sls = sorted(list(set(potential_sls + pattern_sls_list)))
            market_sl = combined_sls[0] + atr_buffer if combined_sls else current_price + atr * 2.0 * timeframe_multiplier
            logger.info(f"{sentiment}: Targets = {targets}, User SL = {user_sl}, Market SL = {market_sl}")
        else:  # Neutral/Sideways
            # Targets: Conservative, use nearest resistance and next level
            potential_tgts = sorted([r for r in resistances if r > current_price])
            pattern_tgts = [t for p, t in pattern_targets.items() if t > current_price]
            combined_tgts = sorted(list(set(potential_tgts + pattern_tgts)))
            if combined_tgts:
                tgt1 = combined_tgts[0] + atr_buffer
                targets.append(tgt1)
                if len(combined_tgts) > 1:
                    tgt2 = combined_tgts[1] + atr_buffer
                    if tgt2 - tgt1 <= max_diff * timeframe_multiplier * adjustment:
                        targets.append(tgt2)
                else:
                    targets.append(current_price + atr * 6.0 * timeframe_multiplier * adjustment)
            else:
                targets.append(current_price + atr * 3.5 * timeframe_multiplier * adjustment)
                targets.append(current_price + atr * 6.0 * timeframe_multiplier * adjustment)
            # Market stop-loss: Nearest support below current price
            potential_sls = sorted([s for s in supports if s < current_price], reverse=True)
            pattern_sls_list = [s for p, s in pattern_sls.items() if s < current_price]
            combined_sls = sorted(list(set(potential_sls + pattern_sls_list)), reverse=True)
            market_sl = combined_sls[0] - atr_buffer if combined_sls else current_price - atr * 2.0 * timeframe_multiplier
            logger.info(f"Neutral/Sideways: Targets = {targets}, User SL = {user_sl}, Market SL = {market_sl}")

        # Ensure exactly two targets
        targets = sorted(set(targets))[:2]
        if len(targets) < 2:
            if sentiment in ["Bullish", "Strong Bullish"] or (sentiment == "Neutral/Sideways" and position_type == "long"):
                targets.append(targets[0] + atr * 2.5 * timeframe_multiplier * adjustment if targets else current_price + atr * 6.0 * timeframe_multiplier * adjustment)
            else:
                targets.append(targets[0] - atr * 2.5 * timeframe_multiplier * adjustment if targets else current_price - atr * 6.0 * timeframe_multiplier * adjustment)
            targets = sorted(set(targets))[:2]

        # Ensure R:R ratio of at least 1.5 for targets
        risk = abs(entry_price - user_sl) if user_sl != 0 else atr
        for i in range(len(targets)):
            reward = abs(targets[i] - entry_price)
            rr = reward / risk if risk > 0 else 0
            if rr < 1.5:
                if sentiment in ["Bullish", "Strong Bullish"] or (sentiment == "Neutral/Sideways" and position_type == "long"):
                    targets[i] = entry_price + risk * 1.5
                else:
                    targets[i] = entry_price - risk * 1.5
                logger.info(f"Adjusted target {i+1} to {targets[i]} for minimum R:R of 1.5")

        return targets, user_sl, market_sl

# Main Pipeline
def advisory_pipeline(client, input_json):
    try:
        data = InputValidator.validate_and_normalize(input_json)
        logger.info(f"Processing input: {data}")

        # Validate coin symbol
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
                "tgt1": round(analysis['targets'][0], 2),
                "tgt2": round(analysis['targets'][1], 2) if len(analysis['targets']) > 1 else None,
                "user_sl": round(analysis['user_sl'], 2),
                "market_sl": round(analysis['market_sl'], 2)
            },
            "market_confidence": analysis['market_confidence'],
            "position_confidence": f"{analysis['position_confidence']}%",
            "patterns": analysis['patterns'],
            "sentiment": analysis['sentiment'],
            "warning": warning
        }
        logger.info(f"Analysis output: {output}")
        return output
    except ValueError as ve:
        logger.error(f"ValueError in advisory_pipeline: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in advisory_pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Input Model for FastAPI
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