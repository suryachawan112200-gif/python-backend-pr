import json
import random
from pybit.unified_trading import HTTP
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import os
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
import uuid

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

# Login codes (hardcoded for testing)
LOGIN_CODES = ["AIVISOR1", "AIVISOR2", "AIVISOR3", "AIVISOR4", "AIVISOR5", "AIVISOR6"]

# In-memory storage for usage tracking and paid codes
daily_usage = defaultdict(lambda: {"count": 0, "last_date": None})
daily_codes_used = defaultdict(set)
valid_codes = {}  # {code: {"expiry": timestamp_ms, "last_used": date_str}}

# Input & Validation
class InputValidator:
    REQUIRED_FIELDS = {"coin", "market", "entry_price", "quantity", "position_type", "timeframe"}
    VALID_TIMEFRAMES = ["5m", "15m", "4h", "1d", "1month"]

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
        data["has_both_positions"] = data.get("has_both_positions", False)
        data["risk_pct"] = data.get("risk_pct", 0.02)
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

def get_all_coins(client):
    spot_symbols = []
    futures_symbols = []
    try:
        spot_info = client.get_instruments_info(category="spot")
        spot_symbols = [s['symbol'] for s in spot_info['result']['list'] if s['status'] == 'Trading']
    except Exception as e:
        logger.error(f"Error fetching spot symbols: {str(e)}")
    try:
        futures_info = client.get_instruments_info(category="linear")
        futures_symbols = [s['symbol'] for s in futures_info['result']['list'] if s['status'] == 'Trading']
    except Exception as e:
        logger.error(f"Error fetching futures symbols: {str(e)}")
    all_coins = list(set(spot_symbols + futures_symbols))
    return sorted(all_coins)

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
        "5m": 5,
        "15m": 15,
        "4h": 240,
        "1d": "D",
        "1month": "M"
    }
    multiplier_adjust = {
        "5m": 1.5,
        "15m": 2.0,
        "4h": 2.5,
        "1d": 3.0,
        "1month": 5.0
    }
    lookback_periods = {
        "5m": 25920,
        "15m": 8640,
        "4h": 540,
        "1d": 90,
        "1month": 3
    }
    return mapping[timeframe], multiplier_adjust[timeframe], lookback_periods[timeframe]

def get_ohlcv_df(client, coin, timeframe, market):
    interval, _, lookback = get_interval_mapping(timeframe)
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

def get_24h_trend_df(client, coin, market):
    try:
        if market == "spot":
            klines = client.get_kline(
                category="spot",
                symbol=coin,
                interval=60,
                limit=24
            )
        else:
            klines = client.get_kline(
                category="linear",
                symbol=coin,
                interval=60,
                limit=24
            )
        if not klines['result'] or 'list' not in klines['result'] or not klines['result']['list']:
            raise ValueError(f"No 24h OHLCV data available for {coin}")
        df = pd.DataFrame(klines['result']['list'], columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce').astype('int64')
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.sort_values('open_time')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        daily_move_pct = ((df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0]) * 100 if df['open'].iloc[0] != 0 else 0.0
        logger.info(f"24h data for {coin}: Open (first) = {df['open'].iloc[0]}, Close (last) = {df['close'].iloc[-1]}, Daily move: {daily_move_pct:.2f}%")
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.error(f"Failed to fetch 24h OHLCV data for {coin}: {str(e)}")
        return pd.DataFrame()

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

def detect_support_resistance(df, current_price, entry_price, window=3, lookback=1000, tol=0.005):
    if df.empty:
        logger.warning("Empty DataFrame in detect_support_resistance, returning ATR-based levels")
        atr = compute_atr(df) if not df.empty else 0.001 * current_price
        return [current_price - atr], [current_price + atr]
    
    df_close = df['close']
    df_volume = df['volume']
    avg_volume = df_volume.rolling(20).mean().iloc[-1] if len(df_volume) >= 20 else df_volume.iloc[-1]
    lows = df['low'].rolling(window, center=True).min()
    highs = df['high'].rolling(window, center=True).max()
    candidate_supports = df['low'][(df['low'] == lows) & (df['volume'] > avg_volume * 1.0)].tail(lookback).unique().tolist()
    candidate_resistances = df['high'][(df['high'] == highs) & (df['volume'] > avg_volume * 1.0)].tail(lookback).unique().tolist()

    def count_touches(level, series, tol):
        return sum(abs(series - level) < level * tol)

    support_counts = {s: count_touches(s, df_close.tail(lookback), tol) for s in candidate_supports}
    resistance_counts = {r: count_touches(r, df_close.tail(lookback), tol) for r in candidate_resistances}

    supports = [s for s in candidate_supports if support_counts[s] >= 1]
    resistances = [r for r in candidate_resistances if resistance_counts[r] >= 1]

    pivot_supports, pivot_resistances, _ = compute_pivot_points(df)
    supports.extend(pivot_supports)
    resistances.extend(pivot_resistances)

    supports = sorted(list(set(supports)))
    resistances = sorted(list(set(resistances)))

    if not supports:
        atr = compute_atr(df) if not df.empty else 0.001 * current_price
        supports = [current_price - atr]
        logger.warning(f"No supports detected, using ATR-based support: {supports}")
    if not resistances:
        atr = compute_atr(df) if not df.empty else 0.001 * current_price
        resistances = [current_price + atr]
        logger.warning(f"No resistances detected, using ATR-based resistance: {resistances}")

    nearby_supports = sorted(supports, key=lambda s: abs(s - current_price))[:3]
    nearby_supports = sorted(nearby_supports, reverse=True)
    nearby_resistances = sorted(resistances, key=lambda r: abs(r - current_price))[:3]
    nearby_resistances = sorted(nearby_resistances)

    logger.info(f"Supports: {nearby_supports}, Resistances: {nearby_resistances}")
    return nearby_supports, nearby_resistances

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

    if body / total_range < 0.05 and last_volume > avg_volume * 1.5:
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

    logger.info(f"Detected candlestick patterns: {patterns}")
    return patterns

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
        if pl_pct > 0.5:
            comment = "Profit above avg move."
        elif pl_pct < -0.5:
            comment = "Loss above avg move."
        else:
            comment = "Standard profit/loss."
        return f"{pl:+.4f} ({pl_pct:.2f}%)", comment

    @staticmethod
    def determine_dominant_trend(df_ohlcv, df_24h, detected_candles, detected_charts):
        if df_ohlcv.empty:
            logger.warning("OHLCV DataFrame is empty, returning neutral trend")
            return "neutral", 50, 50, "No OHLCV data available."

        ema_length = 20
        ma_length = 50
        rsi_length = 14
        macd_fast = 12
        macd_slow = 26
        macd_signal = 9

        close = df_ohlcv['close']
        ema20 = compute_ema(close, ema_length).iloc[-1] if len(close) >= ema_length else close.iloc[-1]
        ma50 = compute_ma(close, ma_length).iloc[-1] if len(close) >= ma_length else close.iloc[-1]
        rsi_val = compute_rsi(close, rsi_length)
        macd, signal, hist = compute_macd(close, macd_fast, macd_slow, macd_signal)

        if not df_24h.empty:
            open_24h = df_24h['open'].iloc[0]
            close_24h = df_24h['close'].iloc[-1]
            daily_move_pct = ((close_24h - open_24h) / open_24h) * 100 if open_24h != 0 else 0.0
        else:
            prev_close = close.iloc[-2] if len(close) > 1 else close.iloc[-1]
            ma50_prev = compute_ma(close, ma_length).iloc[-2] if len(close) >= ma_length else prev_close
            daily_move_pct = ((close.iloc[-1] - ma50_prev) / ma50_prev) * 100 if ma50_prev != 0 else 0.0

        logger.info(f"Daily move %: {daily_move_pct:.2f}% (Open: {open_24h if not df_24h.empty else 'N/A'}, Close: {close_24h if not df_24h.empty else close.iloc[-1]})")

        trend = "sideways"
        if daily_move_pct > 2 and ema20 > ma50:
            trend = "bullish"
        elif daily_move_pct < -2 and ema20 < ma50:
            trend = "bearish"
        elif ema20 < ma50 and daily_move_pct > 0:
            trend = "possible reversal"
        elif ema20 > ma50 and daily_move_pct < 0:
            trend = "possible reversal"

        last_open = df_ohlcv['open'].iloc[-1]
        last_close = close.iloc[-1]
        last_high = df_ohlcv['high'].iloc[-1]
        last_low = df_ohlcv['low'].iloc[-1]
        bullish_candle = last_close > last_open and (last_close - last_open) / (last_high - last_low) > 0.6 if last_high != last_low else False
        bearish_candle = last_close < last_open and (last_open - last_close) / (last_high - last_low) > 0.6 if last_high != last_low else False

        bullish_score = 0.0
        bearish_score = 0.0

        if trend == "bullish":
            bullish_score += 6 if abs(daily_move_pct) > 10 else 4
        elif trend == "bearish":
            bearish_score += 6 if abs(daily_move_pct) > 10 else 4
        elif trend == "possible reversal":
            if daily_move_pct > 0:
                bullish_score += 3
            else:
                bearish_score += 3

        if rsi_val < 30:
            bullish_score += 2
        elif rsi_val > 70:
            bearish_score += 2
        elif 50 < rsi_val < 70:
            bullish_score += 1

        if macd > signal and hist > 0:
            bullish_score += 1.5
        elif macd < signal and hist < 0 and abs(hist) > 0.001:
            bearish_score += 1.5

        bullish_candle_count = sum(1 for p in detected_candles if p in AnalysisEngine.bullish_candles)
        bearish_candle_count = sum(1 for p in detected_candles if p in AnalysisEngine.bearish_candles)
        bullish_score += bullish_candle_count * 0.5
        bearish_score += bearish_candle_count * 0.5

        bullish_chart_count = sum(1 for p in detected_charts if p in AnalysisEngine.bullish_charts)
        bearish_chart_count = sum(1 for p in detected_charts if p in AnalysisEngine.bearish_charts)
        bullish_score += bullish_chart_count * 1
        bearish_score += bearish_chart_count * 1

        bullish_score = min(bullish_score, 10)
        bearish_score = min(bearish_score, 10)

        total_score = max(bullish_score + bearish_score, 1)
        long_conf = int((bullish_score / total_score) * 100)
        short_conf = int((bearish_score / total_score) * 100)

        dominant_bias = "long" if bullish_score > bearish_score else "short" if bearish_score > bullish_score else "neutral"

        pattern_comment = f"Trend: {trend}. Bullish score: {bullish_score:.2f}, Bearish score: {bearish_score:.2f}. RSI: {rsi_val:.2f}, MACD: {macd:.4f}, Signal: {signal:.4f}, Histogram: {hist:.4f}. 24h move: {daily_move_pct:.2f}%."

        logger.info(f"Dominant trend: {dominant_bias}, Long conf: {long_conf}%, Short conf: {short_conf}%")
        return dominant_bias, long_conf, short_conf, pattern_comment

    @staticmethod
    def market_trend(df_ohlcv, df_24h):
        if df_ohlcv.empty:
            return "sideways", 50, "No OHLCV data available."
        close = df_ohlcv['close']
        ema20 = compute_ema(close, 20).iloc[-1] if len(close) >= 20 else close.iloc[-1]
        ma50 = compute_ma(close, 50).iloc[-1] if len(close) >= 50 else close.iloc[-1]
        last_close = close.iloc[-1]

        if not df_24h.empty:
            open_24h = df_24h['open'].iloc[0]
            close_24h = df_24h['close'].iloc[-1]
            daily_move_pct = ((close_24h - open_24h) / open_24h) * 100 if open_24h != 0 else 0.0
        else:
            prev_close = close.iloc[-2] if len(close) > 1 else last_close
            daily_move_pct = ((last_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0

        trend = "sideways"
        confidence = 60
        comment = "Market moving sideways."

        if daily_move_pct > 2 and ema20 > ma50:
            trend = "bullish"
            confidence = 90 if abs(daily_move_pct) > 10 else 80
            comment = "EMA and 24h move bullish, momentum strong."
        elif daily_move_pct < -2 and ema20 < ma50:
            trend = "bearish"
            confidence = 90 if abs(daily_move_pct) > 10 else 80
            comment = "EMA and 24h move bearish, momentum strong."
        else:
            if ema20 < ma50 and daily_move_pct > 0:
                trend = "possible reversal"
                confidence = 70
                comment = "Indicators bearish but price moves bullish - possible reversal."
            if ema20 > ma50 and daily_move_pct < 0:
                trend = "possible reversal"
                confidence = 70
                comment = "Indicators bullish but price drops - possible reversal."

        return trend, confidence, comment

    @staticmethod
    def adjust_based_on_patterns(dominant_bias, detected_candles, detected_charts, trend_conf, df_ohlcv, atr):
        multiplier = 1.0
        extra_conf = 0
        pattern_comment = ""
        is_bullish_pattern = any(p in AnalysisEngine.bullish_candles for p in detected_candles) or any(p in AnalysisEngine.bullish_charts for p in detected_charts)
        is_bearish_pattern = any(p in AnalysisEngine.bearish_candles for p in detected_candles) or any(p in AnalysisEngine.bearish_charts for p in detected_charts)

        rsi = compute_rsi(df_ohlcv['close']) if not df_ohlcv.empty else 50.0
        if rsi < 30:
            is_bullish_pattern = True
            pattern_comment += " RSI oversold."
        elif rsi > 70:
            is_bearish_pattern = True
            pattern_comment += " RSI overbought."

        macd, signal, hist = compute_macd(df_ohlcv['close']) if not df_ohlcv.empty else (0.0, 0.0, 0.0)
        if macd > signal and hist > 0:
            is_bullish_pattern = True
            pattern_comment += " MACD bullish crossover."
        elif macd < signal and hist < 0 and abs(hist) > 0.001:
            is_bearish_pattern = True
            pattern_comment += " MACD bearish crossover."

        if dominant_bias == "long":
            if is_bullish_pattern:
                multiplier = 2.0
                extra_conf = 15
                pattern_comment += " Bullish patterns detected, extending target range and increasing confidence."
            elif is_bearish_pattern:
                multiplier = 0.75
                extra_conf = -5
                pattern_comment += " Bearish patterns detected, tightening range and decreasing confidence."
            else:
                multiplier = 1.5
                extra_conf = 10
                pattern_comment += " No strong patterns, moderately extending range for bullish bias."
        elif dominant_bias == "short":
            if is_bearish_pattern:
                multiplier = 2.0
                extra_conf = 15
                pattern_comment += " Bearish patterns detected, extending target range and increasing confidence."
            elif is_bullish_pattern:
                multiplier = 0.75
                extra_conf = -5
                pattern_comment += " Bullish patterns detected, tightening range and decreasing confidence."
            else:
                multiplier = 1.5
                extra_conf = 10
                pattern_comment += " No strong patterns, moderately extending range for bearish bias."
        else:
            multiplier = 1.0
            extra_conf = 0
            pattern_comment += " Neutral trend; using standard ranges."

        logger.info(f"Pattern adjustment: Multiplier = {multiplier}, Extra conf = {extra_conf}, Comment = {pattern_comment}")
        return multiplier, extra_conf, pattern_comment

    @staticmethod
    def target_stoploss(dominant_bias, supports, resistances, entry_price, pattern_targets, pattern_sls, atr, current_price, timeframe, max_diff=100, df_ohlcv=None):
        targets = []
        stoplosses = []
        trend_strength = abs((compute_ema(df_ohlcv['close'], 20).iloc[-1] - compute_ma(df_ohlcv['close'], 50).iloc[-1]) / compute_ma(df_ohlcv['close'], 50).iloc[-1]) if df_ohlcv is not None and not df_ohlcv.empty and compute_ma(df_ohlcv['close'], 50).iloc[-1] != 0 else 0.0
        adjustment = 1.0 + trend_strength * 2
        _, timeframe_multiplier, _ = get_interval_mapping(timeframe)
        atr_buffer = atr * 0.5
        max_diff = max(atr * 10, max_diff)

        logger.info(f"Calculating targets/stoplosses: ATR = {atr:.6f}, Trend strength = {trend_strength:.2f}, Timeframe multiplier = {timeframe_multiplier}, Max diff = {max_diff:.6f}")

        if dominant_bias == "long":
            potential_tgts = sorted([r for r in resistances if r > current_price])
            potential_sls = sorted([s for s in supports if s < current_price], reverse=True)
            if potential_tgts:
                tgt1 = potential_tgts[0] + atr_buffer
                targets.append(tgt1)
                for t in potential_tgts[1:]:
                    if t - tgt1 <= max_diff * timeframe_multiplier * adjustment:
                        targets.append(t + atr_buffer)
                    if len(targets) >= 2:
                        break
                logger.info(f"Long bias: Using resistance-based targets {targets}")
            else:
                tgt1 = current_price + atr * 3.5 * timeframe_multiplier * adjustment
                targets.append(tgt1)
                tgt2 = current_price + atr * 6.0 * timeframe_multiplier * adjustment
                targets.append(tgt2)
                logger.info(f"Long bias: No resistances, using ATR-based targets {targets}")
            if potential_sls:
                sl1 = potential_sls[0] - atr_buffer
                stoplosses.append(sl1)
                for s in potential_sls[1:]:
                    if sl1 - s <= max_diff * timeframe_multiplier * adjustment:
                        stoplosses.append(s - atr_buffer)
                    if len(stoplosses) >= 2:
                        break
                logger.info(f"Long bias: Using support-based stoplosses {stoplosses}")
            else:
                sl1 = current_price - atr * 2.0 * timeframe_multiplier * adjustment
                stoplosses.append(sl1)
                logger.info(f"Long bias: No supports, using ATR-based stoploss {stoplosses}")
            for pat, ptgt in pattern_targets.items():
                if pat in AnalysisEngine.bullish_charts and ptgt > current_price and (not targets or abs(ptgt - targets[-1]) > atr / 2):
                    if len(targets) < 3:
                        targets.append(ptgt)
            for pat, psl in pattern_sls.items():
                if pat in AnalysisEngine.bullish_charts and psl < current_price and (not stoplosses or abs(psl - stoplosses[-1]) > atr / 2):
                    if len(stoplosses) < 3:
                        stoplosses.append(psl)
        elif dominant_bias == "short":
            potential_tgts = sorted([s for s in supports if s < current_price], reverse=True)
            potential_sls = sorted([r for r in resistances if r > current_price])
            if potential_tgts:
                tgt1 = potential_tgts[0] - atr_buffer
                targets.append(tgt1)
                for t in potential_tgts[1:]:
                    if tgt1 - t <= max_diff * timeframe_multiplier * adjustment:
                        targets.append(t - atr_buffer)
                    if len(targets) >= 2:
                        break
                logger.info(f"Short bias: Using support-based targets {targets}")
            else:
                tgt1 = current_price - atr * 3.5 * timeframe_multiplier * adjustment
                targets.append(tgt1)
                tgt2 = current_price - atr * 6.0 * timeframe_multiplier * adjustment
                targets.append(tgt2)
                logger.info(f"Short bias: No supports, using ATR-based targets {targets}")
            if potential_sls:
                sl1 = potential_sls[0] + atr_buffer
                stoplosses.append(sl1)
                for s in potential_sls[1:]:
                    if s - sl1 <= max_diff * timeframe_multiplier * adjustment:
                        stoplosses.append(s + atr_buffer)
                    if len(stoplosses) >= 2:
                        break
                logger.info(f"Short bias: Using resistance-based stoplosses {stoplosses}")
            else:
                sl1 = current_price + atr * 2.0 * timeframe_multiplier * adjustment
                stoplosses.append(sl1)
                logger.info(f"Short bias: No resistances, using ATR-based stoploss {stoplosses}")
            for pat, ptgt in pattern_targets.items():
                if pat in AnalysisEngine.bearish_charts and ptgt < current_price and (not targets or abs(ptgt - targets[-1]) > atr / 2):
                    if len(targets) < 3:
                        targets.append(ptgt)
            for pat, psl in pattern_sls.items():
                if pat in AnalysisEngine.bearish_charts and psl > current_price and (not stoplosses or abs(psl - stoplosses[-1]) > atr / 2):
                    if len(stoplosses) < 3:
                        stoplosses.append(psl)
        else:
            atr = compute_atr(df_ohlcv) if df_ohlcv is not None and not df_ohlcv.empty else 0.001 * current_price
            nearest_support = min(supports, key=lambda s: abs(s - current_price)) if supports else current_price - atr
            nearest_resistance = min(resistances, key=lambda r: abs(r - current_price)) if resistances else current_price + atr
            targets = [nearest_resistance + atr_buffer if nearest_resistance > current_price else nearest_support - atr_buffer]
            stoplosses = [nearest_support - atr_buffer if nearest_support < current_price else nearest_resistance + atr_buffer]
            logger.info(f"Neutral bias: Targets = {targets}, Stoplosses = {stoplosses}")

        # Ensure minimum R:R ratio of 1.5 for favorable bias
        if dominant_bias in ["long", "short"]:
            risk = abs(entry_price - stoplosses[0]) if stoplosses else atr
            for i, tgt in enumerate(targets):
                reward = abs(tgt - entry_price)
                rr = reward / risk if risk > 0 else 0
                if rr < 1.5:
                    if dominant_bias == "long":
                        targets[i] = entry_price + risk * 1.5
                    else:
                        targets[i] = entry_price - risk * 1.5
                    logger.info(f"Adjusted target {i+1} to {targets[i]} for minimum R:R of 1.5")

        logger.info(f"Final Targets: {targets}, Stoplosses: {stoplosses}")
        return sorted(set(targets)), sorted(set(stoplosses))

    @staticmethod
    def calculate_user_sl(dominant_bias, entry_price, supports, resistances, atr, risk_pct):
        tol = atr * 0.5
        if dominant_bias == "long":
            user_sl_raw = entry_price * (1 - risk_pct)
            close_levels = [s for s in supports if abs(s - user_sl_raw) < tol and s < entry_price]
            if close_levels:
                return max(close_levels)
            return user_sl_raw
        elif dominant_bias == "short":
            user_sl_raw = entry_price * (1 + risk_pct)
            close_levels = [r for r in resistances if abs(r - user_sl_raw) < tol and r > entry_price]
            if close_levels:
                return min(close_levels)
            return user_sl_raw
        else:
            return entry_price * (1 - 0.01)

    @staticmethod
    def calculate_rr_ratios(dominant_bias, entry_price, targets, user_sl):
        rr_ratios = []
        risk = abs(entry_price - user_sl) if user_sl != 0 else 1e-6
        for tgt in targets:
            reward = abs(tgt - entry_price)
            rr = reward / risk if risk > 0 else 0
            rr_ratios.append(round(rr, 2))
        return rr_ratios

# Main Pipeline
def advisory_pipeline(client, input_json):
    try:
        data = InputValidator.validate_and_normalize(input_json)
        # Force timeframe to 4h for all analyses
        data["timeframe"] = "4h"
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
        df_ohlcv = get_ohlcv_df(client, data["coin"], data["timeframe"], data["market"])
        df_24h = get_24h_trend_df(client, data["coin"], data["market"])

        atr = compute_atr(df_ohlcv)
        detected_candles = detect_candlestick_patterns(df_ohlcv)
        detected_charts, pattern_targets, pattern_sls = detect_chart_patterns(df_ohlcv)

        profit_loss, profit_comment = AnalysisEngine.profitability(data["position_type"], data["entry_price"], current_price, data["quantity"])
        trend, trend_conf, trend_comment = AnalysisEngine.market_trend(df_ohlcv, df_24h)
        dominant_bias, long_conf, short_conf, trend_pattern_comment = AnalysisEngine.determine_dominant_trend(df_ohlcv, df_24h, detected_candles, detected_charts)

        if data["has_both_positions"]:
            analysis_bias = dominant_bias
            warning = f"Since you have both positions, analyzing based on dominant trend ({dominant_bias}). Consider closing the opposing position."
        else:
            analysis_bias = data["position_type"]
            warning = ""
            if dominant_bias != "neutral" and dominant_bias != data["position_type"]:
                warning = f"Your {data['position_type']} position opposes the {dominant_bias} trend—consider exiting to align with market momentum."

        if dominant_bias == "long":
            detected_charts = [p for p in detected_charts if p in AnalysisEngine.bullish_charts]
        elif dominant_bias == "short":
            detected_charts = [p for p in detected_charts if p in AnalysisEngine.bearish_charts]

        multiplier, extra_conf, pattern_comment = AnalysisEngine.adjust_based_on_patterns(analysis_bias, detected_candles, detected_charts, trend_conf, df_ohlcv, atr)
        tol = atr * 0.5
        supports, resistances = detect_support_resistance(df_ohlcv, current_price, data["entry_price"], tol=tol)
        max_diff = atr * 5
        targets, stoplosses = AnalysisEngine.target_stoploss(
            analysis_bias, supports, resistances, data["entry_price"], pattern_targets, pattern_sls, atr, current_price, data["timeframe"], max_diff=max_diff, df_ohlcv=df_ohlcv
        )
        user_sl = AnalysisEngine.calculate_user_sl(analysis_bias, data["entry_price"], supports, resistances, atr, data["risk_pct"])
        rr_ratios = AnalysisEngine.calculate_rr_ratios(analysis_bias, data["entry_price"], targets, user_sl)

        recommended_action = "Go long" if dominant_bias == "long" else "Go short" if dominant_bias == "short" else "Stay neutral/avoid new positions"

        output = {
            "coin": data["coin"],
            "market": data["market"],
            "position_type": data["position_type"],
            "entry_price": data["entry_price"],
            "current_price": current_price,
            "profit_loss": profit_loss,
            "profitability_comment": profit_comment,
            "market_trend": trend,
            "dominant_bias": dominant_bias,
            "confidence_meter": {"long": long_conf, "short": short_conf},
            "trend_comment": trend_comment + " " + trend_pattern_comment + " " + pattern_comment,
            "support_levels": supports,
            "resistance_levels": resistances,
            "detected_patterns": {
                "candlesticks": detected_candles,
                "chart_patterns": detected_charts
            },
            "targets": targets,
            "rr_ratios": rr_ratios,
            "market_stoplosses": stoplosses,
            "user_stoploss": user_sl,
            "recommended_action": recommended_action,
            "warning": warning
        }
        if data.get("name"):
            output["user_name"] = data["name"]
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
    has_both_positions: Optional[bool] = False
    risk_pct: Optional[float] = 0.02
    code: Optional[str] = None
    name: Optional[str] = None

# Payment Input Model
class PaymentInput(BaseModel):
    wallet_address: str
    amount_usd: float

@app.get("/")
async def root():
    return {"message": "Welcome to the Trading Analysis API. Use /analyze with POST to analyze positions."}

@app.options("/analyze")
async def options_analyze(request: Request):
    return JSONResponse(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "https://position-analyzer-app.vercel.app",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )

@app.post("/analyze")
async def analyze_position(input_data: TradeInput, request: Request):
    ip = request.client.host
    today = date.today()

    # Check and reset daily usage if new day
    if daily_usage[ip]["last_date"] != today:
        daily_usage[ip] = {"count": 0, "last_date": today}
        daily_codes_used[today].clear()

    # Check if free analyses are available
    if daily_usage[ip]["count"] < 2:
        daily_usage[ip]["count"] += 1
    else:
        code = input_data.code
        if not code:
            logger.info(f"IP {ip} exceeded 2 free analyses, code required")
            raise HTTPException(status_code=403, detail="Require login code: 2 free analyses per day exceeded")
        if code not in LOGIN_CODES and code not in valid_codes:
            logger.info(f"Invalid code {code} from IP {ip}")
            raise HTTPException(status_code=403, detail="Invalid login code")
        if code in valid_codes:
            code_info = valid_codes[code]
            current_time_ms = int(datetime.now().timestamp() * 1000)
            if code_info["expiry"] < current_time_ms:
                logger.info(f"Expired code {code} from IP {ip}")
                raise HTTPException(status_code=403, detail="Login code expired")
            last_used = datetime.fromisoformat(code_info["last_used"]).date() if code_info["last_used"] else None
            if last_used == today:
                logger.info(f"Code {code} already used today by IP {ip}")
                raise HTTPException(status_code=403, detail="Code already used today")
            valid_codes[code]["last_used"] = datetime.now().isoformat()

    try:
        client = initialize_client()
        data = input_data.dict(exclude={"code", "name"})  # Exclude code and name from analysis data
        validated_data = InputValidator.validate_and_normalize(data)
        output = advisory_pipeline(client, validated_data)
        return output
    except ValueError as ve:
        logger.error(f"ValueError in /analyze: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error in /analyze: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/verify-payment")
async def verify_payment(payment: PaymentInput):
    try:
        # Placeholder: Simulate payment verification
        if not payment.wallet_address.startswith("bc1"):
            raise HTTPException(status_code=400, detail="Invalid Bitcoin wallet address")
        if payment.amount_usd != 10.0:
            raise HTTPException(status_code=400, detail="Payment must be $10")

        # Generate unique code
        code = str(uuid.uuid4())
        expiry = int((datetime.now() + timedelta(days=30)).timestamp() * 1000)
        valid_codes[code] = {"expiry": expiry, "last_used": None}
        logger.info(f"Generated code {code} for wallet {payment.wallet_address}")

        return {"code": code, "expiry": expiry}
    except Exception as e:
        logger.error(f"Error in /verify-payment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Payment verification failed: {str(e)}")