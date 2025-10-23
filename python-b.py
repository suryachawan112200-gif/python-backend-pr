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

# Login codes (hardcoded for testing)
LOGIN_CODES = ["AIVISOR1", "AIVISOR2", "AIVISOR3", "AIVISOR4", "AIVISOR5", "AIVISOR6"]

# In-memory storage for usage tracking and paid codes
daily_usage = defaultdict(lambda: {"count": 0, "last_date": None})
daily_codes_used = defaultdict(set)
valid_codes = {}  # {code: {"expiry": timestamp_ms, "last_used": date_str}}

# Feature flag for multi-timeframe analysis
MULTI_TIMEFRAME_ENABLED = os.environ.get("MULTI_TIMEFRAME_ENABLED", "true").lower() == "true"

# Input & Validation
class InputValidator:
    REQUIRED_FIELDS = {"coin", "market", "entry_price", "quantity", "position_type"}
    VALID_TIMEFRAMES = ["1h", "4h", "1d", "5m", "15m", "1month"]

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
        data["timeframes"] = data.get("timeframes", ["1h", "4h", "1d"] if MULTI_TIMEFRAME_ENABLED else ["4h"])
        for tf in data["timeframes"]:
            if tf not in InputValidator.VALID_TIMEFRAMES:
                raise ValueError(f"Invalid timeframe {tf}. Must be one of: {', '.join(InputValidator.VALID_TIMEFRAMES)}")
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
        "1h": 60,
        "4h": 240,
        "1d": "D",
        "1month": "M"
    }
    multiplier_adjust = {
        "5m": 1.5,
        "15m": 2.0,
        "1h": 2.2,
        "4h": 2.5,
        "1d": 3.0,
        "1month": 5.0
    }
    lookback_periods = {
        "5m": 25920,
        "15m": 8640,
        "1h": 2160,
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
            raise ValueError(f"No OHLCV data available for {coin} on {timeframe}")
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
        logger.error(f"Failed to fetch OHLCV data for {coin} on {timeframe}: {str(e)}")
        raise ValueError(f"Failed to fetch OHLCV data for {coin} on {timeframe}: {str(e)}")

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

def compute_adx(df, period=14):
    if len(df) < period:
        return 0.0
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
    adx = dx.rolling(period).mean()
    return adx.iloc[-1] if not adx.empty else 0.0

def compute_vwap(df):
    if df.empty:
        return 0.0
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap.iloc[-1] if not vwap.empty else 0.0

def compute_indicators_for_df(df):
    if df.empty:
        return {
            "ema20": 0.0,
            "ma50": 0.0,
            "atr": 0.0,
            "rsi": 50.0,
            "macd": (0.0, 0.0, 0.0),
            "adx": 0.0,
            "vwap": 0.0
        }
    close = df['close']
    ema20 = compute_ema(close, 20).iloc[-1] if len(close) >= 20 else close.iloc[-1]
    ma50 = compute_ma(close, 50).iloc[-1] if len(close) >= 50 else close.iloc[-1]
    atr = compute_atr(df)
    rsi = compute_rsi(close)
    macd, signal, hist = compute_macd(close)
    adx = compute_adx(df)
    vwap = compute_vwap(df) if 'volume' in df else 0.0
    return {
        "ema20": ema20,
        "ma50": ma50,
        "atr": atr,
        "rsi": rsi,
        "macd": (macd, signal, hist),
        "adx": adx,
        "vwap": vwap
    }

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

def detect_supports_resistances_for_df(df, current_price, lookback_months):
    if df.empty:
        logger.warning("Empty DataFrame in detect_supports_resistances_for_df, returning ATR-based levels")
        atr = compute_atr(df) if not df.empty else 0.001 * current_price
        return [current_price - atr], [current_price + atr], {}, {}
    
    df_close = df['close']
    df_volume = df['volume']
    avg_volume = df_volume.rolling(20).mean().iloc[-1] if len(df_volume) >= 20 else df_volume.iloc[-1]
    window = 3
    lookback = int(lookback_months * 30 * 24 / (1 if df.index[-1].hour == 0 else 4))  # Approximate bars
    lows = df['low'].rolling(window, center=True).min()
    highs = df['high'].rolling(window, center=True).max()
    candidate_supports = df['low'][(df['low'] == lows) & (df['volume'] > avg_volume * 1.0)].tail(lookback).unique().tolist()
    candidate_resistances = df['high'][(df['high'] == highs) & (df['volume'] > avg_volume * 1.0)].tail(lookback).unique().tolist()

    def count_touches(level, series, tol):
        return sum(abs(series - level) < level * tol)

    tol = compute_atr(df) * 0.5 if not df.empty else 0.005 * current_price
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

    support_weights = {s: support_counts.get(s, 0) * (df_volume[df['low'] == s].iloc[-1] / avg_volume if s in df['low'].values else 1.0) for s in nearby_supports}
    resistance_weights = {r: resistance_counts.get(r, 0) * (df_volume[df['high'] == r].iloc[-1] / avg_volume if r in df['high'].values else 1.0) for r in nearby_resistances}

    logger.info(f"Supports: {nearby_supports}, Resistances: {nearby_resistances}")
    return nearby_supports, nearby_resistances, support_weights, resistance_weights

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
    def analyze_single_timeframe(df, current_price, entry_price, timeframe, quantity, extra_params):
        if df.empty:
            logger.warning(f"Empty DataFrame for {timeframe}, returning default analysis")
            return {
                "bias": "neutral",
                "confidence": 50,
                "targets": [],
                "stoplosses": [],
                "patterns": {"candlesticks": [], "chart_patterns": []},
                "supports": [],
                "resistances": [],
                "indicators": {
                    "ema20": 0.0,
                    "ma50": 0.0,
                    "atr": 0.0,
                    "rsi": 50.0,
                    "macd": (0.0, 0.0, 0.0),
                    "adx": 0.0,
                    "vwap": 0.0
                },
                "comment": f"No data available for {timeframe}"
            }

        indicators = compute_indicators_for_df(df)
        ema20, ma50, atr, rsi, (macd, signal, hist), adx, vwap = (
            indicators["ema20"],
            indicators["ma50"],
            indicators["atr"],
            indicators["rsi"],
            indicators["macd"],
            indicators["adx"],
            indicators["vwap"]
        )

        lookback_months = {"1h": 0.5, "4h": 1.5, "1d": 6.0}.get(timeframe, 1.5)
        supports, resistances, support_weights, resistance_weights = detect_supports_resistances_for_df(df, current_price, lookback_months)
        candlestick_patterns = detect_candlestick_patterns(df)
        chart_patterns, pattern_targets, pattern_sls = detect_chart_patterns(df)

        # Compute confidence and bias
        trend_score = 0.0
        if ema20 > ma50:
            trend_score += 0.4
        elif ema20 < ma50:
            trend_score -= 0.4
        adx_normalized = min(adx / 50, 1.0)  # Normalize ADX to 0-1
        trend_score += 0.3 * adx_normalized if adx > 25 else 0.0
        if macd > signal and hist > 0:
            trend_score += 0.3
        elif macd < signal and hist < 0:
            trend_score -= 0.3

        momentum_score = abs(rsi - 50) / 50  # Distance from neutral
        if abs(hist) > atr * 0.5:
            momentum_score += 0.3
        momentum_score = min(momentum_score, 1.0)

        avg_volume = df['volume'].rolling(20).mean().iloc[-1] if len(df['volume']) >= 20 else df['volume'].iloc[-1]
        volume_score = min(df['volume'].iloc[-1] / avg_volume, 2.0) if avg_volume > 0 else 0.0

        pattern_score = 0.0
        strong_patterns = ["Head and Shoulders", "Inverse Head and Shoulders", "Double Top", "Double Bottom", "Triple Top", "Triple Bottom"]
        for p in chart_patterns:
            pattern_score += 0.6 if p in strong_patterns else 0.4
        for p in candlestick_patterns:
            pattern_score += 0.3 if p in (AnalysisEngine.bullish_candles + AnalysisEngine.bearish_candles) else 0.2
        pattern_score = min(pattern_score, 1.0)

        raw_score = 0.4 * trend_score + 0.25 * momentum_score + 0.2 * volume_score + 0.15 * pattern_score
        confidence = round(abs(raw_score) * 100)
        bias = "neutral"
        if trend_score > 0.55 and momentum_score > 0.5:
            bias = "long" if trend_score > 0 else "short"
        elif trend_score < -0.55 and momentum_score > 0.5:
            bias = "short"

        # Compute targets and stoplosses
        _, timeframe_multiplier, _ = get_interval_mapping(timeframe)
        trend_strength = abs((ema20 - ma50) / ma50) if ma50 != 0 else 0.0
        adjustment = 1.0 + trend_strength * 2
        atr_buffer = atr * 0.5
        max_diff = atr * 5

        if bias == "long":
            potential_tgts = sorted([r for r in resistances if r > current_price])
            potential_sls = sorted([s for s in supports if s < current_price], reverse=True)
            targets = []
            stoplosses = []
            if potential_tgts:
                tgt1 = potential_tgts[0] + atr_buffer
                targets.append({"price": tgt1, "confirmed_by": [timeframe]})
                for t in potential_tgts[1:]:
                    if t - tgt1 <= max_diff * timeframe_multiplier * adjustment:
                        targets.append({"price": t + atr_buffer, "confirmed_by": [timeframe]})
                    if len(targets) >= 2:
                        break
            else:
                tgt1 = current_price + atr * 3.5 * timeframe_multiplier * adjustment
                targets.append({"price": tgt1, "confirmed_by": [timeframe]})
                tgt2 = current_price + atr * 6.0 * timeframe_multiplier * adjustment
                targets.append({"price": tgt2, "confirmed_by": [timeframe]})
            if potential_sls:
                sl1 = potential_sls[0] - atr_buffer
                stoplosses.append({"price": sl1, "confirmed_by": [timeframe]})
                for s in potential_sls[1:]:
                    if sl1 - s <= max_diff * timeframe_multiplier * adjustment:
                        stoplosses.append({"price": s - atr_buffer, "confirmed_by": [timeframe]})
                    if len(stoplosses) >= 2:
                        break
            else:
                sl1 = current_price - atr * 2.0 * timeframe_multiplier * adjustment
                stoplosses.append({"price": sl1, "confirmed_by": [timeframe]})
            for pat, ptgt in pattern_targets.items():
                if pat in AnalysisEngine.bullish_charts and ptgt > current_price:
                    targets.append({"price": ptgt, "confirmed_by": [timeframe]})
            for pat, psl in pattern_sls.items():
                if pat in AnalysisEngine.bullish_charts and psl < current_price:
                    stoplosses.append({"price": psl, "confirmed_by": [timeframe]})
        elif bias == "short":
            potential_tgts = sorted([s for s in supports if s < current_price], reverse=True)
            potential_sls = sorted([r for r in resistances if r > current_price])
            targets = []
            stoplosses = []
            if potential_tgts:
                tgt1 = potential_tgts[0] - atr_buffer
                targets.append({"price": tgt1, "confirmed_by": [timeframe]})
                for t in potential_tgts[1:]:
                    if tgt1 - t <= max_diff * timeframe_multiplier * adjustment:
                        targets.append({"price": t - atr_buffer, "confirmed_by": [timeframe]})
                    if len(targets) >= 2:
                        break
            else:
                tgt1 = current_price - atr * 3.5 * timeframe_multiplier * adjustment
                targets.append({"price": tgt1, "confirmed_by": [timeframe]})
                tgt2 = current_price - atr * 6.0 * timeframe_multiplier * adjustment
                targets.append({"price": tgt2, "confirmed_by": [timeframe]})
            if potential_sls:
                sl1 = potential_sls[0] + atr_buffer
                stoplosses.append({"price": sl1, "confirmed_by": [timeframe]})
                for s in potential_sls[1:]:
                    if s - sl1 <= max_diff * timeframe_multiplier * adjustment:
                        stoplosses.append({"price": s + atr_buffer, "confirmed_by": [timeframe]})
                    if len(stoplosses) >= 2:
                        break
            else:
                sl1 = current_price + atr * 2.0 * timeframe_multiplier * adjustment
                stoplosses.append({"price": sl1, "confirmed_by": [timeframe]})
            for pat, ptgt in pattern_targets.items():
                if pat in AnalysisEngine.bearish_charts and ptgt < current_price:
                    targets.append({"price": ptgt, "confirmed_by": [timeframe]})
            for pat, psl in pattern_sls.items():
                if pat in AnalysisEngine.bearish_charts and psl > current_price:
                    stoplosses.append({"price": psl, "confirmed_by": [timeframe]})
        else:
            atr = compute_atr(df) if not df.empty else 0.001 * current_price
            nearest_support = min(supports, key=lambda s: abs(s - current_price)) if supports else current_price - atr
            nearest_resistance = min(resistances, key=lambda r: abs(r - current_price)) if resistances else current_price + atr
            targets = [{"price": nearest_resistance + atr_buffer if nearest_resistance > current_price else nearest_support - atr_buffer, "confirmed_by": [timeframe]}]
            stoplosses = [{"price": nearest_support - atr_buffer if nearest_support < current_price else nearest_resistance + atr_buffer, "confirmed_by": [timeframe]}]

        # Ensure minimum R:R ratio of 1.5
        risk = abs(entry_price - stoplosses[0]["price"]) if stoplosses else atr
        for i, tgt in enumerate(targets):
            reward = abs(tgt["price"] - entry_price)
            rr = reward / risk if risk > 0 else 0
            if rr < 1.5:
                if bias == "long":
                    targets[i]["price"] = entry_price + risk * 1.5
                elif bias == "short":
                    targets[i]["price"] = entry_price - risk * 1.5
                logger.info(f"Adjusted target {i+1} to {targets[i]['price']} for minimum R:R of 1.5")

        return {
            "bias": bias,
            "confidence": confidence,
            "targets": sorted(targets, key=lambda x: x["price"]),
            "stoplosses": sorted(stoplosses, key=lambda x: x["price"], reverse=(bias == "long")),
            "patterns": {"candlesticks": candlestick_patterns, "chart_patterns": chart_patterns},
            "supports": supports,
            "resistances": resistances,
            "indicators": indicators,
            "comment": f"Analysis for {timeframe}: Bias {bias}, Confidence {confidence}%"
        }

    @staticmethod
    def target_stoploss(dominant_bias, per_tf_results, entry_price, current_price, timeframe, df_24h):
        weights = {"1d": 0.40, "4h": 0.35, "1h": 0.25}
        primary_tf = max(per_tf_results, key=lambda tf: per_tf_results[tf]["confidence"] * weights.get(tf, 1.0))
        primary_result = per_tf_results[primary_tf]
        atr = primary_result["indicators"]["atr"]
        max_diff = atr * 5
        _, timeframe_multiplier, _ = get_interval_mapping(primary_tf)
        trend_strength = abs((primary_result["indicators"]["ema20"] - primary_result["indicators"]["ma50"]) / primary_result["indicators"]["ma50"]) if primary_result["indicators"]["ma50"] != 0 else 0.0
        adjustment = 1.0 + trend_strength * 2
        atr_buffer = atr * 0.5

        targets = []
        stoplosses = []
        for tf in per_tf_results:
            tf_result = per_tf_results[tf]
            for tgt in tf_result["targets"]:
                if tf == primary_tf or abs(tgt["price"] - entry_price) < max_diff * timeframe_multiplier * adjustment:
                    existing = next((t for t in targets if abs(t["price"] - tgt["price"]) < atr_buffer), None)
                    if existing:
                        existing["confirmed_by"].extend(tgt["confirmed_by"])
                    else:
                        targets.append(tgt)
            for sl in tf_result["stoplosses"]:
                if tf == primary_tf or abs(sl["price"] - entry_price) < max_diff * timeframe_multiplier * adjustment:
                    existing = next((s for s in stoplosses if abs(s["price"] - sl["price"]) < atr_buffer), None)
                    if existing:
                        existing["confirmed_by"].extend(sl["confirmed_by"])
                    else:
                        stoplosses.append(sl)

        if dominant_bias == "long":
            targets = sorted([t for t in targets if t["price"] > current_price], key=lambda x: x["price"])[:3]
            stoplosses = sorted([s for s in stoplosses if s["price"] < current_price], key=lambda x: x["price"], reverse=True)[:2]
        elif dominant_bias == "short":
            targets = sorted([t for t in targets if t["price"] < current_price], key=lambda x: x["price"], reverse=True)[:3]
            stoplosses = sorted([s for s in stoplosses if s["price"] > current_price], key=lambda x: x["price"])[:2]
        else:
            targets = sorted(targets, key=lambda x: abs(x["price"] - current_price))[:2]
            stoplosses = sorted(stoplosses, key=lambda x: abs(x["price"] - current_price))[:2]

        if not df_24h.empty:
            daily_move = (df_24h['close'].iloc[-1] - df_24h['open'].iloc[0]) / df_24h['open'].iloc[0] if df_24h['open'].iloc[0] != 0 else 0.0
            if abs(daily_move) > 0.05:
                for tgt in targets:
                    tgt["price"] += atr_buffer * (1 if daily_move > 0 else -1)
                for sl in stoplosses:
                    sl["price"] += atr_buffer * (-1 if daily_move > 0 else 1)

        return targets, stoplosses

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
            reward = abs(tgt["price"] - entry_price)
            rr = reward / risk if risk > 0 else 0
            rr_ratios.append(round(rr, 2))
        return rr_ratios

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
        df_24h = get_24h_trend_df(client, data["coin"], data["market"])

        per_tf_results = {}
        for tf in data["timeframes"]:
            try:
                df_ohlcv = get_ohlcv_df(client, data["coin"], tf, data["market"])
                tf_result = AnalysisEngine.analyze_single_timeframe(
                    df_ohlcv,
                    current_price,
                    data["entry_price"],
                    tf,
                    data["quantity"],
                    {"risk_pct": data["risk_pct"], "has_both_positions": data["has_both_positions"]}
                )
                per_tf_results[tf] = tf_result
            except ValueError as ve:
                logger.warning(f"Skipping timeframe {tf} due to: {str(ve)}")
                per_tf_results[tf] = {
                    "bias": "neutral",
                    "confidence": 0,
                    "targets": [],
                    "stoplosses": [],
                    "patterns": {"candlesticks": [], "chart_patterns": []},
                    "supports": [],
                    "resistances": [],
                    "indicators": {
                        "ema20": 0.0,
                        "ma50": 0.0,
                        "atr": 0.0,
                        "rsi": 50.0,
                        "macd": (0.0, 0.0, 0.0),
                        "adx": 0.0,
                        "vwap": 0.0
                    },
                    "comment": f"Failed to fetch data: {str(ve)}"
                }

        # Aggregate results
        weights = {"1d": 0.40, "4h": 0.35, "1h": 0.25}
        bias_map = {"long": 1, "neutral": 0, "short": -1}
        bias_score = 0.0
        total_weight = 0.0
        for tf, result in per_tf_results.items():
            if result["confidence"] > 0:
                weight = weights.get(tf, 1.0 / len(per_tf_results))
                bias_score += weight * bias_map[result["bias"]] * (result["confidence"] / 100)
                total_weight += weight

        if total_weight == 0:
            total_weight = 1.0
        bias_score /= total_weight
        aggregated_bias = "long" if bias_score > 0 else "short" if bias_score < 0 else "neutral"
        aggregated_confidence = min(round(abs(bias_score) * 100), 100)
        primary_tf = max(per_tf_results, key=lambda tf: per_tf_results[tf]["confidence"] * weights.get(tf, 1.0))
        conflict = any(per_tf_results[tf1]["bias"] != per_tf_results[tf2]["bias"] and per_tf_results[tf1]["confidence"] > 50 and per_tf_results[tf2]["confidence"] > 50 for tf1 in per_tf_results for tf2 in per_tf_results if tf1 < tf2)

        if conflict and aggregated_confidence > 50:
            aggregated_bias = "neutral"
            aggregated_confidence = min(aggregated_confidence, 70)
            recommendation = "Watch/avoid new positions due to conflicting timeframes"
            remarks = f"Conflict detected: {', '.join([f'{tf}: {res['bias']} ({res['confidence']}%)' for tf, res in per_tf_results.items() if res['confidence'] > 50])}"
        else:
            recommendation = f"Go {aggregated_bias}" if aggregated_bias in ["long", "short"] else "Stay neutral/avoid new positions"
            remarks = f"Primary timeframe: {primary_tf}. Confidence: {aggregated_confidence}%"

        profit_loss, profit_comment = AnalysisEngine.profitability(data["position_type"], data["entry_price"], current_price, data["quantity"])
        warning = ""
        if data["has_both_positions"]:
            warning = f"Since you have both positions, analyzing based on dominant trend ({aggregated_bias}). Consider closing the opposing position."
        elif aggregated_bias != "neutral" and aggregated_bias != data["position_type"]:
            warning = f"Your {data['position_type']} position opposes the {aggregated_bias} trend—consider exiting to align with market momentum."

        targets, stoplosses = AnalysisEngine.target_stoploss(aggregated_bias, per_tf_results, data["entry_price"], current_price, primary_tf, df_24h)
        supports = list(set(sum([res["supports"] for res in per_tf_results.values()], [])))
        resistances = list(set(sum([res["resistances"] for res in per_tf_results.values()], [])))
        user_sl = AnalysisEngine.calculate_user_sl(aggregated_bias, data["entry_price"], supports, resistances, per_tf_results[primary_tf]["indicators"]["atr"], data["risk_pct"])
        rr_ratios = AnalysisEngine.calculate_rr_ratios(aggregated_bias, data["entry_price"], targets, user_sl)

        output = {
            "coin": data["coin"],
            "market": data["market"],
            "position_type": data["position_type"],
            "entry_price": data["entry_price"],
            "current_price": current_price,
            "profit_loss": profit_loss,
            "profitability_comment": profit_comment,
            "dominant_bias": aggregated_bias,
            "confidence": aggregated_confidence,
            "primary_timeframe": primary_tf,
            "recommendation": recommendation,
            "remarks": remarks,
            "conflict": conflict,
            "support_levels": sorted(supports, reverse=True),
            "resistance_levels": sorted(resistances),
            "targets": [t["price"] for t in targets],
            "target_confirmations": {str(i): t["confirmed_by"] for i, t in enumerate(targets)},
            "market_stoplosses": [s["price"] for s in stoplosses],
            "stoploss_confirmations": {str(i): s["confirmed_by"] for i, s in enumerate(stoplosses)},
            "user_stoploss": user_sl,
            "rr_ratios": rr_ratios,
            "per_timeframe": {
                tf: {
                    "bias": res["bias"],
                    "confidence": res["confidence"],
                    "targets": [t["price"] for t in res["targets"]],
                    "stoplosses": [s["price"] for s in res["stoplosses"]],
                    "patterns": res["patterns"],
                    "supports": res["supports"],
                    "resistances": res["resistances"],
                    "indicators": res["indicators"],
                    "comment": res["comment"]
                } for tf, res in per_tf_results.items()
            }
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
    timeframes: Optional[List[str]] = None
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

# Backtest Script
def run_backtest(csv_file, coin, market, entry_price, quantity, position_type):
    df = pd.read_csv(csv_file)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    
    single_tf_input = {
        "coin": coin,
        "market": market,
        "entry_price": entry_price,
        "quantity": quantity,
        "position_type": position_type,
        "timeframes": ["4h"],
        "has_both_positions": False,
        "risk_pct": 0.02
    }
    
    multi_tf_input = single_tf_input.copy()
    multi_tf_input["timeframes"] = ["1h", "4h", "1d"]

    class MockClient:
        def get_instruments_info(self, category, symbol):
            return {"result": {"list": [{"symbol": symbol, "status": "Trading"}]}}
        def get_tickers(self, category, symbol):
            return {"result": {"list": [{"lastPrice": str(df['close'].iloc[-1])}]}}
        def get_kline(self, category, symbol, interval, limit):
            tf_map = {5: "5m", 15: "15m", 60: "1h", 240: "4h", "D": "1d", "M": "1month"}
            tf = tf_map[interval]
            resampled = df.resample(tf, on='open_time').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            return {
                "result": {
                    "list": [
                        [int(row.name.timestamp() * 1000), str(row.open), str(row.high), str(row.low), str(row.close), str(row.volume), "0"]
                        for _, row in resampled.tail(limit).iterrows()
                    ]
                }
            }
        def get_server_time(self):
            return {"result": {"time": int(time.time() * 1000)}}

    client = MockClient()
    single_tf_result = advisory_pipeline(client, single_tf_input)
    multi_tf_result = advisory_pipeline(client, multi_tf_input)

    print("Single Timeframe (4h) Result:")
    print(json.dumps(single_tf_result, indent=2))
    print("\nMulti Timeframe Result:")
    print(json.dumps(multi_tf_result, indent=2))

    hit_rate_single = sum(1 for t in single_tf_result["targets"] if (position_type == "long" and t > entry_price) or (position_type == "short" and t < entry_price)) / len(single_tf_result["targets"]) if single_tf_result["targets"] else 0
    hit_rate_multi = sum(1 for t in multi_tf_result["targets"] if (position_type == "long" and t > entry_price) or (position_type == "short" and t < entry_price)) / len(multi_tf_result["targets"]) if multi_tf_result["targets"] else 0

    print(f"\nHit Rate (Single TF): {hit_rate_single:.2%}")
    print(f"Hit Rate (Multi TF): {hit_rate_multi:.2%}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        run_backtest(
            csv_file="sample_candles.csv",
            coin="BTCUSDT",
            market="futures",
            entry_price=60000.0,
            quantity=0.1,
            position_type="long"
        )