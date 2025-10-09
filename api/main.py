import json
import random
from pybit.unified_trading import HTTP  # Bybit's official Python library
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware  # Explicit import
from fastapi.responses import JSONResponse
import time
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://position-analyzer-app.vercel.app", "http://localhost:3000"],  # Frontend domain + local for testing
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
        return data

# Bybit Data Fetch
def initialize_client():
    max_retries = 5
    for attempt in range(max_retries):
        try:
            client = HTTP(
                api_key=API_KEY,
                api_secret=API_SECRET,
                testnet=False  # Use mainnet
            )
            # Test connection with a simple request
            response = client.get_server_time()
            print(f"Server time response: {response}")  # Add logging for debugging
            return client
        except Exception as e:
            error_msg = str(e).lower()
            print(f"Failed to initialize Bybit client (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if "rate limit" in error_msg or "403" in error_msg:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit or 403 detected, waiting {wait_time} seconds before retry...")
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
        print(f"Error fetching spot symbols: {str(e)}")
    try:
        futures_info = client.get_instruments_info(category="linear")  # USDT perpetuals
        futures_symbols = [s['symbol'] for s in futures_info['result']['list'] if s['status'] == 'Trading']
    except Exception as e:
        print(f"Error fetching futures symbols: {str(e)}")
    all_coins = list(set(spot_symbols + futures_symbols))
    return sorted(all_coins)

def get_current_price(client, coin, market):
    try:
        if market == "spot":
            ticker = client.get_tickers(category="spot", symbol=coin)
            return float(ticker['result']['list'][0]['lastPrice'])
        else:
            ticker = client.get_tickers(category="linear", symbol=coin)  # USDT perpetuals
            return float(ticker['result']['list'][0]['lastPrice'])
    except Exception as e:
        raise ValueError(f"Failed to get current price for {coin}: {str(e)}")

def get_interval_mapping(timeframe):
    mapping = {
        "5m": 5,
        "15m": 15,
        "4h": 240,
        "1d": "D",
        "1month": "M"
    }
    return mapping[timeframe]

def get_ohlcv_df(client, coin, timeframe, lookback, market):
    interval = get_interval_mapping(timeframe)
    try:
        if market == "spot":
            klines = client.get_kline(
                category="spot",
                symbol=coin,
                interval=interval,
                limit=lookback  # Adjust limit as needed (max 1000)
            )
        else:
            klines = client.get_kline(
                category="linear",
                symbol=coin,
                interval=interval,
                limit=lookback
            )
        if not klines['result'] or 'list' not in klines['result']:
            raise ValueError(f"No OHLCV data available for {coin}")
        df = pd.DataFrame(klines['result']['list'], columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        raise ValueError(f"Failed to fetch OHLCV data for {coin}: {str(e)}")

def get_24h_trend_df(client, coin, market):
    # Fetch last 24 hours using 60m interval, limit=24
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
        if not klines['result'] or 'list' not in klines['result']:
            raise ValueError(f"No 24h OHLCV data available for {coin}")
        df = pd.DataFrame(klines['result']['list'], columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        raise ValueError(f"Failed to fetch 24h OHLCV data for {coin}: {str(e)}")

# Indicator & Analysis Helpers
def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_ma(series, window):
    return series.rolling(window=window).mean()

def compute_atr(df, period=14):
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.iloc[-1] if not atr.empty and len(atr) >= period else 0.0

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - 100 / (1 + rs)
    return rsi.iloc[-1] if not rsi.empty and len(rsi) >= period else 50.0

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = compute_ema(macd, signal)
    histogram = macd - signal_line
    return (macd.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]) if len(macd) >= signal else (0.0, 0.0, 0.0)

def compute_pivot_points(df, period=15):
    # Traditional pivot points based on last 'period' candles aggregated
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
    supports = [s1, s2, s3]
    resistances = [r1, r2, r3]
    return supports, resistances, pivot

def detect_support_resistance(df, current_price, entry_price, window=5, lookback=90, tol=0.001):
    df_close = df['close']
    lows = df['low'].rolling(window, center=True).min()
    highs = df['high'].rolling(window, center=True).max()
    candidate_supports = df['low'][(df['low'] == lows)].tail(lookback).unique().tolist()
    candidate_resistances = df['high'][(df['high'] == highs)].tail(lookback).unique().tolist()

    def count_touches(level, series, tol):
        return sum(abs(series - level) < tol)

    support_counts = {s: count_touches(s, df_close.tail(lookback), tol) for s in candidate_supports}
    resistance_counts = {r: count_touches(r, df_close.tail(lookback), tol) for r in candidate_resistances}

    supports = [s for s in candidate_supports if support_counts[s] >= 2]
    resistances = [r for r in candidate_resistances if resistance_counts[r] >= 2]

    # Add pivot points
    pivot_supports, pivot_resistances, _ = compute_pivot_points(df)
    supports.extend(pivot_supports)
    resistances.extend(pivot_resistances)

    # Remove duplicates and sort
    supports = sorted(list(set(supports)))
    resistances = sorted(list(set(resistances)))

    # Filter to 3 major nearby (below current/entry for supports, above for resistances)
    avg_price = (current_price + entry_price) / 2
    nearby_supports = sorted([s for s in supports if s < avg_price], reverse=True)[:3]
    nearby_resistances = sorted([r for r in resistances if r > avg_price])[:3]

    return nearby_supports, nearby_resistances

def detect_candlestick_patterns(df):
    patterns = []
    if len(df) < 1:
        return patterns
    open_ = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    last_open = open_.iloc[-1]
    last_high = high.iloc[-1]
    last_low = low.iloc[-1]
    last_close = close.iloc[-1]
    body = abs(last_open - last_close)
    upper_shadow = last_high - max(last_open, last_close)
    lower_shadow = min(last_open, last_close) - last_low
    total_range = last_high - last_low if last_high > last_low else 1e-6

    if body / total_range < 0.05:
        patterns.append("Doji")
    if lower_shadow > 2 * body and upper_shadow < 0.1 * body and last_close > last_open:
        patterns.append("Hammer")
    if upper_shadow > 2 * body and lower_shadow < 0.1 * body and last_close > last_open:
        patterns.append("Inverted Hammer")
    if upper_shadow > 2 * body and lower_shadow < 0.1 * body and last_close < last_open:
        patterns.append("Shooting Star")
    if lower_shadow > 2 * body and upper_shadow < 0.1 * body and last_close < last_open:
        patterns.append("Hanging Man")
    if body / total_range < 0.3 and upper_shadow > body and lower_shadow > body:
        patterns.append("Spinning Top")
    if upper_shadow < 0.05 * body and lower_shadow < 0.05 * body and last_close > last_open:
        patterns.append("Bullish Marubozu")
    if upper_shadow < 0.05 * body and lower_shadow < 0.05 * body and last_close < last_open:
        patterns.append("Bearish Marubozu")
    if lower_shadow < 0.05 * body and last_close > last_open and last_open <= last_low + 0.001 * total_range:
        patterns.append("Bullish Belt Hold")
    if upper_shadow < 0.05 * body and last_close < last_open and last_open >= last_high - 0.001 * total_range:
        patterns.append("Bearish Belt Hold")

    if len(df) >= 2:
        prev_open = open_.iloc[-2]
        prev_close = close.iloc[-2]
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]

        if prev_close < prev_open and last_close > prev_open and last_open < prev_close and last_close > last_open:
            patterns.append("Bullish Engulfing")
        if prev_close > prev_open and last_close < prev_open and last_open > prev_close and last_close < last_open:
            patterns.append("Bearish Engulfing")
        mid_prev = (prev_open + prev_close) / 2
        if prev_close < prev_open and last_open < prev_close and last_close > mid_prev and last_close < prev_open:
            patterns.append("Piercing Line")
        if prev_close > prev_open and last_open > prev_close and last_close < mid_prev and last_close > prev_open:
            patterns.append("Dark Cloud Cover")
        if prev_close < prev_open and last_open > prev_close and last_close < prev_open and last_close > last_open:
            patterns.append("Bullish Harami")
        if prev_close > prev_open and last_open < prev_close and last_close > prev_open and last_close < last_open:
            patterns.append("Bearish Harami")
        if abs(last_low - prev_low) < 0.001 * last_low and prev_close < prev_open and last_close > last_open:
            patterns.append("Tweezer Bottom")
        if abs(last_high - prev_high) < 0.001 * last_high and prev_close > prev_open and last_close < last_open:
            patterns.append("Tweezer Top")

    if len(df) >= 3:
        p2_open = open_.iloc[-3]
        p2_close = close.iloc[-3]
        p1_open = open_.iloc[-2]
        p1_close = close.iloc[-2]

        if p2_close < p2_open and abs(p1_close - p1_open) < 0.3 * abs(p2_close - p2_open) and last_close > last_open and last_close > (p2_open + p2_close) / 2:
            patterns.append("Morning Star")
        if p2_close > p2_open and abs(p1_close - p1_open) < 0.3 * abs(p2_close - p2_open) and last_close < last_open and last_close < (p2_open + p2_close) / 2:
            patterns.append("Evening Star")
        if all(close.iloc[-i] > open_.iloc[-i] for i in range(1, 4)) and close.iloc[-2] > close.iloc[-3] and close.iloc[-1] > close.iloc[-2]:
            patterns.append("Three White Soldiers")
        if all(close.iloc[-i] < open_.iloc[-i] for i in range(1, 4)) and close.iloc[-2] < close.iloc[-3] and close.iloc[-1] < close.iloc[-2]:
            patterns.append("Three Black Crows")

    return patterns

def get_peak_indices(series):
    return series.index[(series.shift(1) < series) & (series.shift(-1) < series)].tolist()

def get_trough_indices(series):
    return series.index[(series.shift(1) > series) & (series.shift(-1) > series)].tolist()

def detect_chart_patterns(df):
    df_close = df['close']
    patterns = []
    pattern_targets = {}
    pattern_sls = {}
    peak_indices = get_peak_indices(df_close)[-10:]
    trough_indices = get_trough_indices(df_close)[-10:]

    if len(peak_indices) >= 2:
        p1_idx = peak_indices[-2]
        p2_idx = peak_indices[-1]
        p1 = df_close[p1_idx]
        p2 = df_close[p2_idx]
        if abs(p1 - p2) / ((p1 + p2) / 2) < 0.02:
            patterns.append("Double Top")
            neckline = min(df_close[p1_idx:p2_idx])
            height = p1 - neckline
            pattern_targets["Double Top"] = neckline - height
            pattern_sls["Double Top"] = max(p1, p2) + height * 0.1  # Conservative SL above peaks

    if len(trough_indices) >= 2:
        t1_idx = trough_indices[-2]
        t2_idx = trough_indices[-1]
        t1 = df_close[t1_idx]
        t2 = df_close[t2_idx]
        if abs(t1 - t2) / ((t1 + t2) / 2) < 0.02:
            patterns.append("Double Bottom")
            neckline = max(df_close[t1_idx:t2_idx])
            height = neckline - t1
            pattern_targets["Double Bottom"] = neckline + height
            pattern_sls["Double Bottom"] = min(t1, t2) - height * 0.1  # Conservative SL below troughs

    if len(peak_indices) >= 3:
        p1_idx = peak_indices[-3]
        p2_idx = peak_indices[-2]
        p3_idx = peak_indices[-1]
        p1 = df_close[p1_idx]
        p2 = df_close[p2_idx]
        p3 = df_close[p3_idx]
        if max(abs(p1 - p2), abs(p2 - p3), abs(p1 - p3)) / ((p1 + p2 + p3) / 3) < 0.02:
            patterns.append("Triple Top")
        if p2 > p1 and p2 > p3 and abs(p1 - p3) / p2 < 0.02:
            patterns.append("Head and Shoulders")
            neckline = (p1 + p3) / 2
            height = p2 - neckline
            pattern_targets["Head and Shoulders"] = neckline - height
            pattern_sls["Head and Shoulders"] = p2 + height * 0.1

    if len(trough_indices) >= 3:
        t1_idx = trough_indices[-3]
        t2_idx = trough_indices[-2]
        t3_idx = trough_indices[-1]
        t1 = df_close[t1_idx]
        t2 = df_close[t2_idx]
        t3 = df_close[t3_idx]
        if max(abs(t1 - t2), abs(t2 - t3), abs(t1 - t3)) / ((t1 + t2 + t3) / 3) < 0.02:
            patterns.append("Triple Bottom")
        if t2 < t1 and t2 < t3 and abs(t1 - t3) / abs(t2) < 0.02:
            patterns.append("Inverse Head and Shoulders")
            neckline = (t1 + t3) / 2
            height = neckline - t2
            pattern_targets["Inverse Head and Shoulders"] = neckline + height
            pattern_sls["Inverse Head and Shoulders"] = t2 - height * 0.1

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

        if abs(p2 - p1) / df_close.mean() < 0.01 and t2 > t1:
            patterns.append("Ascending Triangle")
            height = p1 - t1
            pattern_targets["Ascending Triangle"] = p2 + height

        if abs(t2 - t1) / df_close.mean() < 0.01 and p2 < p1:
            patterns.append("Descending Triangle")
            height = p1 - t1
            pattern_targets["Descending Triangle"] = t2 - height

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

        if p2 > p1 and t2 > t1:
            if abs(slope_peak - slope_trough) < 0.001:
                patterns.append("Ascending Channel")
            elif p2 - t2 < p1 - t1:
                patterns.append("Rising Wedge")
                height = p1 - t1
                pattern_targets["Rising Wedge"] = df_close.iloc[-1] - height
                pattern_sls["Rising Wedge"] = max(p1, p2) + height * 0.1

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
        trend, confidence, comment = AnalysisEngine.market_trend(df_ohlcv, df_24h)
        rsi = compute_rsi(df_ohlcv['close'])
        macd, signal, hist = compute_macd(df_ohlcv['close'])

        bullish_score = 0
        bearish_score = 0

        if trend in ["bullish", "possible reversal"] and "bullish" in comment.lower():
            bullish_score += 2
        elif trend in ["bearish", "possible reversal"] and "bearish" in comment.lower():
            bearish_score += 2

        if rsi < 30:
            bullish_score += 1
        elif rsi > 70:
            bearish_score += 1

        if macd > signal and hist > 0:
            bullish_score += 1
        elif macd < signal and hist < 0:
            bearish_score += 1

        bullish_pattern = any(p in AnalysisEngine.bullish_candles for p in detected_candles) or any(p in AnalysisEngine.bullish_charts for p in detected_charts)
        bearish_pattern = any(p in AnalysisEngine.bearish_candles for p in detected_candles) or any(p in AnalysisEngine.bearish_charts for p in detected_charts)

        if bullish_pattern:
            bullish_score += 1
        if bearish_pattern:
            bearish_score += 1

        total_score = bullish_score + bearish_score
        if total_score == 0:
            long_conf = 50
            short_conf = 50
            dominant_bias = "neutral"
            pattern_comment = "Mixed signals; market is sideways/neutral."
        else:
            long_conf = int((bullish_score / total_score) * 100)
            short_conf = 100 - long_conf
            if bullish_score > bearish_score:
                dominant_bias = "long"
                pattern_comment = "Overall bullish trend detected."
            elif bearish_score > bullish_score:
                dominant_bias = "short"
                pattern_comment = "Overall bearish trend detected."
            else:
                dominant_bias = "neutral"
                pattern_comment = "Mixed signals; market is sideways/neutral."

        return dominant_bias, long_conf, short_conf, pattern_comment + f" (Bullish score: {bullish_score}, Bearish score: {bearish_score})"

    @staticmethod
    def market_trend(df_ohlcv, df_24h):
        close = df_ohlcv['close']
        ema20 = compute_ema(close, 20).iloc[-1] if len(close) >= 20 else close.iloc[-1] if not close.empty else 0.0
        ma50 = compute_ma(close, 50).iloc[-1] if len(close) >= 50 else close.iloc[-1] if not close.empty else 0.0
        last_close = close.iloc[-1] if not close.empty else 0.0

        # Improved 24h trend
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
            confidence = 85
            comment = "EMA and 24h move bullish, momentum strong."
        elif daily_move_pct < -2 and ema20 < ma50:
            trend = "bearish"
            confidence = 85
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

        rsi = compute_rsi(df_ohlcv['close'])
        if rsi < 30:
            is_bullish_pattern = True
            pattern_comment += " RSI oversold."
        elif rsi > 70:
            is_bearish_pattern = True
            pattern_comment += " RSI overbought."

        macd, signal, hist = compute_macd(df_ohlcv['close'])
        if macd > signal and hist > 0:
            is_bullish_pattern = True
            pattern_comment += " MACD bullish crossover."
        elif macd < signal and hist < 0:
            is_bearish_pattern = True
            pattern_comment += " MACD bearish crossover."

        if dominant_bias == "long":
            if is_bullish_pattern:
                multiplier = 1.5
                extra_conf = 10
                pattern_comment += " Bullish patterns detected, extending target range and increasing confidence."
            elif is_bearish_pattern:
                multiplier = 0.5
                extra_conf = -10
                pattern_comment += " Bearish patterns detected, tightening range and decreasing confidence."
        elif dominant_bias == "short":
            if is_bearish_pattern:
                multiplier = 1.5
                extra_conf = 10
                pattern_comment += " Bearish patterns detected, extending target range and increasing confidence."
            elif is_bullish_pattern:
                multiplier = 0.5
                extra_conf = -10
                pattern_comment += " Bullish patterns detected, tightening range and decreasing confidence."
        else:
            multiplier = 0.75
            extra_conf = -5
            pattern_comment += " Neutral trend; using tighter ranges."

        return multiplier, extra_conf, pattern_comment

    @staticmethod
    def target_stoploss(dominant_bias, supports, resistances, entry_price, pattern_targets, pattern_sls, atr, current_price, max_diff=100):
        targets = []
        stoplosses = []
        if dominant_bias == "long":
            potential_tgts = sorted([r for r in resistances if r > entry_price])
            potential_sls = sorted([s for s in supports if s < entry_price], reverse=True)
            if potential_tgts:
                tgt1 = potential_tgts[0]
                targets.append(tgt1)
                for t in potential_tgts[1:]:
                    if t - tgt1 <= max_diff * 1.5:  # Improved spacing
                        targets.append(t)
                    if len(targets) >= 2:
                        break
            else:
                tgt1 = entry_price + atr * 2.5  # Adjusted multiplier
                targets.append(tgt1)
                tgt2 = entry_price + atr * 5
                targets.append(tgt2)
            if potential_sls:
                sl1 = potential_sls[0]
                stoplosses.append(sl1)
                for s in potential_sls[1:]:
                    if sl1 - s <= max_diff:
                        stoplosses.append(s)
                    if len(stoplosses) >= 2:
                        break
            else:
                sl1 = entry_price - atr * 1.5  # Tighter SL for better RR
                stoplosses.append(sl1)
            for pat, ptgt in pattern_targets.items():
                if pat in AnalysisEngine.bullish_charts and ptgt > entry_price and (not targets or abs(ptgt - targets[-1]) > atr / 2):
                    if len(targets) < 3:
                        targets.append(ptgt)
            for pat, psl in pattern_sls.items():
                if pat in AnalysisEngine.bullish_charts and psl < entry_price and (not stoplosses or abs(psl - stoplosses[-1]) > atr / 2):
                    if len(stoplosses) < 3:
                        stoplosses.append(psl)
        elif dominant_bias == "short":
            potential_tgts = sorted([s for s in supports if s < entry_price], reverse=True)
            potential_sls = sorted([r for r in resistances if r > entry_price])
            if potential_tgts:
                tgt1 = potential_tgts[0]
                targets.append(tgt1)
                for t in potential_tgts[1:]:
                    if tgt1 - t <= max_diff * 1.5:
                        targets.append(t)
                    if len(targets) >= 2:
                        break
            else:
                tgt1 = entry_price - atr * 2.5
                targets.append(tgt1)
                tgt2 = entry_price - atr * 5
                targets.append(tgt2)
            if potential_sls:
                sl1 = potential_sls[0]
                stoplosses.append(sl1)
                for s in potential_sls[1:]:
                    if s - sl1 <= max_diff:
                        stoplosses.append(s)
                    if len(stoplosses) >= 2:
                        break
            else:
                sl1 = entry_price + atr * 1.5
                stoplosses.append(sl1)
            for pat, ptgt in pattern_targets.items():
                if pat in AnalysisEngine.bearish_charts and ptgt < entry_price and (not targets or abs(ptgt - targets[-1]) > atr / 2):
                    if len(targets) < 3:
                        targets.append(ptgt)
            for pat, psl in pattern_sls.items():
                if pat in AnalysisEngine.bearish_charts and psl > entry_price and (not stoplosses or abs(psl - stoplosses[-1]) > atr / 2):
                    if len(stoplosses) < 3:
                        stoplosses.append(psl)
        else:
            nearest_support = min(supports, key=lambda s: abs(s - current_price)) if supports else current_price - atr
            nearest_resistance = min(resistances, key=lambda r: abs(r - current_price)) if resistances else current_price + atr
            targets = [nearest_resistance] if nearest_resistance > current_price else [nearest_support]
            stoplosses = [nearest_support] if nearest_support < current_price else [nearest_resistance]

        return sorted(set(targets)), sorted(set(stoplosses))

    @staticmethod
    def calculate_user_sl(dominant_bias, entry_price, supports, resistances, atr):
        risk_pct = random.choice([0.01, 0.02, 0.03])  # Tighter risk for better honesty
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
    data = InputValidator.validate_and_normalize(input_json)

    if data["market"] == "spot":
        info = client.get_instruments_info(category="spot", symbol=data["coin"])
        if not info['result']['list']:
            raise ValueError(f"Coin {data['coin']} not found or invalid symbol on Bybit spot market!")
    else:
        futures_info = client.get_instruments_info(category="linear", symbol=data["coin"])  # USDT perpetuals
        if not futures_info['result']['list']:
            raise ValueError(f"Coin {data['coin']} not found or invalid symbol on Bybit futures market!")

    current_price = get_current_price(client, data["coin"], data["market"])
    df_ohlcv = get_ohlcv_df(client, data["coin"], data["timeframe"], 200, data["market"])
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

    multiplier, extra_conf, pattern_comment = AnalysisEngine.adjust_based_on_patterns(analysis_bias, detected_candles, detected_charts, trend_conf, df_ohlcv, atr)
    tol = atr * 0.5
    supports, resistances = detect_support_resistance(df_ohlcv, current_price, data["entry_price"], tol=tol)
    max_diff = atr * 5
    targets, stoplosses = AnalysisEngine.target_stoploss(analysis_bias, supports, resistances, data["entry_price"], pattern_targets, pattern_sls, atr, current_price, max_diff=max_diff)
    user_sl = AnalysisEngine.calculate_user_sl(analysis_bias, data["entry_price"], supports, resistances, atr)
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
    return output

# Input Model for FastAPI
class TradeInput(BaseModel):
    coin: str
    market: str
    entry_price: float
    quantity: float
    position_type: str
    timeframe: str
    has_both_positions: Optional[bool] = False

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
    try:
        client = initialize_client()
        data = input_data.dict()
        validated_data = InputValidator.validate_and_normalize(data)
        output = advisory_pipeline(client, validated_data)
        return output
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")