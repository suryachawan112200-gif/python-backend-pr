import json
from fastapi import Response
from starlette import status
import random
from pybit.unified_trading import HTTP
import pandas as pd
import numpy as np  # Added for numerical operations
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Union # Added Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import os
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
import uuid
import warnings

# Suppress pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https.position-analyzer-app.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Bybit Mainnet API keys from environment variables
API_KEY = os.environ.get("BYBIT_API_KEY")
API_SECRET = os.environ.get("BYBIT_API_SECRET")

# --- Input & Validation ---

class InputValidator:
    # MODIFIED: Removed 'risk_pct' and 'has_both_positions' from validation
    REQUIRED_FIELDS = {"coin", "market", "entry_price", "quantity", "position_type"}
    VALID_TIMEFRAMES = ["1h", "4h", "1d", "5m", "15m", "1month"]

    @staticmethod
    def validate_and_normalize(data):
        # Allow 'timeframe' (singular) to be missing if 'timeframes' (plural) is present
        if "timeframes" not in data:
            if "timeframe" not in data:
                 raise ValueError("Missing field: must provide 'timeframes' (list) or 'timeframe' (string)")
        
        # Validate other required fields
        for field in InputValidator.REQUIRED_FIELDS:
             if field not in data:
                 raise ValueError(f"Missing field: {field}")

        data["coin"] = data["coin"].upper().strip()
        data["market"] = data["market"].lower().strip()
        if data["market"] not in ["spot", "futures"]:
            raise ValueError("Market must be 'spot' or 'futures'")
        data["position_type"] = data["position_type"].lower().strip()
        
        # Validate timeframe(s) if present
        if "timeframe" in data:
            data["timeframe"] = data["timeframe"].lower().strip()
            if data["timeframe"] not in InputValidator.VALID_TIMEFRAMES:
                raise ValueError(f"Invalid 'timeframe'. Must be one of: {', '.join(InputValidator.VALID_TIMEFRAMES)}")
        
        if "timeframes" in data:
            if not isinstance(data["timeframes"], list) or not data["timeframes"]:
                raise ValueError("'timeframes' must be a non-empty list")
            validated_tfs = []
            for tf in data["timeframes"]:
                tf_lower = str(tf).lower().strip()
                if tf_lower not in InputValidator.VALID_TIMEFRAMES:
                     raise ValueError(f"Invalid timeframe in 'timeframes' list: {tf}. Must be one of: {', '.join(InputValidator.VALID_TIMEFRAMES)}")
                validated_tfs.append(tf_lower)
            data["timeframes"] = validated_tfs

        # REMOVED: 'has_both_positions' and 'risk_pct' validation
        return data

# --- Bybit Data Fetch ---

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

# MODIFIED: Added "1h" mappings and adjusted lookbacks
def get_interval_mapping(timeframe):
    """
    Returns the Bybit API interval string, ATR multiplier, and lookback period in candles.
    """
    mapping = {
        "5m": 5,
        "15m": 15,
        "1h": 60,     # Added
        "4h": 240,
        "1d": "D",
        "1month": "M"
    }
    # ATR multipliers for stop-loss calculation
    multiplier_adjust = {
        "5m": 1.5,
        "15m": 2.0,
        "1h": 2.2,    # Added
        "4h": 2.5,
        "1d": 3.0,
        "1month": 5.0
    }
    # Lookback periods in candles, aligned with S/R analysis needs
    lookback_periods = {
        "5m": 720,    # ~2.5 days
        "15m": 480,   # ~5 days
        "1h": 360,    # 15 days (for 0.5 month S/R)
        "4h": 540,    # 90 days (for 1.5 month S/R, existing value)
        "1d": 180,    # 180 days (for 6 month S/R)
        "1month": 60  # 5 years
    }
    
    tf_key = timeframe.lower()
    if tf_key not in mapping:
        raise ValueError(f"Invalid timeframe provided to get_interval_mapping: {timeframe}")
        
    return mapping[tf_key], multiplier_adjust[tf_key], lookback_periods[tf_key]

# MODIFIED: No code change, but logic is now robust for 1h, 4h, 1d via get_interval_mapping
def get_ohlcv_df(client, coin, timeframe, market):
    """
    Fetches OHLCV data. The limit logic is adapted via get_interval_mapping.
    """
    interval, _, lookback = get_interval_mapping(timeframe)
    
    # Bybit API max limit is 1000 candles per request
    api_limit = min(lookback, 1000) 
    
    try:
        if market == "spot":
            category = "spot"
        else:
            category = "linear"
            
        klines = client.get_kline(
            category=category,
            symbol=coin,
            interval=interval,
            limit=api_limit 
        )

        if not klines['result'] or 'list' not in klines['result'] or not klines['result']['list']:
            raise ValueError(f"No OHLCV data available for {coin} on {timeframe}")
            
        df = pd.DataFrame(klines['result']['list'], columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce').astype('int64')
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.sort_values('open_time')
        
        # Convert all OHLCV columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            
        # Drop any rows that failed conversion
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'open_time'])
        
        if len(df) < 20: # Need at least ~20 periods for most indicators
             raise ValueError(f"Insufficient OHLCV data for analysis on {timeframe} (got {len(df)} candles)")

        logger.info(f"OHLCV data for {coin} ({timeframe}): {len(df)} candles retrieved")
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
        
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV data for {coin} ({timeframe}): {str(e)}")
        raise ValueError(f"Failed to fetch OHLCV data for {coin} ({timeframe}): {str(e)}")

def get_24h_trend_df(client, coin, market):
    try:
        if market == "spot":
            category = "spot"
        else:
            category = "linear"
            
        klines = client.get_kline(
            category=category,
            symbol=coin,
            interval=60, # 1-hour interval
            limit=24     # Last 24 hours
        )
        
        if not klines['result'] or 'list' not in klines['result'] or not klines['result']['list']:
            raise ValueError(f"No 24h OHLCV data available for {coin}")
            
        df = pd.DataFrame(klines['result']['list'], columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce').astype('int64')
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.sort_values('open_time')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'open_time'])

        if df.empty:
            raise ValueError(f"No valid 24h OHLCV data after processing for {coin}")

        daily_move_pct = ((df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0]) * 100 if df['open'].iloc[0] != 0 else 0.0
        logger.info(f"24h data for {coin}: Open (first) = {df['open'].iloc[0]}, Close (last) = {df['close'].iloc[-1]}, Daily move: {daily_move_pct:.2f}%")
        
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        logger.error(f"Failed to fetch 24h OHLCV data for {coin}: {str(e)}")
        return pd.DataFrame()

# --- Indicator & Analysis Helpers ---

# (Existing helpers: compute_ema, compute_ma, compute_atr, compute_rsi, compute_macd)
def compute_ema(series, span):
    if series.empty or len(series) < span:
        return pd.Series([np.nan] * len(series), index=series.index)
    return series.ewm(span=span, adjust=False).mean()

def compute_ma(series, window):
    if series.empty or len(series) < window:
        return pd.Series([np.nan] * len(series), index=series.index)
    return series.rolling(window=window).mean()

def compute_atr(df, period=14):
    if len(df) < period:
        return np.nan
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.iloc[-1] if not atr.empty else np.nan

def compute_rsi(series, period=14):
    if len(series) < period:
        return 50.0 # Return neutral if not enough data
    delta = series.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0

def compute_macd(series, fast=12, slow=26, signal=9):
    if len(series) < slow:
        return (0.0, 0.0, 0.0)
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = compute_ema(macd, signal)
    histogram = macd - signal_line
    
    if any(pd.isna([macd.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]])):
        return (0.0, 0.0, 0.0)
        
    return (macd.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1])

# NEW: Helper function for ADX
def compute_adx(df, period=14):
    """
    Computes ADX (Average Directional Index), +DI, and -DI.
    """
    if len(df) < period * 2: # ADX needs more data
        return (20.0, 20.0, 20.0) # Return neutral ADX

    df['tr'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    df['dmp'] = (df['high'] - df['high'].shift()).apply(lambda x: x if x > 0 else 0)
    df['dmn'] = (df['low'].shift() - df['low']).apply(lambda x: x if x > 0 else 0)

    # Smooth TR, +DM, -DM
    df['atr_adx'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    df['dmp_smooth'] = df['dmp'].ewm(alpha=1/period, adjust=False).mean()
    df['dmn_smooth'] = df['dmn'].ewm(alpha=1/period, adjust=False).mean()

    # Calculate +DI and -DI
    df['plus_di'] = 100 * (df['dmp_smooth'] / df['atr_adx'].replace(0, 1e-10))
    df['minus_di'] = 100 * (df['dmn_smooth'] / df['atr_adx'].replace(0, 1e-10))
    
    # Calculate DX
    df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, 1e-10))
    
    # Calculate ADX
    df['adx'] = df['dx'].ewm(alpha=1/period, adjust=False).mean()

    adx_val = df['adx'].iloc[-1]
    plus_di_val = df['plus_di'].iloc[-1]
    minus_di_val = df['minus_di'].iloc[-1]

    if any(pd.isna([adx_val, plus_di_val, minus_di_val])):
        return (20.0, 20.0, 20.0) # Neutral fallback

    return (adx_val, plus_di_val, minus_di_val)

# NEW: Helper function for VWAP
def compute_vwap(df, period=20):
    """
    Computes a rolling VWAP (Volume Weighted Average Price).
    """
    if 'volume' not in df.columns or len(df) < period:
        return pd.Series([np.nan] * len(df), index=df.index)
        
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tpv = (typical_price * df['volume']).rolling(window=period).sum()
    volume_sum = df['volume'].rolling(window=period).sum()
    
    vwap = tpv / volume_sum.replace(0, 1e-10)
    return vwap

# NEW: Main indicator computation helper
def compute_indicators_for_df(df: pd.DataFrame) -> Dict[str, Union[float, tuple]]:
    """
    Computes all required indicators for a given OHLCV DataFrame.
    """
    if df.empty or len(df) < 20: # Ensure minimum data
        logger.warning("Insufficient data for indicator computation.")
        return {
            "ema20": 0.0, "ma50": 0.0, "atr": 0.0, "rsi": 50.0,
            "macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
            "adx": 20.0, "plus_di": 20.0, "minus_di": 20.0,
            "vwap": 0.0, "volume_avg_20": 0.0, "latest_volume": 0.0
        }

    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()

    close_prices = df_copy['close']
    
    # Standard indicators
    ema20 = compute_ema(close_prices, 20).iloc[-1]
    ma50 = compute_ma(close_prices, 50).iloc[-1]
    atr = compute_atr(df_copy, 14)
    rsi = compute_rsi(close_prices, 14)
    macd, macd_signal, macd_hist = compute_macd(close_prices)
    
    # ADX/DI
    adx, plus_di, minus_di = compute_adx(df_copy, 14)
    
    # VWAP
    vwap = compute_vwap(df_copy, 20).iloc[-1]
    
    # Volume
    volume_avg_20 = df_copy['volume'].rolling(20).mean().iloc[-1]
    latest_volume = df_copy['volume'].iloc[-1]
    
    # Handle NaNs from computation
    current_price = close_prices.iloc[-1]
    
    return {
        "ema20": ema20 if not pd.isna(ema20) else current_price,
        "ma50": ma50 if not pd.isna(ma50) else current_price,
        "atr": atr if not pd.isna(atr) else 0.0,
        "rsi": rsi if not pd.isna(rsi) else 50.0,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "vwap": vwap if not pd.isna(vwap) else current_price,
        "volume_avg_20": volume_avg_20 if not pd.isna(volume_avg_20) else 0.0,
        "latest_volume": latest_volume if not pd.isna(latest_volume) else 0.0
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
    
    supports = [s for s in [s1, s2, s3] if s > 0 and s < close]
    resistances = [r for r in [r1, r2, r3] if r > 0 and r > close]
    
    return supports, resistances, pivot

# REPLACED/MODIFIED: Replaced old S/R function with this new one as requested
def detect_supports_resistances_for_df(df, current_price, timeframe, window=5, tol=0.01):
    """
    Detects Support and Resistance levels based on timeframe-specific lookbacks,
    touch count, and volume weighting.
    """
    if df.empty:
        return [], []

    # 1. Define timeframe-specific lookback in months and convert to candles
    lookback_months_map = {"1h": 0.5, "4h": 1.5, "1d": 6.0}
    # Candles per day for each timeframe
    candles_per_day_map = {"1h": 24, "4h": 6, "1d": 1}
    
    # Use default if timeframe is not in the primary list (e.g., "15m")
    default_months = 1.0
    default_candles_per_day = 24 * (60 / 15) # Default to 15m logic
    if timeframe in ["5m", "15m"]:
         default_candles_per_day = 24 * (60 / int(timeframe[:-1]))
    elif timeframe == "1month":
         default_candles_per_day = 1.0 / 30.5 # approx

    lookback_months = lookback_months_map.get(timeframe, default_months)
    candles_per_day = candles_per_day_map.get(timeframe, default_candles_per_day)
    
    lookback_candles = int(lookback_months * 30.5 * candles_per_day)
    
    # Ensure lookback is not larger than the dataframe
    lookback_candles = min(lookback_candles, len(df))
    if lookback_candles < window * 2: # Need some data
        return [], []
        
    analysis_df = df.tail(lookback_candles).copy()
    
    # 2. Find pivot candidates (local minima/maxima)
    analysis_df['lows'] = analysis_df['low'].rolling(window, center=True).min()
    analysis_df['highs'] = analysis_df['high'].rolling(window, center=True).max()
    
    candidate_supports_idx = analysis_df.index[analysis_df['low'] == analysis_df['lows']]
    candidate_resistances_idx = analysis_df.index[analysis_df['high'] == analysis_df['highs']]
    
    candidate_supports = analysis_df.loc[candidate_supports_idx]
    candidate_resistances = analysis_df.loc[candidate_resistances_idx]

    # 3. Rank levels by touch count and volume
    avg_volume = analysis_df['volume'].mean()
    
    def get_ranked_levels(candidates, series):
        levels = {} # {price_cluster: {"touches": 0, "total_volume": 0, "prices": []}}
        
        for idx, row in candidates.iterrows():
            level = row['low'] if series == 'low' else row['high']
            volume = row['volume']
            
            # Cluster nearby levels
            found_cluster = False
            for cluster_price in levels.keys():
                if abs(level - cluster_price) / cluster_price < tol:
                    levels[cluster_price]["touches"] += 1
                    levels[cluster_price]["total_volume"] += volume
                    levels[cluster_price]["prices"].append(level)
                    found_cluster = True
                    break
            
            if not found_cluster:
                levels[level] = {"touches": 1, "total_volume": volume, "prices": [level]}
        
        ranked_list = []
        for cluster_price, data in levels.items():
            # Scoring: (touch count) * (log of volume ratio)
            # Use log to prevent massive volume spikes from dominating
            volume_ratio = data["total_volume"] / (data["touches"] * avg_volume + 1e-10)
            score = data["touches"] * (1 + np.log1p(max(0, volume_ratio)))
            
            # Use the average price of the cluster
            avg_price = np.mean(data["prices"])
            ranked_list.append({"price": avg_price, "rank": score, "touches": data["touches"]})
            
        return sorted(ranked_list, key=lambda x: x["rank"], reverse=True)

    supports = get_ranked_levels(candidate_supports, 'low')
    resistances = get_ranked_levels(candidate_resistances, 'high')

    # 4. Add Pivot Points
    pivot_supports, pivot_resistances, _ = compute_pivot_points(analysis_df)
    for p in pivot_supports:
        supports.append({"price": p, "rank": 1.5, "touches": 1}) # Give pivots a decent rank
    for p in pivot_resistances:
        resistances.append({"price": p, "rank": 1.5, "touches": 1})

    # 5. Filter and sort final lists
    final_supports = sorted(
        [s for s in supports if s["price"] < current_price], 
        key=lambda x: x["price"], 
        reverse=True
    )
    final_resistances = sorted(
        [r for r in resistances if r["price"] > current_price], 
        key=lambda x: x["price"]
    )

    # 6. Fallback to ATR if no levels are found
    if not final_supports or not final_resistances:
        atr = compute_atr(df)
        if pd.isna(atr) or atr == 0:
            atr = current_price * 0.01 # 1% fallback
            
        if not final_supports:
            final_supports = [{"price": current_price - atr, "rank": 1, "touches": 0, "type": "atr_fallback"}]
        if not final_resistances:
            final_resistances = [{"price": current_price + atr, "rank": 1, "touches": 0, "type": "atr_fallback"}]

    logger.info(f"S/R for {timeframe}: {len(final_supports)} supports, {len(final_resistances)} resistances found.")
    
    # Return top N levels (e.g., top 5)
    return final_supports[:5], final_resistances[:5]


# (Existing: detect_candlestick_patterns, get_peak_indices, get_trough_indices, detect_chart_patterns)
# ... (These functions are long, assuming they exist as provided) ...
def detect_candlestick_patterns(df):
    patterns = []
    if len(df) < 1:
        return patterns
    
    try:
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

        if total_range > 0 and body / total_range < 0.1 and last_volume > avg_volume * 1.2:
            patterns.append("Doji")
        if total_range > 0 and body > 0 and lower_shadow > 2 * body and upper_shadow < 0.3 * body and last_close > last_open and last_volume > avg_volume:
            patterns.append("Hammer")
        if total_range > 0 and body > 0 and upper_shadow > 2 * body and lower_shadow < 0.3 * body and last_close < last_open and last_volume > avg_volume:
            patterns.append("Shooting Star")

        if len(df) >= 2:
            prev_open = open_.iloc[-2]
            prev_close = close.iloc[-2]
            prev_volume = volume.iloc[-2]
            if prev_close < prev_open and last_close > prev_open and last_open < prev_close and last_close > last_open and last_volume > prev_volume * 1.1:
                patterns.append("Bullish Engulfing")
            if prev_close > prev_open and last_close < prev_open and last_open > prev_close and last_close < last_open and last_volume > prev_volume * 1.1:
                patterns.append("Bearish Engulfing")

        if len(df) >= 3:
            p2_open = open_.iloc[-3]
            p2_close = close.iloc[-3]
            p1_open = open_.iloc[-2]
            p1_close = close.iloc[-2]
            p1_volume = volume.iloc[-2]
            p1_body = abs(p1_open - p1_close)
            p2_body = abs(p2_open - p2_close)
            if p2_close < p2_open and p1_body < 0.3 * p2_body and last_close > last_open and last_close > (p2_open + p2_close) / 2 and last_volume > p1_volume:
                patterns.append("Morning Star")
            if p2_close > p2_open and p1_body < 0.3 * p2_body and last_close < last_open and last_close < (p2_open + p2_close) / 2 and last_volume > p1_volume:
                patterns.append("Evening Star")
    except IndexError:
        logger.warning("Index error in candlestick detection, likely short DF.")
    except Exception as e:
         logger.error(f"Error in detect_candlestick_patterns: {e}")

    logger.info(f"Detected candlestick patterns: {patterns}")
    return patterns

def get_peak_indices(series):
    return series.index[(series.shift(1) < series) & (series.shift(-1) < series)].tolist()

def get_trough_indices(series):
    return series.index[(series.shift(1) > series) & (series.shift(-1) > series)].tolist()

def detect_chart_patterns(df):
    if df.empty or len(df) < 50: # Chart patterns need more data
        logger.warning("Empty or short DataFrame in detect_chart_patterns")
        return [], {}, {}
        
    df_close = df['close']
    patterns = []
    pattern_targets = {}
    pattern_sls = {}
    
    try:
        peak_indices = get_peak_indices(df_close)[-10:]
        trough_indices = get_trough_indices(df_close)[-10:]
        current = df_close.iloc[-1]

        if len(peak_indices) >= 2:
            p1_idx, p2_idx = peak_indices[-2], peak_indices[-1]
            p1, p2 = df_close[p1_idx], df_close[p2_idx]
            if abs(p1 - p2) / ((p1 + p2) / 2) < 0.02: # Within 2%
                slice_data = df_close[p1_idx:p2_idx+1]
                if not slice_data.empty:
                    neckline = slice_data.min()
                    height = p1 - neckline
                    if height > 0 and current < (neckline + 0.1 * height): # Breaking neckline
                        patterns.append("Double Top")
                        pattern_targets["Double Top"] = neckline - height
                        pattern_sls["Double Top"] = max(p1, p2)

        if len(trough_indices) >= 2:
            t1_idx, t2_idx = trough_indices[-2], trough_indices[-1]
            t1, t2 = df_close[t1_idx], df_close[t2_idx]
            if abs(t1 - t2) / ((t1 + t2) / 2) < 0.02:
                slice_data = df_close[t1_idx:t2_idx+1]
                if not slice_data.empty:
                    neckline = slice_data.max()
                    height = neckline - t1
                    if height > 0 and current > (neckline - 0.1 * height): # Breaking neckline
                        patterns.append("Double Bottom")
                        pattern_targets["Double Bottom"] = neckline + height
                        pattern_sls["Double Bottom"] = min(t1, t2)

        # (Simplified, add H&S, Triangles etc. here following the same logic)
        # ...
    
    except IndexError:
        logger.warning("Index error in chart pattern detection.")
    except Exception as e:
         logger.error(f"Error in detect_chart_patterns: {e}")
         
    logger.info(f"Detected chart patterns: {patterns}")
    return patterns, pattern_targets, pattern_sls

# --- NEW: Single Timeframe Analysis Core ---

def _compute_confidence_and_bias(indicators: dict, patterns: list, df: pd.DataFrame) -> (str, int):
    """
    Internal helper to compute bias and confidence score for a single timeframe.
    """
    
    # 1. Trend Score (0-1)
    ema_score = 0.5
    if indicators["ema20"] > indicators["ma50"]:
        ema_score = 1.0
    elif indicators["ema20"] < indicators["ma50"]:
        ema_score = 0.0
    
    # ADX: 0 if < 20, 1 if > 50. Linearly scaled.
    adx_score = min(max(indicators["adx"] - 20, 0) / 30, 1) 
    
    # MACD Hist: 1 if hist matches trend, 0 if not
    macd_hist_score = 0.5
    if indicators["macd_hist"] > 0 and ema_score > 0.5:
        macd_hist_score = 1.0
    elif indicators["macd_hist"] < 0 and ema_score < 0.5:
        macd_hist_score = 1.0
    elif indicators["macd_hist"] > 0 and ema_score < 0.5:
        macd_hist_score = 0.0
    elif indicators["macd_hist"] < 0 and ema_score > 0.5:
        macd_hist_score = 0.0
        
    trend_score = (0.5 * ema_score) + (0.3 * adx_score) + (0.2 * macd_hist_score)

    # 2. Momentum Score (0-1)
    # RSI: 0 at 30, 0.5 at 50, 1.0 at 70.
    rsi_score = min(max((indicators["rsi"] - 30) / 40, 0), 1)
    
    # MACD Hist Mag: Normalized by ATR
    atr_val = indicators["atr"] if indicators["atr"] > 0 else df['close'].iloc[-1] * 0.01
    macd_hist_mag_norm = abs(indicators["macd_hist"]) / atr_val
    # Scale: 0 if 0, 1 if mag > 0.05 * ATR (arbitrary threshold)
    macd_mag_score = min(macd_hist_mag_norm / 0.05, 1.0) 
    
    momentum_score = (0.7 * rsi_score) + (0.3 * macd_mag_score)
    
    # 3. Volume Score (0-1)
    volume_score = 0.5
    if indicators["latest_volume"] > 0 and indicators["volume_avg_20"] > 0:
        volume_ratio = indicators["latest_volume"] / indicators["volume_avg_20"]
        # Scale: 0.5 at 1x, 1.0 at 2x, 0.0 at 0.5x
        if volume_ratio > 1:
            volume_score = 0.5 + min((volume_ratio - 1) / 2, 0.5) # Caps at 3x avg
        else:
            volume_score = max(volume_ratio * 0.5, 0) # Scales down to 0
            
    # 4. Pattern Score (0-1)
    pattern_score = 0.0
    strong_patterns = [
        "Bullish Engulfing", "Bearish Engulfing", "Morning Star", "Evening Star",
        "Double Bottom", "Double Top", "Head and Shoulders", "Inverse Head and Shoulders"
    ]
    if any(p in patterns for p in strong_patterns):
        pattern_score = 0.4 # Base score for a strong pattern
    elif patterns:
        pattern_score = 0.2 # Base score for any minor pattern
        
    # 5. Raw Score
    raw_score_tf = (0.4 * trend_score + 
                    0.25 * momentum_score + 
                    0.2 * volume_score + 
                    0.15 * pattern_score)
    
    confidence_tf = round(raw_score_tf * 100)

    # 6. Bias Direction
    bias = "neutral"
    # Bias logic: If trend is clear (>0.55) and momentum confirms (>0.5), set bias.
    if trend_score > 0.6 and momentum_score > 0.55:
        bias = "long" if ema_score > 0.5 else "short" # Bias from trend direction
    elif trend_score < 0.4 and momentum_score < 0.45:
         bias = "short" if ema_score < 0.5 else "long"
    
    # Override bias if strong conflicting pattern
    if bias == "long" and any(p in patterns for p in ["Bearish Engulfing", "Evening Star", "Double Top"]):
        bias = "neutral"
    if bias == "short" and any(p in patterns for p in ["Bullish Engulfing", "Morning Star", "Double Bottom"]):
        bias = "neutral"
        
    return bias, confidence_tf

# --- (This is the RECENTLY FIXED function) ---
def _compute_targets_and_stoploss(
    bias: str, 
    current_price: float, 
    supports: List[dict], 
    resistances: List[dict], 
    pattern_targets: dict, 
    pattern_sls: dict,
    atr: float,
    atr_multiplier: float
) -> (List[dict], List[dict]):
    """
    Internal helper to compute targets and stoploss for a single timeframe.
    Returns lists of dictionaries: [{"price": 123.45, "type": "sr", "rank": 2.5}, ...]
    """
    
    targets = []
    stoplosses = []
    
    # Use pattern targets/sl if they match the bias
    if bias == "long":
        for p_name, p_target in pattern_targets.items():
            if "Bottom" in p_name or "Bullish" in p_name or "Inverse" in p_name:
                if p_target > current_price:
                    targets.append({"price": p_target, "type": f"pattern_{p_name}", "rank": 10})
        for p_name, p_sl in pattern_sls.items():
             if "Bottom" in p_name or "Bullish" in p_name or "Inverse" in p_name:
                if p_sl < current_price:
                    stoplosses.append({"price": p_sl, "type": f"pattern_{p_name}", "rank": 10})
                    
    elif bias == "short":
        for p_name, p_target in pattern_targets.items():
            if "Top" in p_name or "Bearish" in p_name or "Head and Shoulders" in p_name:
                if p_target < current_price:
                    targets.append({"price": p_target, "type": f"pattern_{p_name}", "rank": 10})
        for p_name, p_sl in pattern_sls.items():
            if "Top" in p_name or "Bearish" in p_name or "Head and Shoulders" in p_name:
                if p_sl > current_price:
                    stoplosses.append({"price": p_sl, "type": f"pattern_{p_name}", "rank": 10})

    # Add S/R levels
    if bias == "long":
        # Targets are resistances above
        for r in resistances:
            targets.append({"price": r["price"], "type": r.get("type", "sr_resistance"), "rank": r.get("rank", 1)})
        # Stoplosses are supports below
        for s in supports:
            stoplosses.append({"price": s["price"], "type": s.get("type", "sr_support"), "rank": s.get("rank", 1)})
            
    elif bias == "short":
        # Targets are supports below
        for s in supports:
            targets.append({"price": s["price"], "type": s.get("type", "sr_support"), "rank": s.get("rank", 1)})
        # Stoplosses are resistances above
        for r in resistances:
            stoplosses.append({"price": r["price"], "type": r.get("type", "sr_resistance"), "rank": r.get("rank", 1)})

    # Add ATR-based fallback levels
    # FIX 2: Check for (not pd.isna(atr) and atr > 0) to handle NaN values
    if not pd.isna(atr) and atr > 0:
        atr_sl_long = current_price - (atr * atr_multiplier)
        atr_tp1_long = current_price + (atr * atr_multiplier * 1.5) # 1.5 R:R
        atr_tp2_long = current_price + (atr * atr_multiplier * 3.0) # 3.0 R:R
        
        atr_sl_short = current_price + (atr * atr_multiplier)
        atr_tp1_short = current_price - (atr * atr_multiplier * 1.5)
        atr_tp2_short = current_price - (atr * atr_multiplier * 3.0)
        
        if bias == "long":
            if not stoplosses:
                stoplosses.append({"price": atr_sl_long, "type": "atr_fallback", "rank": 0.5})
            if not targets:
                targets.append({"price": atr_tp1_long, "type": "atr_fallback", "rank": 0.5})
                targets.append({"price": atr_tp2_long, "type": "atr_fallback", "rank": 0.5})
                
        elif bias == "short":
            if not stoplosses:
                stoplosses.append({"price": atr_sl_short, "type": "atr_fallback", "rank": 0.5})
            if not targets:
                targets.append({"price": atr_tp1_short, "type": "atr_fallback", "rank": 0.5})
                targets.append({"price": atr_tp2_short, "type": "atr_fallback", "rank": 0.5})

    # Sort and filter
    if bias == "long":
        final_targets = sorted([t for t in targets if t["price"] > current_price], key=lambda x: x["price"])[:3]
        final_sl = sorted([s for s in stoplosses if s["price"] < current_price], key=lambda x: x["price"], reverse=True)[:1]
    elif bias == "short":
        final_targets = sorted([t for t in targets if t["price"] < current_price], key=lambda x: x["price"], reverse=True)[:3]
        final_sl = sorted([s for s in stoplosses if s["price"] > current_price], key=lambda x: x["price"])[:1]
    
    # FIX 1: This 'else' block is modified to add the 'type' key
    else: # Neutral
        # Targets are the closest resistances
        final_targets_raw = sorted([r for r in resistances if r["price"] > current_price], key=lambda x: x["price"])[:2]
        # Stoplosses are the closest supports
        final_sl_raw = sorted([s for s in supports if s["price"] < current_price], key=lambda x: x["price"], reverse=True)[:1]

        # Re-format them to include the 'type' key
        # Use .get() to safely read the type from the raw S/R dict (in case it was an atr_fallback)
        final_targets = [{"price": t["price"], "type": t.get("type", "sr_resistance"), "rank": t.get("rank", 1)} for t in final_targets_raw]
        final_sl = [{"price": s["price"], "type": s.get("type", "sr_support"), "rank": s.get("rank", 1)} for s in final_sl_raw]

        # FIX 2: Add NaN check to the fallback logic for neutral bias as well
        if (not pd.isna(atr) and atr > 0):
            if not final_targets:
                final_targets = [{"price": current_price + (atr * atr_multiplier * 1.5), "type": "atr_fallback", "rank": 0.5}]
            if not final_sl:
                final_sl = [{"price": current_price - (atr * atr_multiplier), "type": "atr_fallback", "rank": 0.5}]

    return final_targets, final_sl
# --- (End of fixed function) ---


# NEW: Main function to analyze a single timeframe
def analyze_single_timeframe(
    df: pd.DataFrame, 
    current_price: float, 
    entry_price: float, 
    timeframe: str, 
    quantity: float,
    extra_params: dict
) -> Dict:
    """
    Performs a complete analysis for a single timeframe and returns a results dictionary.
    """
    logger.info(f"--- Analyzing Timeframe: {timeframe} ---")
    
    # 1. Compute Indicators
    indicators = compute_indicators_for_df(df)
    
    # 2. Detect Supports & Resistances
    supports, resistances = detect_supports_resistances_for_df(df, current_price, timeframe)
    
    # 3. Detect Patterns
    candlestick_patterns = detect_candlestick_patterns(df)
    chart_patterns, pattern_targets, pattern_sls = detect_chart_patterns(df)
    all_patterns = list(set(candlestick_patterns + chart_patterns))
    
    # 4. Compute Bias and Confidence
    bias, confidence = _compute_confidence_and_bias(indicators, all_patterns, df)
    
    # 5. Compute Targets and Stoploss
    _, atr_multiplier, _ = get_interval_mapping(timeframe)
    atr = indicators.get("atr", 0.0)
    
    targets, stoplosses = _compute_targets_and_stoploss(
        bias, current_price, supports, resistances,
        pattern_targets, pattern_sls, atr, atr_multiplier
    )
    
    logger.info(f"--- Analysis Complete: {timeframe} (Bias: {bias}, Conf: {confidence}) ---")

    # 6. Format and Return Results
    return {
        "bias": bias,
        "confidence": confidence,
        "targets": targets,
        "stoplosses": stoplosses,
        "patterns": all_patterns,
        "supports": supports,
        "resistances": resistances,
        "indicators": indicators # Full indicator data
    }


# --- (Existing: PnL, Recommendation Helpers) ---
# ... (Assuming compute_pnl_rr, generate_recommendation exist) ...
def compute_pnl_rr(entry_price, current_price, targets, stoplosses, position_type, has_both_positions=False):
    if not targets or not stoplosses:
        return {"unrealized_pnl_pct": 0, "best_rr": 0, "pnl_at_tp1": 0, "loss_at_sl1": 0}

    pnl_pct = 0
    if position_type == 'long':
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
    elif position_type == 'short':
        pnl_pct = ((entry_price - current_price) / entry_price) * 100

    sl_price = stoplosses[0]['price']
    tp_price = targets[0]['price']
    
    potential_loss = 0
    potential_gain = 0

    if position_type == 'long':
        potential_loss = abs(((sl_price - entry_price) / entry_price) * 100)
        potential_gain = abs(((tp_price - entry_price) / entry_price) * 100)
    elif position_type == 'short':
        potential_loss = abs(((sl_price - entry_price) / entry_price) * 100)
        potential_gain = abs(((tp_price - entry_price) / entry_price) * 100)
        
    best_rr = potential_gain / potential_loss if potential_loss > 0 else 0

    return {
        "unrealized_pnl_pct": round(pnl_pct, 2),
        "best_rr": round(best_rr, 2),
        "pnl_at_tp1": round(potential_gain, 2),
        "loss_at_sl1": round(potential_loss, 2)
    }

# MODIFIED: Renamed function and removed unused 'quantity' and 'risk_pct' params
def get_primary_levels(entry_price, stoplosses, targets, position_type):
    """
    Gets the primary SL and TP levels for display.
    """
    if not stoplosses or not targets:
        return None, None
        
    sl_price = stoplosses[0]['price']
    tp_price = targets[0]['price']
    
    sl_pct = abs(sl_price - entry_price) / entry_price if entry_price > 0 else 0
    tp_pct = abs(tp_price - entry_price) / entry_price if entry_price > 0 else 0
    
    return {"sl_price": sl_price, "sl_pct": sl_pct}, \
           {"tp_price": tp_price, "tp_pct": tp_pct}


def generate_recommendation(bias, confidence, pnl_rr, position_type, entry_price, current_price, patterns):
    pnl_pct = pnl_rr.get('unrealized_pnl_pct', 0)
    rr = pnl_rr.get('best_rr', 0)
    
    # Rationale is the "sentiment"
    rationale = f"Aggregated bias is {bias} with {confidence}% confidence."
    
    if "conflict" in patterns: # Using patterns list to pass conflict flag
        rationale += " However, bias is neutral due to strong conflicts between timeframes. Risk is high."
        return "Avoid New Positions", rationale

    if bias == "neutral":
        rationale += " The market shows no clear direction."
        if pnl_pct > 10:
             return "Consider Taking Partial Profit", rationale + " Secure gains during consolidation."
        elif pnl_pct < -10:
             return "Consider Reducing Position", rationale + " Market is consolidating against you."
        return "Hold / Monitor", rationale

    if (bias == "long" and position_type == "short") or (bias == "short" and position_type == "long"):
        rationale += f" The market trend is {bias}, which opposes your {position_type} position."
        if pnl_pct > 0:
            return "Take Profit", rationale + " Secure profits before trend reverses."
        return "Close Position / Hedge", rationale + " Trend is strongly against your position."

    # Position matches bias
    rationale += f" The market trend ({bias}) aligns with your {position_type} position."
    if rr < 1.0:
        rationale += f" However, the current Risk/Reward ratio ({rr}) is poor."
        if pnl_pct > 0:
            return "Take Profit", rationale + " Entry R:R is unfavorable."
        return "Hold / Monitor", rationale + " Avoid adding to this position."
        
    rationale += f" The Risk/Reward ratio ({rr}) is favorable."
    
    if pnl_pct > pnl_rr.get('pnl_at_tp1', 3) * 0.8: # Near TP1
        return "Take Partial Profit", rationale + " Position is nearing the first target."
    if pnl_pct < 0:
        return "Hold Position", rationale + " Bias is favorable, but position is in drawdown."
    
    return "Add to Position", rationale + " Favorable R:R and matching bias suggest adding."

# --- NEW: Aggregation Logic ---

def _aggregate_results(per_tf_results: Dict[str, Dict]) -> Dict:
    """
    Aggregates results from multiple timeframes into a single recommendation.
    """
    
    # --- 1. Handle Single TF Case (Backward Compatibility) ---
    if len(per_tf_results) == 1:
        tf_key = list(per_tf_results.keys())[0]
        tf_result = per_tf_results[tf_key]
        
        # Format targets/stoplosses to match aggregated format
        agg_targets = [
            {"price": t["price"], "type": t["type"], "confirmed_by": [tf_key]}
            for t in tf_result["targets"]
        ]
        agg_sl = [
            {"price": s["price"], "type": s["type"], "confirmed_by": [tf_key]}
            for s in tf_result["stoplosses"]
        ]

        return {
            "bias": tf_result["bias"],
            "confidence": tf_result["confidence"],
            "targets": agg_targets,
            "stoplosses": agg_sl,
            "primary_timeframe": tf_key,
            "remarks": [f"Single timeframe analysis based on {tf_key}."],
            "conflict": False,
            "patterns": tf_result["patterns"] # Pass patterns for recommendation engine
        }
        
    # --- 2. Multi-Timeframe Aggregation ---
    weights = {"1d": 0.40, "4h": 0.35, "1h": 0.25}
    bias_map = {"long": 1, "neutral": 0, "short": -1}
    inv_bias_map = {1: "long", 0: "neutral", -1: "short"}
    
    bias_score = 0.0
    weighted_confidences = {}
    total_weight = 0.0
    
    tf_biases = {}
    
    # Calculate weighted bias score and find primary timeframe
    for tf, result in per_tf_results.items():
        if tf not in weights: # Skip TFs not in our primary list (e.g., 15m)
            continue
            
        w = weights[tf]
        bias_numeric = bias_map.get(result["bias"], 0)
        confidence_norm = result["confidence"] / 100.0
        
        # bias_score = sum(w_tf * bias_numeric_tf * (confidence_tf/100))
        bias_score += w * bias_numeric * confidence_norm
        
        # Find TF with highest weighted *confidence* (not bias score)
        weighted_confidences[tf] = w * confidence_norm
        
        total_weight += w
        tf_biases[tf] = result["bias"]

    if total_weight > 0:
        # Normalize score by the sum of weights *used*
        bias_score_normalized = bias_score / total_weight 
    else:
        bias_score_normalized = 0.0 # Should not happen if 1h/4h/1d are present
        
    # Determine primary timeframe (highest weighted confidence)
    if not weighted_confidences:
         # Fallback if only non-standard TFs were used (e.g., ["15m", "5m"])
         primary_timeframe = list(per_tf_results.keys())[0]
    else:
        primary_timeframe = max(weighted_confidences, key=weighted_confidences.get)

    # Determine final bias and confidence
    # final confidence = abs(bias_score) * 100 (clamped)
    # Note: Using the normalized score gives a better 0-100 confidence
    final_confidence = min(round(abs(bias_score_normalized) * 100), 100)
    
    # final bias = sign(bias_score)
    # Use a threshold to default to neutral
    if bias_score_normalized > 0.15: # Needs > 15% weighted bullishness
        final_bias = "long"
    elif bias_score_normalized < -0.15: # Needs > 15% weighted bearishness
        final_bias = "short"
    else:
        final_bias = "neutral"

    # --- 3. Handle Conflicts and Remarks ---
    remarks = []
    is_conflict = False
    
    d_bias = tf_biases.get("1d", "neutral")
    h4_bias = tf_biases.get("4h", "neutral")
    h1_bias = tf_biases.get("1h", "neutral")
    
    # Strong conflict check (e.g., 1D Long vs 4H Short)
    if (d_bias == "long" and (h4_bias == "short" or h1_bias == "short")) or \
       (d_bias == "short" and (h4_bias == "long" or h1_bias == "long")):
        remarks.append(f"Strong Conflict: 1D bias ({d_bias}) opposes lower timeframe(s) ({h4_bias} / {h1_bias}).")
        final_bias = "neutral" # Override bias to neutral
        is_conflict = True
    elif (h4_bias == "long" and h1_bias == "short") or (h4_bias == "short" and h1_bias == "long"):
         remarks.append(f"Mid-level Conflict: 4H bias ({h4_bias}) opposes 1H bias ({h1_bias}).")
         # Don't override, but note it.
         
    if final_bias == "neutral" and not is_conflict and any(b != "neutral" for b in tf_biases.values()):
        remarks.append("Aggregated bias is neutral due to mixed or weak signals.")
    elif not remarks:
        remarks.append("Timeframes are in general agreement.")
        
    remarks.append(f"Primary analysis timeframe: {primary_timeframe} (highest weighted confidence).")

    # --- 4. Aggregate Targets and Stoploss ---
    
    # Start with Primary Timeframe's SL
    primary_sl = per_tf_results[primary_timeframe]["stoplosses"]
    agg_stoploss = [
        {"price": s["price"], "type": s["type"], "confirmed_by": [primary_timeframe]}
        for s in primary_sl
    ]
    
    # Cluster all targets that match the final bias
    all_targets_map = {} # {cluster_center_price: {"prices": [], "confirmed_by": set(), "types": set()}}
    
    for tf, result in per_tf_results.items():
        # Only include targets from TFs that *agree* with the final bias
        if bias_map.get(result["bias"]) == bias_map[final_bias]:
            for tgt in result["targets"]:
                price = tgt["price"]
                
                # Find a cluster (within 0.5%)
                found_cluster = False
                for cluster_price in all_targets_map.keys():
                    if abs(price - cluster_price) / cluster_price < 0.005: # 0.5% tolerance
                        all_targets_map[cluster_price]["prices"].append(price)
                        all_targets_map[cluster_price]["confirmed_by"].add(tf)
                        all_targets_map[cluster_price]["types"].add(tgt["type"])
                        found_cluster = True
                        break
                        
                if not found_cluster:
                    all_targets_map[price] = {
                        "prices": [price],
                        "confirmed_by": {tf},
                        "types": {tgt["type"]}
                    }
                    
    # Format the clustered targets
    agg_targets = []
    for cluster_price_key, data in all_targets_map.items():
        avg_price = np.mean(data["prices"])
        confirmed_by_list = sorted(list(data["confirmed_by"]), key=lambda x: weights.get(x, 0), reverse=True)
        # Type: prefer 'pattern', then 'sr', then 'atr'
        type_str = "target"
        if any("pattern" in t for t in data["types"]):
            type_str = "pattern_target"
        elif any("sr" in t for t in data["types"]):
            type_str = "sr_target"
        elif any("atr" in t for t in data["types"]):
            type_str = "atr_target"
            
        agg_targets.append({
            "price": avg_price,
            "type": type_str,
            "confirmed_by": confirmed_by_list
        })
        
    # Sort final targets list
    agg_targets = sorted(agg_targets, key=lambda x: x["price"], reverse=(final_bias == "short"))

    # Pass conflict flag to recommendation engine via patterns list
    patterns = per_tf_results[primary_timeframe]["patterns"]
    if is_conflict:
        patterns.append("conflict")

    return {
        "bias": final_bias,
        "confidence": final_confidence,
        "targets": agg_targets,
        "stoplosses": agg_stoploss,
        "primary_timeframe": primary_timeframe,
        "remarks": remarks,
        "conflict": is_conflict,
        "patterns": patterns # Pass patterns from primary TF
    }
    
# --- Main FastAPI Endpoint ---

# REMOVED: /auth/check-code endpoint

@app.get("/coins")
async def get_coins_endpoint():
    try:
        client = initialize_client()
        coins = get_all_coins(client)
        return JSONResponse(content={"coins": coins})
    except Exception as e:
        logger.error(f"Error in /coins endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch coin list: {str(e)}")


# MODIFIED: advisory_pipeline (Main Refactor)
@app.post("/analyze")
async def advisory_pipeline(request: Request):
    try:
        data = await request.json()
        
        # --- 1. Validation ---
        try:
            validated_data = InputValidator.validate_and_normalize(data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # --- 2. Determine Timeframes to Analyze ---
        # New default: ["1h", "4h", "1d"]
        # Backward compatibility: If 'timeframe' is sent but 'timeframes' is not, use ['timeframe']
        
        timeframes_to_analyze = validated_data.get("timeframes")
        if timeframes_to_analyze is None:
            single_tf = validated_data.get("timeframe")
            if single_tf:
                # Old request, just run that one TF
                timeframes_to_analyze = [single_tf]
                logger.info(f"Received single 'timeframe' key. Running analysis for: {timeframes_to_analyze}")
            else:
                # New request, no 'timeframes' specified, use new default
                timeframes_to_analyze = ["1h", "4h", "1d"]
                logger.info(f"No 'timeframes' key. Running default analysis for: {timeframes_to_analyze}")
        else:
             logger.info(f"Received 'timeframes' list. Running analysis for: {timeframes_to_analyze}")
             
        # Extract common parameters
        coin = validated_data["coin"]
        market = validated_data["market"]
        entry_price = float(validated_data["entry_price"])
        quantity = float(validated_data["quantity"])
        position_type = validated_data["position_type"]
        # REMOVED: risk_pct and has_both_positions
        
        # --- 3. Initialize Client & Fetch Common Data ---
        client = initialize_client()
        
        # Get current price once
        current_price = get_current_price(client, coin, market)
        
        # Get 24h trend data once
        try:
            df_24h = get_24h_trend_df(client, coin, market)
            daily_move_pct = ((df_24h['close'].iloc[-1] - df_24h['open'].iloc[0]) / df_24h['open'].iloc[0]) * 100 if not df_24h.empty and df_24h['open'].iloc[0] != 0 else 0.0
            daily_trend = "up" if daily_move_pct > 0.5 else "down" if daily_move_pct < -0.5 else "sideways"
        except Exception as e:
            logger.warning(f"Could not get 24h trend data: {e}")
            daily_move_pct = 0.0
            daily_trend = "unknown"

        # --- 4. Per-Timeframe Analysis Loop ---
        per_tf_results = {}
        errors = {}
        
        for tf in timeframes_to_analyze:
            try:
                # 4a. Fetch OHLCV
                df = get_ohlcv_df(client, coin, tf, market)
                
                # 4b. Run Analysis
                tf_result = analyze_single_timeframe(
                    df=df,
                    current_price=current_price,
                    entry_price=entry_price,
                    timeframe=tf,
                    quantity=quantity,
                    extra_params={} # Pass any extra params if needed
                )
                per_tf_results[tf] = tf_result
                
            except Exception as e:
                logger.error(f"Failed to analyze timeframe {tf} for {coin}: {str(e)}")
                errors[tf] = str(e)
        
        if not per_tf_results:
            raise HTTPException(status_code=500, detail=f"Failed to analyze all timeframes for {coin}. Errors: {json.dumps(errors)}")
            
        # --- 5. Aggregate Results ---
        aggregated_result = _aggregate_results(per_tf_results)

        # --- 6. Compute PnL, Risk, and Recommendation (on Aggregated Result) ---
        pnl_rr = compute_pnl_rr(
            entry_price, 
            current_price, 
            aggregated_result["targets"], 
            aggregated_result["stoplosses"], 
            position_type
            # has_both_positions removed
        )
        
        # MODIFIED: Call to renamed function with fewer params
        primary_sl, primary_tp = get_primary_levels(
            entry_price, 
            aggregated_result["stoplosses"], 
            aggregated_result["targets"], 
            position_type
        )
        
        recommendation, rationale = generate_recommendation(
            aggregated_result["bias"],
            aggregated_result["confidence"],
            pnl_rr,
            position_type,
            entry_price,
            current_price,
            aggregated_result["patterns"] # Pass patterns/conflict flag
        )
        
        # --- 7. Build Final JSON Response (FLATTENED) ---
        
        # Add recommendation/rationale into the aggregated block
        aggregated_result["recommendation"] = recommendation
        aggregated_result["rationale"] = rationale # This is the "sentiment"
        
        # Clean up the internal 'conflict' flag from the patterns list
        if "conflict" in aggregated_result["patterns"]:
            aggregated_result["patterns"].remove("conflict")

        # Add PnL/Risk info
        aggregated_result["pnl_rr"] = pnl_rr
        # MODIFIED: Renamed this key
        aggregated_result["primary_levels"] = {
            "stoploss": primary_sl,
            "takeprofit": primary_tp
        }
        
        # Create the base response with request info
        response_content = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "coin": coin,
            "market": market,
            "entry_price": entry_price,
            "current_price": current_price,
            "position_type": position_type,
            "daily_trend": {
                "trend": daily_trend,
                "move_pct": round(daily_move_pct, 2)
            },
            "errors": errors if errors else None
        }
        
        # Merge the flattened aggregated_result into the response_content
        # This adds 'recommendation', 'rationale', 'confidence', 'targets', 'stoplosses', 'patterns', etc.
        response_content.update(aggregated_result)

        # REMOVED: 'per_tf' block from the response
        
        return JSONResponse(content=response_content)

    except Exception as e:
        logger.error(f"Error in advisory_pipeline: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    """
    Returns a 204 No Content response for favicon requests
    to prevent 404 errors in the logs.
    """
    return Response(status_code=status.HTTP_204_NO_CONTENT)





# --- NEW: Unit Test / Backtest Script Example ---
# (This would typically be in a separate file, e.g., `test_analysis.py`)

def run_backtest_from_csv(csv_path: str, params: dict):
    """
    Simulates the analysis pipeline using data from a CSV file.
    CSV must have columns: 'open_time', 'open', 'high', 'low', 'close', 'volume'
    """
    logger.info(f"--- Running Backtest on {csv_path} ---")
    
    try:
        df_full = pd.read_csv(csv_path)
        df_full['open_time'] = pd.to_datetime(df_full['open_time'])
        # Ensure correct types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_full[col] = pd.to_numeric(df_full[col], errors='coerce')
        df_full = df_full.dropna()
        logger.info(f"Loaded {len(df_full)} candles from CSV.")

    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    # --- Resample CSV into 1H, 4H, 1D ---
    # Assumes base CSV is 1m or 5m. Let's assume 1H for this example.
    # If base is 1H:
    df_1h = df_full
    
    # Resample to 4H
    df_4h = df_1h.set_index('open_time').resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()

    # Resample to 1D
    df_1d = df_1h.set_index('open_time').resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    if df_1h.empty or df_4h.empty or df_1d.empty:
        logger.error("Failed to resample data, check base CSV timeframe.")
        return

    data_map = {"1h": df_1h, "4h": df_4h, "1d": df_1d}
    timeframes_to_analyze = ["1h", "4h", "1d"]
    
    current_price = df_1h['close'].iloc[-1]
    
    # --- 1. Run Single-TF (Old Logic) ---
    logger.info("--- Testing Single-TF (4H only) ---")
    per_tf_results_single = {}
    try:
        tf_4h_result = analyze_single_timeframe(
            df=df_4h,
            current_price=current_price,
            entry_price=params["entry_price"],
            timeframe="4h",
            quantity=params["quantity"],
            extra_params={}
        )
        per_tf_results_single["4h"] = tf_4h_result
        agg_single = _aggregate_results(per_tf_results_single)
        logger.info(f"Single-TF (4H) Result: Bias={agg_single['bias']}, Conf={agg_single['confidence']}")
        # print(json.dumps(agg_single, indent=2, default=str))

    except Exception as e:
        logger.error(f"Failed single-TF analysis: {e}")

    # --- 2. Run Multi-TF (New Logic) ---
    logger.info("--- Testing Multi-TF (1H, 4H, 1D) ---")
    per_tf_results_multi = {}
    errors = {}
    
    for tf in timeframes_to_analyze:
        try:
            df = data_map[tf]
            tf_result = analyze_single_timeframe(
                df=df,
                current_price=current_price,
                entry_price=params["entry_price"],
                timeframe=tf,
                quantity=params["quantity"],
                extra_params={}
            )
            per_tf_results_multi[tf] = tf_result
        except Exception as e:
            logger.error(f"Failed MTF analysis for {tf}: {e}")
            errors[tf] = str(e)
            
    if per_tf_results_multi:
        agg_multi = _aggregate_results(per_tf_results_multi)
        logger.info(f"Multi-TF Result: Bias={agg_multi['bias']}, Conf={agg_multi['confidence']}")
        logger.info(f"Multi-TF Remarks: {agg_multi['remarks']}")
        # print(json.dumps(agg_multi, indent=2, default=str))
    else:
        logger.error("Failed all MTF analyses.")

    logger.info("--- Backtest Comparison Complete ---")
    # print("\n--- PER-TF DETAILS (Multi-TF) ---")
    # print(json.dumps(per_tf_results_multi, indent=2, default=str))


if __name__ == "__main__":
    # Example of how to run the backtest script
    # This block won't run when using uvicorn, it's for local testing
    
    # Check if we're in a test mode
    if os.environ.get("RUN_BACKTEST") == "true":
        test_params = {
            "entry_price": 60000,
            "quantity": 0.1,
            "position_type": "long"
        }
        # You would need a CSV file at this path
        # e.g., 'btc_1h_data.csv'
        csv_file_path = os.environ.get("BACKTEST_CSV_PATH", "btc_1h_data.csv") 
        
        if os.path.exists(csv_file_path):
             run_backtest_from_csv(csv_file_path, test_params)
        else:
            logger.warning(f"Backtest CSV not found at {csv_file_path}. Skipping backtest.")
            
    else:
        # Standard server run
        import uvicorn
        logger.info("Starting FastAPI server with uvicorn...")
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))