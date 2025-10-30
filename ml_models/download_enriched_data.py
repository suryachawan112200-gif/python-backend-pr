import pandas as pd
import numpy as np  # Fixed: Added numpy for np.nan
from binance.client import Client
from datetime import datetime
import time
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Client (no key for historical)
client = Client()

# Coin to download (change for next)
coin = "AVAXUSDT"  # Change to "ETHUSDT" etc. for others

# Dates (1 year)
start_date = "1 Oct, 2024"
end_date = "30 Oct, 2025"
chunk_size = 2000  # Safe for 15m
total_target = 15000
break_time = 5  # 5s pause

# Your indicator functions (copy from main.py)
def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_ma(series, window):
    return series.rolling(window=window).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)

def compute_atr(df, period=14):
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_adx(df, period=14):
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
    return dx.ewm(span=period, adjust=False).mean()

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower

# Chunked fetch
all_klines = []
current_start = start_date
chunks_fetched = 0

while len(all_klines) < total_target and chunks_fetched < 20:  # Safety
    try:
        klines = client.get_historical_klines(coin, Client.KLINE_INTERVAL_15MINUTE, current_start, end_date, limit=chunk_size)
        if not klines:
            break
        all_klines.extend(klines)
        logger.info(f"Fetched chunk {chunks_fetched+1}: {len(klines)} bars, total {len(all_klines)}")
        chunks_fetched += 1
        time.sleep(break_time)  # 5s break
        # Advance start to last kline date
        if klines:
            last_kline_time = klines[-1][0]
            current_start = datetime.fromtimestamp(last_kline_time / 1000).strftime('%d %b %Y')
    except Exception as e:
        logger.error(f"Chunk error: {e}")
        time.sleep(break_time * 2)
        break

# Build DF
df = pd.DataFrame(all_klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df = df.sort_values('open_time').reset_index(drop=True)
df = df.tail(total_target)  # Cap 15k

if len(df) < 100:
    logger.warning(f"Insufficient data for {coin}")
else:
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    volume = df['volume']
    
    # Compute 8 indicators
    df['ema9'] = compute_ema(close, 9)
    df['ma50'] = compute_ma(close, 50)
    df['rsi'] = compute_rsi(close)
    df['atr'] = compute_atr(df)
    df['adx'] = compute_adx(df)
    sma20, bb_upper, bb_lower = compute_bollinger_bands(close)
    df['bb_sma20'] = sma20
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_sma20']
    df['vol_sma20'] = compute_ma(volume, 20)
    df['vol_ratio'] = volume / df['vol_sma20']
    
    # Pattern features
    range_val = high - low
    df['body_ratio'] = abs(close - open_) / range_val.replace(0, np.nan)  # Fixed: np.nan now works
    df['upper_shadow_ratio'] = (high - np.maximum(open_, close)) / range_val.replace(0, np.nan)
    df['lower_shadow_ratio'] = (np.minimum(open_, close) - low) / range_val.replace(0, np.nan)
    
    # Drop NaNs
    df.dropna(inplace=True)
    
    # Save
    filename = f"data/{coin.lower()}_futures_15m_1yr_enriched.csv"
    df.to_csv(filename, index=False)
    logger.info(f"Saved {len(df)} enriched bars for {coin} to {filename}")

print("Download complete! Check ml_models/data/.")