import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize Bybit (use your API keys if needed; public for historical)
exchange = ccxt.bybit({
    'apiKey': '',  # Optional for public data
    'secret': '',  # Optional
    'sandbox': False,
})

# Fetch 15m data for last 1 year
symbol = 'BTC/USDT'
timeframe = '15m'
since = exchange.parse8601((datetime.now() - timedelta(days=365)).isoformat())
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=None)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

print(f"Fetched {len(df)} 15m candles from {df.index[0]} to {df.index[-1]}")

# Indicators
def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_ma(series, window):
    return series.rolling(window=window).mean()

def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_supertrend(df, period=10, multiplier=3):
    hl2 = (df['high'] + df['low']) / 2
    atr = compute_atr(df, period)
    upper = hl2 + (multiplier * atr)
    lower = hl2 - (multiplier * atr)
    in_uptrend = pd.Series([True] * len(df), index=df.index)
    for i in range(1, len(df)):
        if df['close'].iloc[i] <= upper.iloc[i-1]:
            in_uptrend.iloc[i] = False
        elif df['close'].iloc[i] >= lower.iloc[i-1]:
            in_uptrend.iloc[i] = True
        else:
            in_uptrend.iloc[i] = in_uptrend.iloc[i-1]
            if in_uptrend.iloc[i] and lower.iloc[i] < lower.iloc[i-1]:
                lower.iloc[i] = lower.iloc[i-1]
            if not in_uptrend.iloc[i] and upper.iloc[i] > upper.iloc[i-1]:
                upper.iloc[i] = upper.iloc[i-1]
    return pd.Series(np.where(in_uptrend, 1, -1), index=df.index)

# MTF SuperTrend (resample for 1h/4h)
def get_mtf_supertrend(df):
    df1h = df.resample('1H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    df4h = df.resample('4H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    st15m = compute_supertrend(df)
    st1h = compute_supertrend(df1h).reindex(df.index, method='ffill')
    st4h = compute_supertrend(df4h).reindex(df.index, method='ffill')
    return 0.4 * st15m + 0.3 * st1h + 0.3 * st4h

# Enhanced Bias
def compute_bias(df_slice):
    close = df_slice['close']
    ema20 = compute_ema(close, 20).iloc[-1]
    ma50 = compute_ma(close, 50).iloc[-1]
    rsi = compute_rsi(close).iloc[-1]
    mtf_st = get_mtf_supertrend(df_slice).iloc[-1]
    
    bullish_score = 0.0
    bearish_score = 0.0
    
    if ema20 > ma50:
        bullish_score += 0.3
    else:
        bearish_score += 0.3
    if rsi > 50:
        bullish_score += 0.2
    else:
        bearish_score += 0.2
    if mtf_st > 0:
        bullish_score += 0.3 * (mtf_st / 1)  # Normalized
    else:
        bearish_score += 0.3 * abs(mtf_st / 1)
    
    total = bullish_score + bearish_score
    bullish_pct = (bullish_score / total * 100) if total > 0 else 50
    return bullish_pct, 100 - bullish_pct

# Simple Pivot S/R for TGT adjust
def get_sr(df_slice):
    high = df_slice['high'].rolling(50).max().iloc[-1]
    low = df_slice['low'].rolling(50).min().iloc[-1]
    return low, high  # Support, Resistance

# TGT/SL (your logic)
def get_tgt_sl(close, atr_val, direction, sr):
    support, resistance = sr
    sl = close - (2 * atr_val) if direction == 'long' else close + (2 * atr_val)
    tgt1 = min(resistance, close + (3.5 * atr_val)) if direction == 'long' else max(support, close - (3.5 * atr_val))
    tgt2 = close + (6 * atr_val) if direction == 'long' else close - (6 * atr_val)
    risk = abs(close - sl)
    reward1 = abs(tgt1 - close)
    if reward1 / risk < 1.5:
        tgt1 = close + (risk * 1.5) if direction == 'long' else close - (risk * 1.5)
    return tgt1, tgt2, sl

# Backtest
position_size = 10000  # $10k per trade
trades = []
position = None
for i in range(100, len(df)):  # Warm-up for indicators
    row = df.iloc[i]
    bullish_pct, bearish_pct = compute_bias(df.iloc[:i+1])
    sr = get_sr(df.iloc[:i+1])
    atr_val = compute_atr(df.iloc[:i+1]).iloc[-1]
    
    if bullish_pct > 50 and position is None:
        direction = 'long'
        entry = row['close']
        tgt1, tgt2, sl = get_tgt_sl(entry, atr_val, direction, sr)
        position = {'direction': direction, 'entry': entry, 'tgt1': tgt1, 'tgt2': tgt2, 'sl': sl, 'index': i}
    elif bearish_pct > 50 and position is None:
        direction = 'short'
        entry = row['close']
        tgt1, tgt2, sl = get_tgt_sl(entry, atr_val, direction, sr)
        position = {'direction': direction, 'entry': entry, 'tgt1': tgt1, 'tgt2': tgt2, 'sl': sl, 'index': i}
    
    if position:
        future_df = df.iloc[position['index']+1:]
        exit_price = None
        exit_type = None
        exit_index = None
        for j in range(len(future_df)):
            future_row = future_df.iloc[j]
            if position['direction'] == 'long':
                if future_row['low'] <= position['sl']:
                    exit_price = position['sl']
                    exit_type = 'SL'
                    exit_index = j
                    break
                if future_row['high'] >= position['tgt1']:
                    exit_price = position['tgt1']
                    exit_type = 'TGT1'
                    exit_index = j
                    break
                if future_row['high'] >= position['tgt2']:
                    exit_price = position['tgt2']
                    exit_type = 'TGT2'
                    exit_index = j
                    break
            else:
                if future_row['high'] >= position['sl']:
                    exit_price = position['sl']
                    exit_type = 'SL'
                    exit_index = j
                    break
                if future_row['low'] <= position['tgt1']:
                    exit_price = position['tgt1']
                    exit_type = 'TGT1'
                    exit_index = j
                    break
                if future_row['low'] <= position['tgt2']:
                    exit_price = position['tgt2']
                    exit_type = 'TGT2'
                    exit_index = j
                    break
        if exit_price:
            pnl = ((exit_price - position['entry']) / position['entry'] * 100) if position['direction'] == 'long' else ((position['entry'] - exit_price) / position['entry'] * 100)
            trades.append({
                'entry_time': df.index[position['index']],
                'exit_time': future_df.index[exit_index],
                'direction': position['direction'],
                'entry_price': position['entry'],
                'exit_price': exit_price,
                'pnl_pct': pnl,
                'exit_type': exit_type,
                'bullish_bias': bullish_pct if 'bullish_pct' in locals() else 0
            })
            position = None

# Results
if trades:
    df_trades = pd.DataFrame(trades)
    win_rate = (df_trades['pnl_pct'] > 0).mean() * 100
    total_pnl = df_trades['pnl_pct'].sum()
    num_trades = len(df_trades)
    profit_factor = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].sum() / abs(df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].sum()) if len(df_trades[df_trades['pnl_pct'] < 0]) > 0 else np.inf
    max_dd = (df_trades['pnl_pct'].cumsum().expanding().max() - df_trades['pnl_pct'].cumsum()).max()
    
    print(f"=== BACKTEST RESULTS (15m BTCUSDT, 1 Year, >50% Bias Threshold) ===")
    print(f"Number of trades: {num_trades}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Total P&L: {total_pnl:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print("\nSample Trades (first 10):")
    print(df_trades.head(10).to_string(index=False))
else:
    print("No trades generated - check data or thresholds.")

# Save to CSV
df_trades.to_csv('backtest_results.csv', index=False)
print("\nResults saved to 'backtest_results.csv'")