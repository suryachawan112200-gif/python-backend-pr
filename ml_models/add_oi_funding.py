import pandas as pd
from binance.client import Client
import requests
from datetime import datetime
import time
import numpy as np

client = Client()  # No key for historical

# Coins (your list)
coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT"]

for coin in coins:
    try:
        # Load existing enriched CSV
        filename = f"data/{coin.lower()}_futures_15m_1yr_enriched.csv"
        df = pd.read_csv(filename)
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.sort_values('open_time').reset_index(drop=True)

        # Fetch OI (volume proxy from klines – for true OI, use futures API)
        klines = client.get_historical_klines(coin, Client.KLINE_INTERVAL_15MINUTE, "1 Oct, 2024", "30 Oct, 2025")
        df_oi = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
        df_oi['open_time'] = pd.to_datetime(df_oi['open_time'], unit='ms')
        df_oi = df_oi[['open_time', 'volume']]
        df_oi['oi_change_pct'] = df_oi['volume'].pct_change() * 100  # Proxy OI change
        df = df.merge(df_oi, on='open_time', how='left', suffixes=('', '_oi'))
        df['oi_change_pct'].fillna(0, inplace=True)

        # Fetch Funding Rate (Coinglass API – free)
        url = f"https://api.coingecko.com/api/v3/coins/{coin.lower().replace('usdt', '')}/ohlc?vs_currency=usd&days=365"  # Proxy for funding via OHLC
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df_funding = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df_funding['timestamp'] = pd.to_datetime(df_funding['timestamp'], unit='ms')
            df_funding['funding_rate'] = 0.01  # Mock – replace with real from Coinglass if API key
            df_funding['funding_bias'] = np.where(df_funding['funding_rate'] > 0.01, 1, -1)  # >0.01 = bull bias
            df = df.merge(df_funding[['timestamp', 'funding_bias']], left_on='open_time', right_on='timestamp', how='left')
            df['funding_bias'].fillna(0, inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
        else:
            df['funding_bias'] = 0  # Fallback

        # Save updated file
        new_filename = f"data/{coin.lower()}_enriched_with_oi_funding.csv"
        df.to_csv(new_filename, index=False)
        print(f"Added OI/funding to {coin}: {len(df)} rows. Saved to {new_filename}")

        time.sleep(1)  # Pause

    except Exception as e:
        print(f"Error for {coin}: {e}")

print("All updated! Use new files for prep.")