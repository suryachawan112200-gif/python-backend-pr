import requests
import pandas as pd

# ---------- 1. Binance ----------
def get_binance_data():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url).json()
    coins = []
    for d in data:
        coins.append({
            "exchange": "Binance",
            "symbol": d["symbol"],
            "baseAsset": d["symbol"][:-4] if d["symbol"].endswith("USDT") else d["symbol"],
            "quoteAsset": d["symbol"][-4:] if d["symbol"].endswith("USDT") else "Other",
            "volume": float(d["quoteVolume"])
        })
    return pd.DataFrame(coins)

# ---------- 2. Bybit ----------
def get_bybit_data():
    url = "https://api.bybit.com/v5/market/tickers?category=linear"
    data = requests.get(url).json()["result"]["list"]
    coins = []
    for d in data:
        coins.append({
            "exchange": "Bybit",
            "symbol": d["symbol"],
            "baseAsset": d["symbol"].replace("USDT", ""),
            "quoteAsset": "USDT",
            "volume": float(d["turnover24h"])
        })
    return pd.DataFrame(coins)

# ---------- 3. Combine & Clean ----------
def categorize(df):
    df = df.sort_values("volume", ascending=False)
    top = df.head(30)
    mid = df.iloc[30:100]
    low = df.iloc[100:]
    top["category"] = "Most Traded"
    mid["category"] = "Moderate"
    low["category"] = "Low Volume"
    return pd.concat([top, mid, low])

# ---------- 4. Run ----------
binance_df = get_binance_data()
bybit_df = get_bybit_data()

combined = pd.concat([binance_df, bybit_df])
combined = combined.drop_duplicates(subset=["symbol"])
categorized = categorize(combined)

# ---------- 5. Export ----------
categorized.to_csv("coins_summary.csv", index=False)
print("✅ coins_summary.csv created successfully!")
print(categorized.head(15))