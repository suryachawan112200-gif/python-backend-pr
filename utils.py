# utils.py
import pandas as pd
import numpy as np

def get_target_stop_loss(entry_price, risk_percent=1, reward_percent=2):
    """
    Calculate target and stop loss prices based on entry price and risk/reward percent.
    :param entry_price: Price where position is entered
    :param risk_percent: Stop loss distance in percent
    :param reward_percent: Target distance in percent
    :return: dict with target and stop_loss values
    """
    stop_loss = entry_price * (1 - risk_percent / 100)
    target = entry_price * (1 + reward_percent / 100)
    return {"target": round(target,2), "stop_loss": round(stop_loss,2)}


def calculate_support_resistance(df: pd.DataFrame, window: int = 20):
    """
    Calculates basic support and resistance levels based on historical highs/lows.
    This is a simplified approach; real S&R uses more sophisticated methods.
    """
    if df.empty or len(df) < window:
        return {"support_levels": [], "resistance_levels": []}

    highs = df['high'].values
    lows = df['low'].values

    # Simple approach: Find recent significant swing highs and lows
    support_levels = []
    resistance_levels = []

    # Iterate through the data to find local peaks and troughs
    for i in range(window, len(df) - window):
        # Check for resistance (peak)
        if highs[i] == max(highs[i-window : i+window+1]):
            resistance_levels.append(highs[i])
        # Check for support (trough)
        if lows[i] == min(lows[i-window : i+window+1]):
            support_levels.append(lows[i])

    # Add the highest high and lowest low from recent data as potential levels
    recent_high = df['high'].iloc[-window:].max()
    recent_low = df['low'].iloc[-window:].min()
    
    if recent_high not in resistance_levels: resistance_levels.append(recent_high)
    if recent_low not in support_levels: support_levels.append(recent_low)
    
    # Sort and remove duplicates
    support_levels = sorted(list(set([round(x, 2) for x in support_levels])))
    resistance_levels = sorted(list(set([round(x, 2) for x in resistance_levels])))