import requests
import pandas as pd
import numpy as np

BASE_URL = "https://api.coingecko.com/api/v3"

def get_market_data(coin_id="bitcoin", days=30):
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }

    res = requests.get(url, params=params)
    data = res.json()

    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])

    df = prices.merge(volumes, on="timestamp")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("date")

    return df


# ---------- FEATURE ENGINEERING ----------

def compute_features(df):
    df["return"] = df["price"].pct_change()

    df["ma7"] = df["price"].rolling(7).mean()
    df["ma30"] = df["price"].rolling(30).mean()

    df["volatility"] = df["return"].rolling(7).std()

    # RSI
    delta = df["price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["price"].ewm(span=12).mean()
    ema26 = df["price"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26

    # Volume ratio
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(7).mean()

    # Momentum
    df["momentum_3"] = df["price"].pct_change(3)
    df["momentum_7"] = df["price"].pct_change(7)

    df = df.dropna()

    return df


def get_latest_features(coin_id):
    df = get_market_data(coin_id)
    df = compute_features(df)

    latest = df.iloc[-1]

    return {
        "return": float(latest["return"]),
        "ma7": float(latest["ma7"]),
        "ma30": float(latest["ma30"]),
        "volatility": float(latest["volatility"]),
        "rsi": float(latest["rsi"]),
        "macd": float(latest["macd"]),
        "vol_ratio": float(latest["vol_ratio"]),
        "momentum_3": float(latest["momentum_3"]),
        "momentum_7": float(latest["momentum_7"]),
    }