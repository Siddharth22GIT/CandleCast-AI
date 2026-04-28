import pandas as pd
import numpy as np
import glob
import os
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
import joblib

# =========================
# LOAD DATA
# =========================

files = glob.glob("data/*.csv")
df_list = []

for file in files:
    temp = pd.read_csv(file)
    temp.columns = [col.lower() for col in temp.columns]

    if 'close' not in temp.columns:
        continue

    temp['date'] = pd.to_datetime(temp.get('date', temp.iloc[:, 0]))
    temp['ticker'] = temp.get('ticker', file.split("\\")[-1].replace(".csv", ""))

    cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'marketcap']
    for col in cols:
        if col not in temp.columns:
            temp[col] = np.nan

    df_list.append(temp[cols])

df = pd.concat(df_list, ignore_index=True)
df = df.sort_values(['ticker', 'date'])

print("✅ Cleaned Shape:", df.shape)

# =========================
# CLEAN
# =========================

df = df[df['volume'] > 0]

# =========================
# FEATURES
# =========================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['return'] = df.groupby('ticker')['close'].pct_change()

df['ma7'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(7).mean())
df['ma30'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(30).mean())

df['volatility'] = df.groupby('ticker')['return'].transform(lambda x: x.rolling(7).std())

df['rsi'] = df.groupby('ticker')['close'].transform(compute_rsi)

df['ema12'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=12).mean())
df['ema26'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=26).mean())
df['macd'] = df['ema12'] - df['ema26']

df['vol_ma7'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(7).mean())
df['vol_ratio'] = df['volume'] / df['vol_ma7']

df['momentum_3'] = df.groupby('ticker')['close'].transform(lambda x: x / x.shift(3) - 1)
df['momentum_7'] = df.groupby('ticker')['close'].transform(lambda x: x / x.shift(7) - 1)

# =========================
# FILTERS
# =========================

df = df[df['ma7'] > df['ma30']]
df = df[df['vol_ratio'] > 0.8]

# =========================
# TARGET
# =========================

df['future_close'] = df.groupby('ticker')['close'].shift(-1)
df['target'] = (df['future_close'] > df['close'] * 1.03).astype(int)

df = df.dropna()

print("✅ After Processing:", df.shape)

# =========================
# FEATURES
# =========================

features = [
    'return', 'ma7', 'ma30', 'volatility',
    'rsi', 'macd',
    'vol_ratio',
    'momentum_3', 'momentum_7'
]

X = df[features]
y = df['target']

# =========================
# TIME SPLIT FIRST ✅
# =========================

split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("📊 Train:", X_train.shape)
print("📊 Test:", X_test.shape)

# =========================
# BALANCE ONLY TRAINING SET ✅
# =========================

train_df = X_train.copy()
train_df['target'] = y_train

df_majority = train_df[train_df.target == 0]
df_minority = train_df[train_df.target == 1]

df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=len(df_minority),
    random_state=42
)

train_balanced = pd.concat([df_majority_downsampled, df_minority])

X_train = train_balanced[features]
y_train = train_balanced['target']

print("⚖️ Balanced Train:", X_train.shape)

# =========================
# MODEL
# =========================

model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================

y_pred = model.predict(X_test)

print("\n📊 FINAL REPORT:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE
# =========================

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/crypto_xgb.pkl")

print("\n💾 Model saved!")