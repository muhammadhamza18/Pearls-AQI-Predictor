# deep_diag.py
"""
Deep diagnostic to find why all models give negative R2
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

CSV_PATH    = "final_data/karachi_hopsworks_upload.csv"
TARGET_COLS = ["target_aqi_1d", "target_aqi_2d", "target_aqi_3d"]
TEST_SIZE   = 0.2

df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("="*60)
print("DEEP DIAGNOSTIC")
print("="*60)

# ── 1. Check train vs test distributions ─────────────────────
split_idx = int(len(df) * (1 - TEST_SIZE))
train_df  = df.iloc[:split_idx]
test_df   = df.iloc[split_idx:]

print("\n1. TRAIN vs TEST DISTRIBUTION")
print("-"*60)
print("  " + "Column".ljust(25) +
      "Train Mean".rjust(12) +
      "Test Mean".rjust(12) +
      "Train Std".rjust(12) +
      "Test Std".rjust(12))
print("  " + "-"*60)

for col in df.columns:
    if col == "timestamp":
        continue
    tm = round(train_df[col].mean(), 2)
    vm = round(test_df[col].mean(),  2)
    ts = round(train_df[col].std(),  2)
    vs = round(test_df[col].std(),   2)
    diff = abs(tm - vm)
    flag = " <<< BIG SHIFT" if diff > 30 else ""
    print("  " + str(col).ljust(25) +
          str(tm).rjust(12) +
          str(vm).rjust(12) +
          str(ts).rjust(12) +
          str(vs).rjust(12) + flag)

print("\n  Train date range: " + str(train_df["timestamp"].min()) +
      " to " + str(train_df["timestamp"].max()))
print("  Test  date range: " + str(test_df["timestamp"].min()) +
      " to " + str(test_df["timestamp"].max()))

# ── 2. Check feature vs target alignment ─────────────────────
print("\n\n2. FEATURE vs TARGET ALIGNMENT CHECK")
print("-"*60)
print("  Does aqi at row N match target_aqi_1d at row N-24?")
print("")

# Check if target is properly shifted
# target_aqi_1d should = aqi 24 hours later
for i in [100, 200, 300, 400, 500]:
    curr_aqi    = df.iloc[i]["aqi"]
    target_1d   = df.iloc[i]["target_aqi_1d"]
    future_aqi  = df.iloc[i + 24]["aqi"] if i + 24 < len(df) else None

    print("  Row " + str(i) + ":")
    print("    current aqi      = " + str(curr_aqi))
    print("    target_aqi_1d    = " + str(target_1d))
    print("    aqi 24 rows later= " + str(future_aqi))
    match = "MATCH" if future_aqi is not None and abs(target_1d - future_aqi) < 5 else "MISMATCH!"
    print("    Status           = " + match)
    print("")

# ── 3. Check if target leaks into features ───────────────────
print("\n3. CORRELATION: FEATURES vs TARGETS")
print("-"*60)

drop_cols  = ["timestamp"]
feature_df = df.drop(columns=drop_cols)
corr_matrix= feature_df.corr()

print("\n  Correlation with target_aqi_1d (should be high ~0.7+):")
corr_1d = corr_matrix["target_aqi_1d"].drop(TARGET_COLS).sort_values(ascending=False)
for feat, val in corr_1d.items():
    bar    = "#" * int(abs(val) * 20)
    status = "GOOD" if abs(val) > 0.5 else "WEAK"
    print("    " + str(feat).ljust(25) +
          str(round(val, 4)).rjust(8) +
          "  " + bar + "  [" + status + "]")

# ── 4. Simple baseline test ───────────────────────────────────
print("\n\n4. BASELINE TEST (predict mean of train)")
print("-"*60)
print("  If R2 > 0 here, features have predictive power")
print("")

X = df.drop(columns=TARGET_COLS + ["timestamp"])
y = df[TARGET_COLS].values

split_idx  = int(len(X) * (1 - TEST_SIZE))
X_train    = X.iloc[:split_idx].values
X_test     = X.iloc[split_idx:].values
y_train    = y[:split_idx]
y_test     = y[split_idx:]

# Baseline: always predict train mean
for i, col in enumerate(TARGET_COLS):
    train_mean    = y_train[:, i].mean()
    y_pred_mean   = np.full(len(y_test), train_mean)
    r2_baseline   = r2_score(y_test[:, i], y_pred_mean)
    r2_perfect    = 1.0

    print("  " + col + ":")
    print("    train mean  = " + str(round(train_mean, 2)))
    print("    test mean   = " + str(round(y_test[:, i].mean(), 2)))
    print("    test std    = " + str(round(y_test[:, i].std(), 2)))
    print("    baseline R2 = " + str(round(r2_baseline, 4)) +
          " (predicting train mean every time)")
    diff = round(y_test[:, i].mean() - train_mean, 2)
    print("    mean shift  = " + str(diff) +
          (" <<< TEST SET HAS DIFFERENT DISTRIBUTION!" if abs(diff) > 20 else " OK"))
    print("")

# ── 5. Quick sanity model ─────────────────────────────────────
print("\n5. QUICK SANITY MODEL (just use aqi to predict target)")
print("-"*60)
print("  Training linear model: target = a * aqi + b")
print("")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Use just aqi column (column index 3)
aqi_idx   = list(X.columns).index("aqi")
X_tr_aqi  = X_train_sc[:, aqi_idx].reshape(-1, 1)
X_te_aqi  = X_test_sc[:, aqi_idx].reshape(-1, 1)

from sklearn.linear_model import LinearRegression

for i, col in enumerate(TARGET_COLS):
    lr = LinearRegression()
    lr.fit(X_tr_aqi, y_train[:, i])
    y_pred = lr.predict(X_te_aqi)
    r2     = r2_score(y_test[:, i], y_pred)
    print("  " + col + " using only aqi: R2 = " + str(round(r2, 4)))

print("")

# ── 6. Quick sanity with ALL features ────────────────────────
print("\n6. QUICK SANITY MODEL (all 12 features, linear)")
print("-"*60)

from sklearn.linear_model import LinearRegression

for i, col in enumerate(TARGET_COLS):
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train[:, i])
    y_pred = lr.predict(X_test_sc)
    r2     = r2_score(y_test[:, i], y_pred)
    print("  " + col + " Linear R2 = " + str(round(r2, 4)))

print("")

# ── 7. Check test period AQI pattern ─────────────────────────
print("\n7. TEST PERIOD ANALYSIS")
print("-"*60)
print("  What does AQI look like in test period?")
print("")

test_aqi = test_df["aqi"]
print("  Test AQI stats:")
print("    Min    : " + str(round(test_aqi.min(), 2)))
print("    Max    : " + str(round(test_aqi.max(), 2)))
print("    Mean   : " + str(round(test_aqi.mean(), 2)))
print("    Std    : " + str(round(test_aqi.std(), 2)))

train_aqi = train_df["aqi"]
print("\n  Train AQI stats:")
print("    Min    : " + str(round(train_aqi.min(), 2)))
print("    Max    : " + str(round(train_aqi.max(), 2)))
print("    Mean   : " + str(round(train_aqi.mean(), 2)))
print("    Std    : " + str(round(train_aqi.std(), 2)))

print("\n  Test targets:")
for col in TARGET_COLS:
    test_vals = test_df[col]
    train_vals= train_df[col]
    print("  " + col + ":")
    print("    test  mean=" + str(round(test_vals.mean(), 2)) +
          "  std=" + str(round(test_vals.std(), 2)))
    print("    train mean=" + str(round(train_vals.mean(), 2)) +
          "  std=" + str(round(train_vals.std(), 2)))
    diff = abs(test_vals.mean() - train_vals.mean())
    print("    mean diff =" + str(round(diff, 2)) +
          (" <<< DISTRIBUTION SHIFT!" if diff > 25 else " OK"))

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\nShare output above to identify root cause!")