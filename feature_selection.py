# feature_selection_final.py
"""
Select TOP 12 features (score >= 0.44) + 3 targets
Save final CSV for Hopsworks upload
"""

import os
import pandas as pd

# ============================================================
# LOAD CLEANED DATA
# ============================================================
print("📂 Loading cleaned data...")

df = pd.read_csv('cleaned_data2/karachi_cleaned2.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"✅ Loaded: {df.shape}")

# ============================================================
# TOP 12 FEATURES (score >= 0.44)
# ============================================================
top_12_features = [
    'aqi_rolling_max_24h',      # rank 1  score: 0.986 ⭐
    'pm10',                     # rank 2  score: 0.535
    'pm25',                     # rank 3  score: 0.509
    'aqi',                      # rank 4  score: 0.505
    'aqi_rolling_mean_3h',      # rank 5  score: 0.498
    'aqi_lag_1h',               # rank 6  score: 0.496
    'aqi_rolling_mean_6h',      # rank 7  score: 0.492
    'co',                       # rank 8  score: 0.489
    'aqi_rolling_mean_12h',     # rank 9  score: 0.486
    'aqi_lag_3h',               # rank 10 score: 0.484
    'o3',                       # rank 11 score: 0.484
    'aqi_lag_6h',               # rank 12 score: 0.467
]

# 3 Targets
target_cols = [
    'target_aqi_1d',
    'target_aqi_2d',
    'target_aqi_3d',
]

# ============================================================
# DROPPED FEATURES (rank 13-20)
# ============================================================
dropped_features = {
    'aqi_rolling_mean_24h':  0.465,
    'so2':                   0.447,
    'aqi_lag_12h':           0.445,
    'aqi_rolling_min_24h':   0.441,
    'pm25_rolling_mean_24h': 0.433,
    'no2':                   0.431,
    'aqi_lag_24h':           0.413,
    'pm25_lag_24h':          0.374,
}

print("\n❌ DROPPED FEATURES (score < 0.467):")
print("-"*45)
for feat, score in dropped_features.items():
    print(f"  ❌ {feat:<30} score: {score}")

# ============================================================
# VERIFY COLUMNS EXIST                ← FIXED: top_12_features
# ============================================================
print("\n🔍 Verifying columns...")
print("-"*50)

for feat in top_12_features:          # ← FIXED
    status = "✅" if feat in df.columns else "❌ MISSING"
    print(f"  {status} {feat}")

print()
for target in target_cols:
    status = "🎯" if target in df.columns else "❌ MISSING"
    print(f"  {status} {target}")

# ============================================================
# SELECT FINAL COLUMNS               ← FIXED: top_12_features
# Order: timestamp → 12 features → 3 targets
# ============================================================
available_features = [f for f in top_12_features if f in df.columns]  # ← FIXED
available_targets  = [t for t in target_cols     if t in df.columns]

final_cols = ['timestamp'] + available_features + available_targets
df_final   = df[final_cols].copy()

# ============================================================
# CLEAN
# ============================================================
print("\n🧹 Cleaning...")

before   = len(df_final)
df_final = df_final.dropna(subset=available_targets)
print(f"  ✅ Dropped {before - len(df_final)} rows with missing targets")
print(f"  ✅ Remaining: {len(df_final)} rows")

missing = df_final.isnull().sum().sum()
if missing > 0:
    df_final = df_final.ffill().bfill()
    print(f"  ✅ Filled {missing} missing values")
else:
    print(f"  ✅ No missing values!")

# ============================================================
# SAVE CSV
# ============================================================
os.makedirs('final_data', exist_ok=True)

output_file = 'final_data/karachi_hopsworks_upload.csv'
df_final.to_csv(output_file, index=False)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("✅ FINAL CSV SAVED!")
print("="*60)
print(f"\n📁 File  : {output_file}")
print(f"📊 Rows  : {len(df_final)}")
print(f"📊 Cols  : {len(df_final.columns)}")

print(f"\n📋 Final Column List:")
print("-"*60)

scores = {
    'aqi_rolling_max_24h':  0.986,
    'pm10':                 0.535,
    'pm25':                 0.509,
    'aqi':                  0.505,
    'aqi_rolling_mean_3h':  0.498,
    'aqi_lag_1h':           0.496,
    'aqi_rolling_mean_6h':  0.492,
    'co':                   0.489,
    'aqi_rolling_mean_12h': 0.486,
    'aqi_lag_3h':           0.484,
    'o3':                   0.484,
    'aqi_lag_6h':           0.467,
}

for i, col in enumerate(df_final.columns, 1):
    if col == 'timestamp':
        tag   = "🔑 PRIMARY KEY"
        score = "  -  "
    elif col in available_targets:
        tag   = "🎯 TARGET"
        score = "  -  "
    elif col in ['aqi', 'pm25', 'pm10', 'o3', 'co']:
        tag   = "💨 POLLUTANT"
        score = str(scores.get(col, '-'))
    elif 'lag' in col:
        tag   = "⏰ LAG"
        score = str(scores.get(col, '-'))
    elif 'rolling' in col:
        tag   = "📊 ROLLING"
        score = str(scores.get(col, '-'))
    else:
        tag   = "📌 FEATURE"
        score = "  -  "

    print(f"  {i:2d}. {col:<30} {score:<8} {tag}")

print(f"\n📊 Structure:")
print(f"  🔑 Identifier : 1   col  → timestamp")
print(f"  📊 Features   : {len(available_features):2d}  cols → model input  (X)")
print(f"  🎯 Targets    : {len(available_targets):2d}   cols → model output (y)")
print(f"  📦 Total      : {len(df_final.columns):2d}  cols")

print(f"\n📈 Quick Stats:")
print(f"  AQI mean       : {df_final['aqi'].mean():.1f}")
print(f"  PM2.5 mean     : {df_final['pm25'].mean():.1f}")
print(f"  Target 1d mean : {df_final['target_aqi_1d'].mean():.1f}")
print(f"  Target 3d mean : {df_final['target_aqi_3d'].mean():.1f}")

print(f"\n🎯 Next Step:")
print(f"  python upload_to_hopsworks.py")

## 📊 **Final Output (16 columns):**

#  1. 🔑 timestamp
# ──────────────────────────────────────
#  2. 📊 aqi_rolling_max_24h    0.986 ⭐
#  3. 💨 pm10                   0.535
#  4. 💨 pm25                   0.509
#  5. 💨 aqi                    0.505
#  6. 📊 aqi_rolling_mean_3h    0.498
#  7. ⏰ aqi_lag_1h             0.496
#  8. 📊 aqi_rolling_mean_6h    0.492
#  9. 💨 co                     0.489
# 10. 📊 aqi_rolling_mean_12h   0.486
# 11. ⏰ aqi_lag_3h             0.484
# 12. 💨 o3                     0.484
# 13. ⏰ aqi_lag_6h             0.467
# ──────────────────────────────────────
# 14. 🎯 target_aqi_1d
# 15. 🎯 target_aqi_2d
# 16. 🎯 target_aqi_3d
# ──────────────────────────────────────
# TOTAL = 16 columns