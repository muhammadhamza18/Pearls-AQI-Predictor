# upload_to_hopsworks.py
"""
Upload Karachi AQI Features to Hopsworks Feature Store
Project: karachi_aqipred
"""

import os
import sys
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
from datetime import datetime

# ============================================================
# STEP 1: LOAD API KEY
# ============================================================
print("\n" + "="*60)
print("🔑 STEP 1: LOADING API KEY")
print("="*60)

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    print("\n❌ HOPSWORKS_API_KEY not found in .env!")
    print("\n💡 Add this to your .env file:")
    print("   HOPSWORKS_API_KEY=your_key_here")
    print("\n   Get key from:")
    print("   https://app.hopsworks.ai → Profile → Settings → API Keys")
    sys.exit(1)

masked = HOPSWORKS_API_KEY[:6] + "****" + HOPSWORKS_API_KEY[-4:]
print(f"\n  ✅ API Key found: {masked}")

# ============================================================
# STEP 2: CONNECT TO HOPSWORKS
# ============================================================
print("\n" + "="*60)
print("🔗 STEP 2: CONNECTING TO HOPSWORKS")
print("="*60)

print("\n  📡 Connecting to project: karachi_aqipred ...")
print("  ⏳ Please wait 30-60 seconds...\n")

try:
    project = hopsworks.login(
        project="karachi_aqipred",
        api_key_value=HOPSWORKS_API_KEY
    )

    print(f"\n  ✅ Connected successfully!")
    print(f"  📋 Project : {project.name}")

except Exception as e:
    print(f"\n❌ Connection failed!")
    print(f"   Error: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Check API key is correct in .env")
    print("   2. Check internet connection")
    print("   3. Confirm project name is 'karachi_aqipred'")
    print("      at https://app.hopsworks.ai")
    sys.exit(1)

# ============================================================
# STEP 3: LOAD CSV
# ============================================================
print("\n" + "="*60)
print("📂 STEP 3: LOADING FINAL CSV")
print("="*60)

CSV_PATH = "final_data/karachi_hopsworks_upload.csv"

if not os.path.exists(CSV_PATH):
    print(f"\n❌ File not found: {CSV_PATH}")
    print("\n💡 Run feature_selection_final.py first!")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

print(f"\n  ✅ Loaded: {CSV_PATH}")
print(f"  📊 Rows   : {len(df)}")
print(f"  📊 Cols   : {len(df.columns)}")
print(f"\n  📋 Columns:")
for i, col in enumerate(df.columns, 1):
    tag = "🎯" if "target" in col else \
          "🔑" if col == "timestamp" else "📊"
    print(f"    {tag} {i:2d}. {col}")

# ============================================================
# STEP 4: PREPARE DATAFRAME
# ============================================================
print("\n" + "="*60)
print("🔧 STEP 4: PREPARING DATAFRAME")
print("="*60)

df = df.copy()

# Convert timestamp
print("\n  📅 Converting timestamp...")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"    ✅ {df['timestamp'].min()} → {df['timestamp'].max()}")

# Remove duplicates
print("\n  🔍 Checking duplicates...")
dupes = df.duplicated(subset=["timestamp"]).sum()
if dupes > 0:
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    print(f"    ⚠️  Removed {dupes} duplicates")
else:
    print(f"    ✅ No duplicates")

# Missing values
print("\n  🔍 Checking missing values...")
missing = df.isnull().sum().sum()
if missing > 0:
    df = df.ffill().bfill()
    print(f"    ✅ Filled {missing} missing values")
else:
    print(f"    ✅ No missing values")

# Drop rows with missing targets
print("\n  🗑️  Checking target columns...")
target_cols = ["target_aqi_1d", "target_aqi_2d", "target_aqi_3d"]
before = len(df)
df = df.dropna(subset=target_cols)
print(f"    ✅ {before - len(df)} rows dropped → {len(df)} rows remaining")

# Fix data types
print("\n  🔧 Fixing data types...")
float_cols = df.select_dtypes(include=["float64"]).columns
df[float_cols] = df[float_cols].astype("float32")
print(f"    ✅ {len(float_cols)} float64 → float32")

int_cols = df.select_dtypes(include=["int64"]).columns
df[int_cols] = df[int_cols].astype("int32")
print(f"    ✅ {len(int_cols)} int64  → int32")

print(f"\n  📊 Final shape  : {df.shape}")
print(f"  🔢 Missing vals : {df.isnull().sum().sum()}")
print(f"  💾 Memory       : {df.memory_usage(deep=True).sum()/1024:.1f} KB")

# ============================================================
# STEP 5: CONNECT TO FEATURE STORE
# ============================================================
print("\n" + "="*60)
print("🏪 STEP 5: CONNECTING TO FEATURE STORE")
print("="*60)

try:
    fs = project.get_feature_store()
    print(f"\n  ✅ Feature Store connected: {fs.name}")

except Exception as e:
    print(f"\n❌ Feature Store failed: {e}")
    sys.exit(1)

# ============================================================
# STEP 6: CREATE FEATURE GROUP
# ============================================================
print("\n" + "="*60)
print("📦 STEP 6: CREATING FEATURE GROUP")
print("="*60)

FG_NAME    = "karachi_aqi_features"
FG_VERSION = 1

print(f"\n  📦 Name    : {FG_NAME}")
print(f"  🔢 Version : {FG_VERSION}")
print(f"  🔑 PK      : timestamp")

try:
    feature_group = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        description=(
            "Karachi AQI 12 selected features + 3 targets. "
            "Source: OpenWeatherMap. "
            f"Uploaded: {datetime.now().strftime('%Y-%m-%d')}"
        ),
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False,
    )

    print(f"\n  ✅ Feature Group ready!")

except Exception as e:
    print(f"\n❌ Feature Group failed: {e}")
    sys.exit(1)

# ============================================================
# STEP 7: INSERT DATA
# ============================================================
print("\n" + "="*60)
print("📤 STEP 7: UPLOADING DATA")
print("="*60)

print(f"\n  📤 Uploading {len(df)} rows × {len(df.columns)} columns...")
print(f"  ⏳ This takes 2-5 minutes — do NOT close terminal!\n")

try:
    feature_group.insert(
        df,
        write_options={"wait_for_job": True}
    )

    print(f"\n  ✅ Upload successful!")
    print(f"  📊 Rows     : {len(df)}")
    print(f"  📊 Features : {len(df.columns) - 4}")  # minus timestamp + 3 targets
    print(f"  🎯 Targets  : 3")

except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    print("\n💡 Common fixes:")
    print("   1. Check internet connection")
    print("   2. Check Hopsworks quota:")
    print("      https://app.hopsworks.ai → Project Settings → Quotas")
    sys.exit(1)

# ============================================================
# STEP 8: CREATE FEATURE VIEW
# ============================================================
print("\n" + "="*60)
print("👁️  STEP 8: CREATING FEATURE VIEW")
print("="*60)

FV_NAME    = "karachi_aqi_fv"
FV_VERSION = 1

print(f"\n  👁️  Name    : {FV_NAME}")
print(f"  🔢 Version : {FV_VERSION}")
print(f"  🎯 Labels  : target_aqi_1d, target_aqi_2d, target_aqi_3d")

try:
    feature_view = fs.get_or_create_feature_view(
        name=FV_NAME,
        version=FV_VERSION,
        description="Feature view for Karachi AQI 3-day prediction",
        labels=[
            "target_aqi_1d",
            "target_aqi_2d",
            "target_aqi_3d",
        ],
        query=feature_group.select_all()
    )

    print(f"\n  ✅ Feature View created!")

except Exception as e:
    print(f"\n❌ Feature View failed: {e}")
    print("   ⚠️  Data WAS uploaded successfully!")
    print("   Create Feature View manually in Hopsworks UI")
    sys.exit(1)

# ============================================================
# STEP 9: VERIFY
# ============================================================
print("\n" + "="*60)
print("🔍 STEP 9: VERIFYING UPLOAD")
print("="*60)

print("\n  📥 Reading back from Feature Store...")

try:
    verify_df = feature_group.read()

    print(f"\n  ✅ Verification passed!")
    print(f"  📊 Rows in store : {len(verify_df)}")
    print(f"  📊 Columns       : {len(verify_df.columns)}")
    print(f"\n  📋 Sample (3 rows):")
    print(
        verify_df[[
            "timestamp", "aqi", "pm25",
            "target_aqi_1d", "target_aqi_3d"
        ]].head(3).to_string(index=False)
    )

except Exception as e:
    print(f"\n  ⚠️  Verification skipped: {e}")
    print(f"  Check manually at https://app.hopsworks.ai")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("🎉 🎉 🎉  UPLOAD COMPLETE!  🎉 🎉 🎉")
print("="*60)
print(f"""
  📁 Source File   : {CSV_PATH}
  📊 Rows Uploaded : {len(df)}
  📊 Features      : 12
  🎯 Targets       : 3 (1d, 2d, 3d)

  🏪 Hopsworks:
     Project        : karachi_aqipred
     Feature Group  : {FG_NAME} (v{FG_VERSION})
     Feature View   : {FV_NAME} (v{FV_VERSION})

  🌐 View in UI:
     https://app.hopsworks.ai
     → Feature Groups → karachi_aqi_features ✅
     → Feature Views  → karachi_aqi_fv       ✅

  🎯 Next Step:
     python train_model.py
""")
print("="*60 + "\n")