# re_upload_raw.py
"""
ONE-TIME SCRIPT - Run this ONCE before starting pipelines
==========================================================
Purpose : Read existing CSV → extract raw columns only
          → upload to NEW feature group karachi_aqi_raw (v1)

Why     : Automation pipelines need a raw data starting point
          Hourly pipeline will append to this feature group
          Daily training will fetch from this feature group

Columns : timestamp, aqi, pm10, pm25, co, o3
          (5 raw pollutants — what we have from original fetch)

Run     : python re_upload_raw.py
"""

import os
import sys
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ============================================================
# SETTINGS
# ============================================================
PROJECT_NAME   = "karachi_aqipred"
CSV_PATH       = "final_data/karachi_hopsworks_upload.csv"
FG_NAME        = "karachi_aqi_raw"
FG_VERSION     = 1
FG_DESCRIPTION = (
    "Raw AQI pollutant data for Karachi. "
    "Collected hourly from OpenWeatherMap API. "
    "Used as source for feature engineering in training pipeline."
)

# Raw columns we keep from existing CSV
# (no2, so2, nh3 were not saved in original fetch - that's okay)
RAW_COLS = [
    "timestamp",
    "aqi",
    "pm10",
    "pm25",
    "co",
    "o3",
]

# ============================================================
# STEP 1: CONNECT TO HOPSWORKS
# ============================================================
print("\n" + "="*60)
print("STEP 1: CONNECTING TO HOPSWORKS")
print("="*60)

api_key = os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    print("  ERROR: HOPSWORKS_API_KEY not found in .env!")
    sys.exit(1)

print("\n  Connecting to: " + PROJECT_NAME + " ...")

try:
    project = hopsworks.login(
        project=PROJECT_NAME,
        api_key_value=api_key
    )
    print("  Connected: " + project.name)

except Exception as e:
    print("  Connection failed: " + str(e))
    sys.exit(1)

# ============================================================
# STEP 2: LOAD EXISTING CSV
# ============================================================
print("\n" + "="*60)
print("STEP 2: LOADING EXISTING CSV")
print("="*60)

if not os.path.exists(CSV_PATH):
    print("  ERROR: CSV not found: " + CSV_PATH)
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("\n  Loaded  : " + CSV_PATH)
print("  Shape   : " + str(df.shape))
print("  Columns : " + str(df.columns.tolist()))
print("  Range   : " + str(df["timestamp"].min()) +
      " to " + str(df["timestamp"].max()))

# ============================================================
# STEP 3: EXTRACT RAW COLUMNS ONLY
# ============================================================
print("\n" + "="*60)
print("STEP 3: EXTRACTING RAW COLUMNS")
print("="*60)

missing = [c for c in RAW_COLS if c not in df.columns]
if missing:
    print("  ERROR: Missing columns: " + str(missing))
    sys.exit(1)

df_raw = df[RAW_COLS].copy()

print("\n  Keeping only raw columns:")
for col in RAW_COLS:
    nulls = df_raw[col].isnull().sum()
    print("    " + col.ljust(15) +
          "  nulls=" + str(nulls))

print("\n  Raw data shape: " + str(df_raw.shape))

# ============================================================
# STEP 4: CLEAN RAW DATA
# ============================================================
print("\n" + "="*60)
print("STEP 4: CLEANING RAW DATA")
print("="*60)

original_rows = len(df_raw)

# Remove duplicates based on timestamp
df_raw = df_raw.drop_duplicates(subset=["timestamp"])
dupes_removed = original_rows - len(df_raw)
print("\n  Duplicates removed : " + str(dupes_removed))

# Handle missing values
nulls_before = df_raw.isnull().sum().sum()
df_raw = df_raw.fillna(method="ffill")   # forward fill first
df_raw = df_raw.fillna(method="bfill")   # backward fill remaining
nulls_after = df_raw.isnull().sum().sum()
print("  Nulls before fill  : " + str(nulls_before))
print("  Nulls after fill   : " + str(nulls_after))

# Validate AQI range (should be between 0 and 500)
invalid_aqi = df_raw[
    (df_raw["aqi"] < 0) | (df_raw["aqi"] > 500)
]
if len(invalid_aqi) > 0:
    print("  WARNING: " + str(len(invalid_aqi)) +
          " rows with invalid AQI values (outside 0-500)")
    df_raw = df_raw[
        (df_raw["aqi"] >= 0) & (df_raw["aqi"] <= 500)
    ]
    print("  Removed invalid rows")

print("\n  Final shape: " + str(df_raw.shape))
print("\n  Sample (first 3 rows):")
print(df_raw.head(3).to_string())
print("\n  Sample (last 3 rows):")
print(df_raw.tail(3).to_string())

# ============================================================
# STEP 5: UPLOAD TO HOPSWORKS
# ============================================================
print("\n" + "="*60)
print("STEP 5: UPLOADING TO HOPSWORKS")
print("="*60)

print("\n  Feature Group : " + FG_NAME)
print("  Version       : " + str(FG_VERSION))
print("  Rows          : " + str(len(df_raw)))
print("  Columns       : " + str(df_raw.columns.tolist()))

try:
    fs = project.get_feature_store()

    # Create or get feature group
    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        primary_key=["timestamp"],
        description=FG_DESCRIPTION,
        event_time="timestamp",
    )

    print("\n  Uploading " + str(len(df_raw)) + " rows...")

    fg.insert(df_raw, write_options={"wait_for_job": True})

    print("\n  Upload complete!")
    print("  Feature Group : " + FG_NAME + " (v" + str(FG_VERSION) + ")")
    print("  Rows uploaded : " + str(len(df_raw)))
    print("  Columns       : " + str(df_raw.columns.tolist()))

except Exception as e:
    print("  Upload failed: " + str(e))
    sys.exit(1)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("RE-UPLOAD COMPLETE!")
print("="*60)

print("\n  Feature Group: " + FG_NAME + " (v" + str(FG_VERSION) + ")")
print("  Rows         : " + str(len(df_raw)))
print("  Columns      :")
for col in RAW_COLS:
    print("    - " + col)

print("\n  Date range   : " + str(df_raw["timestamp"].min()) +
      " to " + str(df_raw["timestamp"].max()))

print("\n  What happens next:")
print("    1. Hourly pipeline appends 1 raw row/hour to this group")
print("    2. Daily training fetches ALL rows from this group")
print("    3. Training engineers features + targets from raw data")
print("    4. Best model uploaded to karachi_aqi_predictor registry")

print("\n  Explore in Hopsworks:")
print("    https://app.hopsworks.ai")
print("    Project: " + PROJECT_NAME)
print("    Feature Store → Feature Groups → " + FG_NAME)

print("\n" + "="*60)
print("  Run ONCE complete! Do NOT run this again.")
print("  Hourly pipeline will append from now on.")
print("="*60)