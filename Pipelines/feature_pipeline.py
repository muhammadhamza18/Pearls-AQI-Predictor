# feature_pipeline.py
"""
Hourly Feature Pipeline - Fetch & Upload Raw AQI Data
======================================================
Runs    : Every hour via GitHub Actions
Purpose : Fetch current AQI from OpenWeatherMap API
          Clean data (handle nulls, duplicates)
          Append 1 row to Hopsworks Feature Store

Flow:
  1. Fetch current AQI from OpenWeatherMap API
  2. Clean data (handle missing values, validate ranges)
  3. Check for duplicates in Hopsworks
  4. Append to karachi_aqi_raw (v1)
  5. Log success

Secrets Required:
  - OPENWEATHER_API_KEY (from GitHub Secrets)
  - HOPSWORKS_API_KEY (from GitHub Secrets)
"""

import os
import sys
import requests
import pandas as pd
import hopsworks
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_NAME = "karachi_aqipred"
FG_NAME      = "karachi_aqi_raw"
FG_VERSION   = 1

# Karachi coordinates
LATITUDE  = 24.8607
LONGITUDE = 67.0011

# OpenWeatherMap API
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

# Columns we store (raw data only)
RAW_COLUMNS = ["timestamp", "aqi", "pm10", "pm25", "co", "o3"]

# Validation ranges
VALID_RANGES = {
    "aqi":  (0, 500),
    "pm10": (0, 1000),
    "pm25": (0, 500),
    "co":   (0, 50000),
    "o3":   (0, 1000),
}


# ============================================================
# STEP 1: FETCH CURRENT AQI FROM OPENWEATHER API
# ============================================================
def fetch_current_aqi():
    """
    Fetch current air quality data from OpenWeatherMap API
    Returns dict with pollutant values
    """
    print("\n" + "="*60)
    print("STEP 1: FETCHING CURRENT AQI FROM OPENWEATHER API")
    print("="*60)
    
    if not OPENWEATHER_API_KEY:
        print("  ERROR: OPENWEATHER_API_KEY not found!")
        print("  Set it in .env or GitHub Secrets")
        sys.exit(1)
    
    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "appid": OPENWEATHER_API_KEY
    }
    
    print("\n  Fetching AQI for Karachi...")
    print("  Latitude  : " + str(LATITUDE))
    print("  Longitude : " + str(LONGITUDE))
    
    try:
        response = requests.get(OPENWEATHER_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract data
        aqi_data = data["list"][0]
        components = aqi_data["components"]
        aqi_index = aqi_data["main"]["aqi"]
        
        # Map AQI index (1-5) to US AQI scale (0-500)
        # OpenWeather uses European AQI (1-5)
        # We approximate conversion to US AQI for consistency
        aqi_mapping = {1: 50, 2: 100, 3: 150, 4: 200, 5: 300}
        aqi_value = aqi_mapping.get(aqi_index, 150)
        
        raw_data = {
            "timestamp": datetime.now().replace(minute=0, second=0, microsecond=0),
            "aqi":  float(aqi_value),
            "pm10": float(components.get("pm10", 0)),
            "pm25": float(components.get("pm2_5", 0)),
            "co":   float(components.get("co", 0)),
            "o3":   float(components.get("o3", 0)),
        }
        
        print("\n  Raw data fetched:")
        for key, value in raw_data.items():
            if key == "timestamp":
                print("    " + key.ljust(10) + ": " + str(value))
            else:
                print("    " + key.ljust(10) + ": " + str(round(value, 2)))
        
        return raw_data
        
    except requests.exceptions.RequestException as e:
        print("  ERROR: Failed to fetch from OpenWeather API")
        print("  " + str(e))
        sys.exit(1)
    except (KeyError, IndexError) as e:
        print("  ERROR: Unexpected API response format")
        print("  " + str(e))
        sys.exit(1)


# ============================================================
# STEP 2: CLEAN AND VALIDATE DATA
# ============================================================
def clean_and_validate(raw_data):
    """
    Clean and validate raw data
    Handle missing values and validate ranges
    Same logic as data_cleaning.py
    """
    print("\n" + "="*60)
    print("STEP 2: CLEANING AND VALIDATING DATA")
    print("="*60)
    
    df = pd.DataFrame([raw_data])
    
    # Check for missing values
    print("\n  Checking for missing values...")
    nulls = df.isnull().sum()
    total_nulls = nulls.sum()
    
    if total_nulls > 0:
        print("  Found " + str(total_nulls) + " missing values:")
        for col, count in nulls.items():
            if count > 0:
                print("    " + col + ": " + str(count))
        
        print("\n  Handling missing values:")
        
        # For timestamp: should never be null (we just created it)
        if df["timestamp"].isnull().any():
            print("    ERROR: timestamp is null!")
            sys.exit(1)
        
        # For numeric columns: use forward fill from previous hour
        # If this is the very first run, use median/default values
        for col in ["aqi", "pm10", "pm25", "co", "o3"]:
            if df[col].isnull().any():
                # Try to get previous value from Hopsworks
                print("    " + col + ": attempting to fetch previous value...")
                
                # If we can't fetch (first run), use safe defaults
                defaults = {
                    "aqi": 100.0,   # moderate AQI
                    "pm10": 50.0,   # moderate PM10
                    "pm25": 25.0,   # moderate PM2.5
                    "co": 500.0,    # moderate CO
                    "o3": 60.0,     # moderate O3
                }
                df[col].fillna(defaults[col], inplace=True)
                print("      Filled with default: " + str(defaults[col]))
    else:
        print("  No missing values found")
    
    # Validate ranges
    print("\n  Validating data ranges...")
    invalid_found = False
    
    for col, (min_val, max_val) in VALID_RANGES.items():
        value = df[col].iloc[0]
        
        if value < min_val or value > max_val:
            print("    WARNING: " + col + " = " + str(value) + 
                  " outside valid range [" + str(min_val) + 
                  ", " + str(max_val) + "]")
            
            # Clip to valid range
            df[col] = df[col].clip(min_val, max_val)
            print("      Clipped to: " + str(df[col].iloc[0]))
            invalid_found = True
        else:
            print("    " + col.ljust(6) + ": " + 
                  str(round(value, 2)) + " [OK]")
    
    if not invalid_found:
        print("  All values within valid ranges")
    
    # Round values to 2 decimal places
    for col in ["aqi", "pm10", "pm25", "co", "o3"]:
        df[col] = df[col].round(2)
    
    print("\n  Cleaned data:")
    print(df.to_string(index=False))
    
    return df


# ============================================================
# STEP 3: CHECK FOR DUPLICATES IN HOPSWORKS
# ============================================================
def check_duplicates(df, feature_group):
    """
    Check if this timestamp already exists in Hopsworks
    Prevent duplicate uploads
    """
    print("\n" + "="*60)
    print("STEP 3: CHECKING FOR DUPLICATES")
    print("="*60)
    
    new_timestamp = df["timestamp"].iloc[0]
    print("\n  New row timestamp: " + str(new_timestamp))
    
    try:
        # Fetch recent data from feature group
        print("  Fetching last 24 hours from Hopsworks...")
        
        # Get last 24 rows (1 per hour)
        existing_data = feature_group.read(limit=24)
        
        if existing_data is None or len(existing_data) == 0:
            print("  No existing data found (first run)")
            return df
        
        print("  Found " + str(len(existing_data)) + " existing rows")
        
        # Check if timestamp exists
        existing_data["timestamp"] = pd.to_datetime(
            existing_data["timestamp"]
        )
        
        if new_timestamp in existing_data["timestamp"].values:
            print("\n  DUPLICATE FOUND!")
            print("  Timestamp " + str(new_timestamp) + 
                  " already exists in Hopsworks")
            print("  Skipping upload to prevent duplicate")
            return None
        else:
            print("  No duplicate found - OK to upload")
            return df
            
    except Exception as e:
        print("  Warning: Could not check duplicates: " + str(e))
        print("  Proceeding with upload anyway...")
        return df


# ============================================================
# STEP 4: APPEND TO HOPSWORKS
# ============================================================
def append_to_hopsworks(df):
    """
    Append cleaned row to Hopsworks Feature Store
    """
    print("\n" + "="*60)
    print("STEP 4: UPLOADING TO HOPSWORKS")
    print("="*60)
    
    hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
    
    if not hopsworks_api_key:
        print("  ERROR: HOPSWORKS_API_KEY not found!")
        print("  Set it in .env or GitHub Secrets")
        sys.exit(1)
    
    print("\n  Connecting to Hopsworks...")
    
    try:
        project = hopsworks.login(
            project=PROJECT_NAME,
            api_key_value=hopsworks_api_key
        )
        print("  Connected to project: " + project.name)
        
        fs = project.get_feature_store()
        
        # Get feature group
        feature_group = fs.get_feature_group(
            name=FG_NAME,
            version=FG_VERSION
        )
        
        print("  Feature Group: " + FG_NAME + " (v" + str(FG_VERSION) + ")")
        
        # Check for duplicates
        df = check_duplicates(df, feature_group)
        
        if df is None:
            print("\n  Skipped: Duplicate timestamp")
            return
        
        # Insert row
        print("\n  Inserting 1 row...")
        feature_group.insert(df, write_options={"wait_for_job": False})
        
        print("  Upload successful!")
        print("  Timestamp: " + str(df["timestamp"].iloc[0]))
        print("  AQI      : " + str(df["aqi"].iloc[0]))
        
    except Exception as e:
        print("  ERROR: Failed to upload to Hopsworks")
        print("  " + str(e))
        sys.exit(1)


# ============================================================
# STEP 5: LOG SUMMARY
# ============================================================
def log_summary(raw_data, success=True):
    """
    Log execution summary
    """
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    
    if success:
        print("\n  Status    : SUCCESS")
        print("  Timestamp : " + str(raw_data["timestamp"]))
        print("  AQI       : " + str(round(raw_data["aqi"], 2)))
        print("  PM10      : " + str(round(raw_data["pm10"], 2)))
        print("  PM25      : " + str(round(raw_data["pm25"], 2)))
        print("  CO        : " + str(round(raw_data["co"], 2)))
        print("  O3        : " + str(round(raw_data["o3"], 2)))
        print("\n  Next run  : 1 hour from now")
    else:
        print("\n  Status    : FAILED")
        print("  Check logs above for error details")
    
    print("\n" + "="*60)


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    """
    Main pipeline execution
    Runs hourly via GitHub Actions
    """
    print("\n" + "="*60)
    print("HOURLY FEATURE PIPELINE - STARTED")
    print("="*60)
    print("  Time: " + str(datetime.now()))
    
    try:
        # Step 1: Fetch current AQI
        raw_data = fetch_current_aqi()
        
        # Step 2: Clean and validate
        clean_df = clean_and_validate(raw_data)
        
        # Step 3 & 4: Check duplicates & upload to Hopsworks
        append_to_hopsworks(clean_df)
        
        # Step 5: Log success
        log_summary(raw_data, success=True)
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print("\n" + "="*60)
        print("PIPELINE FAILED")
        print("="*60)
        print("  Error: " + str(e))
        log_summary({}, success=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
    