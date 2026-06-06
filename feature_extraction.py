# features_extraction.py
"""
AQI Feature Extraction Pipeline - Optimized for Model Training
==============================================================
Extracts the MOST IMPORTANT features for 3-day AQI prediction

Features Included:
- Raw pollutant features (PM2.5, PM10, O3, NO2, SO2, CO)
- Time-based features (hour, day, month, season, weekend)
- Derived features (lag, rolling means, change rates)
- Target variables (1d, 2d, 3d ahead AQI)

Saves final CSV ready for ML training
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

print("\n" + "="*70)
print("🔧 AQI FEATURE EXTRACTION PIPELINE")
print("="*70)
print("\nExtracting features from checkpoint data...")
print("="*70 + "\n")


# ============================================================================
# STEP 1: LOAD CHECKPOINT DATA
# ============================================================================

def load_checkpoint():
    """Load data from checkpoint file"""
    
    print("📂 Looking for checkpoint file...")
    
    checkpoint_file = 'checkpoints/karachi_checkpoint.csv'
    
    if not os.path.exists(checkpoint_file):
        print("❌ ERROR: Checkpoint file not found!")
        print(f"   Expected: {checkpoint_file}")
        print("\n💡 Run fetch_historical_openweather.py first!")
        return None
    
    print(f"✅ Found checkpoint: {checkpoint_file}")
    print(f"📖 Loading data...")
    
    try:
        df = pd.read_csv(checkpoint_file)
        print(f"✅ Loaded {len(df)} rows successfully!")
        
        print(f"\n📊 Columns: {df.columns.tolist()}")
        print(f"\n📋 Data Preview:")
        print(df.head(3))
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None


# ============================================================================
# STEP 2: EXTRACT RAW FEATURES (Clean & Prepare)
# ============================================================================

def extract_raw_features(df):
    """Clean and prepare raw pollutant features"""
    
    print("\n" + "="*70)
    print("📦 STEP 1: EXTRACTING RAW FEATURES")
    print("="*70)
    
    # Key pollutant features
    pollutant_features = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    
    print(f"📋 Processing {len(pollutant_features)} pollutant features...")
    
    # Fill missing values with forward fill then backward fill
    for feature in pollutant_features:
        if feature in df.columns:
            missing_before = df[feature].isnull().sum()
            if missing_before > 0:
                print(f"  🔧 Filling {missing_before} missing values in {feature}")
                df[feature] = df[feature].ffill().bfill()
    
    print(f"✅ Raw features cleaned!")
    return df


# ============================================================================
# STEP 3: EXTRACT TIME-BASED FEATURES
# ============================================================================

def extract_time_features(df):
    """Extract time-based features - ESSENTIAL for temporal patterns"""
    
    print("\n" + "="*70)
    print("⏰ STEP 2: EXTRACTING TIME-BASED FEATURES")
    print("="*70)
    
    # Convert timestamp to datetime
    print("📅 Converting timestamp to datetime...")
    df['datetime'] = pd.to_datetime(df['timestamp'])
    
    # Extract MOST IMPORTANT time features
    print("📅 Extracting time components...")
    
    # Hour (0-23) - VERY IMPORTANT for daily patterns
    df['hour'] = df['datetime'].dt.hour
    
    # Day of week (0=Monday, 6=Sunday) - IMPORTANT for weekly patterns
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Month (1-12) - IMPORTANT for seasonal patterns
    df['month'] = df['datetime'].dt.month
    
    # Day of month (1-31)
    df['day'] = df['datetime'].dt.day
    
    # Is weekend? (0=weekday, 1=weekend)
    print("🏖️  Adding weekend indicator...")
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Season - IMPORTANT for Karachi climate (WITH NAMES!)
    print("🌦️  Adding season...")
    def get_season(month):
        if month in [12, 1, 2]:
            return 'winter'   # Winter season
        elif month in [3, 4, 5]:
            return 'spring'   # Spring season
        elif month in [6, 7, 8]:
            return 'summer'   # Summer season
        else:
            return 'autumn'   # Autumn season
    
    df['season'] = df['month'].apply(get_season)
    
    # Is rush hour? (7-9 AM, 5-7 PM) - IMPORTANT for traffic pollution
    print("🚗 Adding rush hour indicator...")
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                           (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    
    print(f"✅ Extracted 7 time-based features!")
    return df


# ============================================================================
# STEP 4: EXTRACT DERIVED FEATURES (Most Important)
# ============================================================================

def extract_derived_features(df):
    """Extract MOST IMPORTANT derived features for AQI prediction"""
    
    print("\n" + "="*70)
    print("🔧 STEP 3: EXTRACTING DERIVED FEATURES")
    print("="*70)
    
    # Sort by datetime FIRST
    print("📊 Sorting by datetime...")
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # ========== LAG FEATURES (Past Values) ==========
    print("\n📈 Creating lag features (past values)...")
    print("  - AQI lags: 1h, 3h, 6h, 12h, 24h")
    
    df['aqi_lag_1h'] = df['aqi'].shift(1)   # 1 hour ago
    df['aqi_lag_3h'] = df['aqi'].shift(3)   # 3 hours ago
    df['aqi_lag_6h'] = df['aqi'].shift(6)   # 6 hours ago
    df['aqi_lag_12h'] = df['aqi'].shift(12) # 12 hours ago
    df['aqi_lag_24h'] = df['aqi'].shift(24) # 24 hours ago (yesterday)
    
    # ========== CHANGE FEATURES (Trends) ==========
    print("\n📊 Creating change features (hourly trends)...")
    
    df['aqi_change_1h'] = df['aqi'].diff(1)   # Change from 1 hour ago
    df['aqi_change_24h'] = df['aqi'].diff(24) # Change from 24 hours ago
    
    # Change rate (percentage)
    df['aqi_change_rate_1h'] = df['aqi'].pct_change(1) * 100
    df['aqi_change_rate_24h'] = df['aqi'].pct_change(24) * 100
    
    # ========== ROLLING STATISTICS (Temporal Dependency) ==========
    print("\n🔄 Creating rolling features (temporal patterns)...")
    print("  - Rolling means: 3h, 6h, 12h, 24h")
    
    # Rolling means - VERY IMPORTANT for capturing trends
    df['aqi_rolling_mean_3h'] = df['aqi'].rolling(window=3, min_periods=1).mean()
    df['aqi_rolling_mean_6h'] = df['aqi'].rolling(window=6, min_periods=1).mean()
    df['aqi_rolling_mean_12h'] = df['aqi'].rolling(window=12, min_periods=1).mean()
    df['aqi_rolling_mean_24h'] = df['aqi'].rolling(window=24, min_periods=1).mean()
    
    # Rolling std - captures volatility
    df['aqi_rolling_std_24h'] = df['aqi'].rolling(window=24, min_periods=1).std()
    
    # Rolling min/max - captures range
    df['aqi_rolling_min_24h'] = df['aqi'].rolling(window=24, min_periods=1).min()
    df['aqi_rolling_max_24h'] = df['aqi'].rolling(window=24, min_periods=1).max()
    
    # ========== PM2.5 FEATURES ==========
    if 'pm25' in df.columns:
        print("\n💨 Creating PM2.5 features...")
        df['pm25_lag_24h'] = df['pm25'].shift(24)
        df['pm25_change_24h'] = df['pm25'].diff(24)
        df['pm25_rolling_mean_24h'] = df['pm25'].rolling(window=24, min_periods=1).mean()
    
    # ========== TREND INDICATOR ==========
    print("\n📈 Creating trend indicators...")
    df['is_aqi_rising'] = (df['aqi_change_1h'] > 0).astype(int)
    
    print(f"\n✅ Extracted 24 derived features!")
    return df


# ============================================================================
# STEP 5: CREATE TARGET VARIABLES
# ============================================================================

def create_target_variables(df):
    """Create target variables - What we want to predict"""
    
    print("\n" + "="*70)
    print("🎯 STEP 4: CREATING TARGET VARIABLES")
    print("="*70)
    
    # Target: AQI 24 hours (1 day) ahead
    print("📅 Creating target_aqi_1d (24 hours ahead)...")
    df['target_aqi_1d'] = df['aqi'].shift(-24)
    
    # Target: AQI 48 hours (2 days) ahead
    print("📅 Creating target_aqi_2d (48 hours ahead)...")
    df['target_aqi_2d'] = df['aqi'].shift(-48)
    
    # Target: AQI 72 hours (3 days) ahead
    print("📅 Creating target_aqi_3d (72 hours ahead)...")
    df['target_aqi_3d'] = df['aqi'].shift(-72)
    
    print(f"✅ Created 3 target variables!")
    return df


# ============================================================================
# STEP 6: SELECT FINAL FEATURES
# ============================================================================

def select_final_features(df):
    """Select MOST IMPORTANT features for the final dataset"""
    
    print("\n" + "="*70)
    print("🎯 STEP 5: SELECTING FINAL FEATURES")
    print("="*70)
    
    # Define feature groups
    
    # 1. IDENTIFIER
    identifier = ['timestamp', 'city']
    
    # 2. RAW POLLUTANT FEATURES (Most Important)
    raw_features = [
        'aqi',      # Current AQI
        'pm25',     # PM2.5 (most correlated with AQI)
        'pm10',     # PM10
        'o3',       # Ozone
        'no2',      # Nitrogen Dioxide
        'so2',      # Sulfur Dioxide
        'co'        # Carbon Monoxide
    ]
    
    # 3. TIME-BASED FEATURES
    time_features = [
        'hour',         # Hour of day (0-23)
        'day_of_week',  # Day of week (0-6)
        'month',        # Month (1-12)
        'day',          # Day of month
        'is_weekend',   # Weekend indicator
        'season',       # Season (0-3)
        'is_rush_hour'  # Rush hour indicator
    ]
    
    # 4. LAG FEATURES (Past values)
    lag_features = [
        'aqi_lag_1h',
        'aqi_lag_3h',
        'aqi_lag_6h',
        'aqi_lag_12h',
        'aqi_lag_24h'
    ]
    
    # 5. CHANGE FEATURES (Trends)
    change_features = [
        'aqi_change_1h',
        'aqi_change_24h',
        'aqi_change_rate_1h',
        'aqi_change_rate_24h'
    ]
    
    # 6. ROLLING FEATURES (Temporal dependency)
    rolling_features = [
        'aqi_rolling_mean_3h',
        'aqi_rolling_mean_6h',
        'aqi_rolling_mean_12h',
        'aqi_rolling_mean_24h',
        'aqi_rolling_std_24h',
        'aqi_rolling_min_24h',
        'aqi_rolling_max_24h'
    ]
    
    # 7. PM2.5 FEATURES
    pm25_features = [
        'pm25_lag_24h',
        'pm25_change_24h',
        'pm25_rolling_mean_24h'
    ]
    
    # 8. TREND FEATURES
    trend_features = [
        'is_aqi_rising'
    ]
    
    # 9. TARGET VARIABLES (At the end!)
    target_features = [
        'target_aqi_1d',
        'target_aqi_2d',
        'target_aqi_3d'
    ]
    
    # Combine all features in order
    all_features = (
        identifier +
        raw_features +
        time_features +
        lag_features +
        change_features +
        rolling_features +
        pm25_features +
        trend_features +
        target_features
    )
    
    # Select only features that exist in dataframe
    available_features = [f for f in all_features if f in df.columns]
    
    df_final = df[available_features].copy()
    
    print(f"\n📊 Feature Selection Summary:")
    print(f"  - Identifier: {len(identifier)}")
    print(f"  - Raw pollutants: {len([f for f in raw_features if f in df.columns])}")
    print(f"  - Time-based: {len([f for f in time_features if f in df.columns])}")
    print(f"  - Lag features: {len([f for f in lag_features if f in df.columns])}")
    print(f"  - Change features: {len([f for f in change_features if f in df.columns])}")
    print(f"  - Rolling features: {len([f for f in rolling_features if f in df.columns])}")
    print(f"  - PM2.5 features: {len([f for f in pm25_features if f in df.columns])}")
    print(f"  - Trend features: {len([f for f in trend_features if f in df.columns])}")
    print(f"  - Target variables: {len([f for f in target_features if f in df.columns])}")
    print(f"\n  ✅ TOTAL FEATURES: {len(available_features)}")
    
    return df_final


# ============================================================================
# STEP 7: CLEAN DATA
# ============================================================================

def clean_data(df):
    """Remove rows with missing targets"""
    
    print("\n" + "="*70)
    print("🧹 STEP 6: CLEANING DATA")
    print("="*70)
    
    original_len = len(df)
    
    # Remove rows with missing target (last 72 hours have no future data)
    print("🗑️  Removing rows with missing target values...")
    df_clean = df.dropna(subset=['target_aqi_3d'])
    
    removed = original_len - len(df_clean)
    
    print(f"✅ Removed {removed} rows (last 72 hours with no future data)")
    print(f"📊 Final dataset: {len(df_clean)} rows")
    
    return df_clean


# ============================================================================
# STEP 8: SAVE TO CSV
# ============================================================================

def save_to_csv(df, city="karachi"):
    """Save final feature-rich CSV"""
    
    print("\n" + "="*70)
    print("💾 STEP 7: SAVING TO CSV")
    print("="*70)
    
    # Create output directories
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('final_data', exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save to processed_data (with timestamp)
    processed_file = f"processed_data/{city}_features_{timestamp}.csv"
    df.to_csv(processed_file, index=False)
    print(f"✅ Saved to: {processed_file}")
    
    # Save to final_data (main file for training)
    final_file = f"final_data/{city}_training_data.csv"
    df.to_csv(final_file, index=False)
    print(f"✅ Saved to: {final_file}")
    
    # Save feature list
    feature_list_file = f"final_data/{city}_feature_list.txt"
    with open(feature_list_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"FEATURE LIST - {city.upper()}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Features: {len(df.columns)}\n")
        f.write(f"Total Records: {len(df)}\n\n")
        
        f.write("FEATURES:\n")
        f.write("-" * 70 + "\n")
        for i, col in enumerate(df.columns, 1):
            f.write(f"{i:3d}. {col}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"📄 Feature list: {feature_list_file}")
    
    print(f"\n📊 File Information:")
    print(f"  - Size: {os.path.getsize(final_file) / 1024:.2f} KB")
    print(f"  - Rows: {len(df)}")
    print(f"  - Columns: {len(df.columns)}")
    
    return final_file


# ============================================================================
# STEP 9: GENERATE SUMMARY
# ============================================================================

def generate_summary(df, city="karachi"):
    """Generate and print final summary"""
    
    print("\n" + "="*70)
    print("📊 FINAL DATA SUMMARY")
    print("="*70)
    
    print(f"\n🌍 City: {city.upper()}")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Calculate duration
    start_date = pd.to_datetime(df['timestamp'].min())
    end_date = pd.to_datetime(df['timestamp'].max())
    duration_days = (end_date - start_date).days
    
    print(f"⏱️  Duration: {duration_days} days (~{duration_days/30:.1f} months)")
    print(f"📊 Total records: {len(df)}")
    print(f"📊 Total features: {len(df.columns)}")
    
    print(f"\n📈 AQI Statistics:")
    print(f"  - Mean:   {df['aqi'].mean():.1f}")
    print(f"  - Median: {df['aqi'].median():.1f}")
    print(f"  - Min:    {df['aqi'].min():.1f}")
    print(f"  - Max:    {df['aqi'].max():.1f}")
    print(f"  - Std:    {df['aqi'].std():.1f}")
    
    print(f"\n💨 PM2.5 Statistics (µg/m³):")
    print(f"  - Mean:   {df['pm25'].mean():.1f}")
    print(f"  - Median: {df['pm25'].median():.1f}")
    print(f"  - Min:    {df['pm25'].min():.1f}")
    print(f"  - Max:    {df['pm25'].max():.1f}")
    
    print(f"\n📋 Sample Data (First 5 rows):")
    display_cols = ['timestamp', 'aqi', 'pm25', 'hour', 'aqi_lag_24h', 
                   'aqi_rolling_mean_24h', 'target_aqi_3d']
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].head())
    
    print("\n" + "="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("🚀 AQI FEATURE EXTRACTION PIPELINE")
    print("="*70)
    print("\n📋 What This Does:")
    print("  ✅ Loads your 6-month historical AQI data")
    print("  ✅ Extracts MOST IMPORTANT features for ML")
    print("  ✅ Creates temporal features (lag, rolling)")
    print("  ✅ Prepares targets (1d, 2d, 3d ahead)")
    print("  ✅ Saves final training-ready CSV")
    print("\n⏱️  Estimated time: 30 seconds")
    print("="*70 + "\n")
    
    input("Press Enter to start feature extraction...")
    
    # Pipeline
    df = load_checkpoint()
    
    if df is None:
        return
    
    df = extract_raw_features(df)
    df = extract_time_features(df)
    df = extract_derived_features(df)
    df = create_target_variables(df)
    df_final = select_final_features(df)
    df_clean = clean_data(df_final)
    
    final_file = save_to_csv(df_clean, city="karachi")
    generate_summary(df_clean, city="karachi")
    
    print("\n" + "="*70)
    print("✅ ✅ ✅  FEATURE EXTRACTION COMPLETE!  ✅ ✅ ✅")
    print("="*70)
    print(f"\n📂 Your training data is ready!")
    print(f"📁 Main file: final_data/karachi_training_data.csv")
    print(f"📊 Features: {len(df_clean.columns)}")
    print(f"📊 Records: {len(df_clean)}")
    print(f"📊 Ready for ML model training! ✅")
    print("\n🎯 Next Steps:")
    print("  1. Upload to Hopsworks Feature Store")
    print("  2. Train your ML model (Random Forest, XGBoost, LSTM)")
    print("  3. Build prediction pipeline")
    print("  4. Deploy Streamlit dashboard")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user!")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()