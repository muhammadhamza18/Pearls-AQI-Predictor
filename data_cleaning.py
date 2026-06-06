# data_cleaning.py
"""
Data Cleaning Pipeline for AQI Prediction System
=================================================
Cleans the feature-rich dataset before uploading to Hopsworks Feature Store

Steps:
1. Check for missing values
2. Check for duplicates
3. Detect and handle outliers
4. Generate cleaning report
5. Save cleaned data
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("\n" + "="*80)
print("🧹 DATA CLEANING PIPELINE - AQI PREDICTION SYSTEM")
print("="*80)
print("\nCleaning dataset before uploading to Hopsworks Feature Store")
print("="*80 + "\n")


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data():
    """Load the feature-extracted dataset"""
    
    print("📂 STEP 1: LOADING DATA")
    print("-" * 80)
    
    filepath = 'final_data/karachi_training_data.csv'
    
    if not os.path.exists(filepath):
        print(f"❌ ERROR: File not found: {filepath}")
        print("\n💡 Run features_extraction.py first!")
        return None
    
    print(f"✅ Found file: {filepath}")
    print(f"📖 Loading data...")
    
    df = pd.read_csv(filepath)
    
    print(f"✅ Loaded successfully!")
    print(f"\n📊 Dataset Info:")
    print(f"  - Rows: {len(df):,}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n📋 Columns: {list(df.columns)}")
    
    return df


# ============================================================================
# STEP 2: CHECK MISSING VALUES
# ============================================================================

def check_missing_values(df):
    """Comprehensive missing value analysis"""
    
    print("\n" + "="*80)
    print("🔍 STEP 2: CHECKING MISSING VALUES")
    print("-" * 80)
    
    # Count missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    # Create missing values summary
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    })
    
    # Filter only columns with missing values
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )
    
    if len(missing_df) == 0:
        print("✅ NO MISSING VALUES FOUND!")
        print("   Dataset is complete - no imputation needed.")
        return None
    
    print(f"⚠️  MISSING VALUES DETECTED!")
    print(f"   Total columns with missing values: {len(missing_df)}")
    print(f"\n📊 Missing Values Summary:\n")
    print(missing_df.to_string(index=False))
    
    # Visualize missing values
    print(f"\n📊 Creating missing values visualization...")
    plt.figure(figsize=(12, 6))
    
    top_missing = missing_df.head(15)  # Top 15 columns with most missing
    
    plt.barh(top_missing['Column'], top_missing['Missing_Percentage'])
    plt.xlabel('Missing Percentage (%)')
    plt.title('Missing Values by Feature (Top 15)')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('cleaning_reports', exist_ok=True)
    plt.savefig('cleaning_reports/missing_values.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot: cleaning_reports/missing_values.png")
    plt.close()
    
    return missing_df


def handle_missing_values(df):
    """Handle missing values with appropriate strategies"""
    
    print("\n" + "-"*80)
    print("🔧 HANDLING MISSING VALUES")
    print("-" * 80)
    
    df_clean = df.copy()
    
    # Strategy 1: Forward fill for time series features (lag, rolling)
    print("\n1️⃣  Forward filling lag and rolling features...")
    
    lag_cols = [col for col in df_clean.columns if 'lag' in col or 'rolling' in col]
    if lag_cols:
        for col in lag_cols:
            before = df_clean[col].isnull().sum()
            if before > 0:
                df_clean[col] = df_clean[col].ffill().bfill()
                after = df_clean[col].isnull().sum()
                print(f"   {col}: {before} → {after} missing")
    
    # Strategy 2: Fill change features with 0 (no change)
    print("\n2️⃣  Filling change features with 0...")
    
    change_cols = [col for col in df_clean.columns if 'change' in col]
    if change_cols:
        for col in change_cols:
            before = df_clean[col].isnull().sum()
            if before > 0:
                df_clean[col] = df_clean[col].fillna(0)
                after = df_clean[col].isnull().sum()
                print(f"   {col}: {before} → {after} missing")
    
    # Strategy 3: Forward fill remaining numeric features
    print("\n3️⃣  Forward filling remaining numeric features...")
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        before = df_clean[col].isnull().sum()
        if before > 0:
            df_clean[col] = df_clean[col].ffill().bfill()
            after = df_clean[col].isnull().sum()
            if after < before:
                print(f"   {col}: {before} → {after} missing")
    
    # Final check
    total_missing = df_clean.isnull().sum().sum()
    
    print(f"\n✅ Missing value handling complete!")
    print(f"   Remaining missing values: {total_missing}")
    
    return df_clean


# ============================================================================
# STEP 3: CHECK DUPLICATES
# ============================================================================

def check_duplicates(df):
    """Check for duplicate rows"""
    
    print("\n" + "="*80)
    print("🔍 STEP 3: CHECKING DUPLICATES")
    print("-" * 80)
    
    # Check for exact duplicates
    duplicates = df.duplicated().sum()
    
    print(f"\n📊 Duplicate Analysis:")
    print(f"  - Total rows: {len(df):,}")
    print(f"  - Duplicate rows: {duplicates:,}")
    print(f"  - Duplicate percentage: {(duplicates/len(df)*100):.2f}%")
    
    if duplicates == 0:
        print("\n✅ NO DUPLICATES FOUND!")
        print("   Dataset is unique.")
        return None
    
    print(f"\n⚠️  DUPLICATES DETECTED!")
    
    # Check duplicates based on timestamp (more relevant)
    if 'timestamp' in df.columns:
        timestamp_dups = df.duplicated(subset=['timestamp']).sum()
        print(f"\n  Timestamp duplicates: {timestamp_dups:,}")
        
        if timestamp_dups > 0:
            print(f"\n  Sample duplicate timestamps:")
            dup_timestamps = df[df.duplicated(subset=['timestamp'], keep=False)]['timestamp'].head(10)
            for ts in dup_timestamps:
                print(f"    - {ts}")
    
    return duplicates


def remove_duplicates(df):
    """Remove duplicate rows"""
    
    print("\n" + "-"*80)
    print("🔧 REMOVING DUPLICATES")
    print("-" * 80)
    
    original_len = len(df)
    
    # Remove exact duplicates, keep first occurrence
    df_clean = df.drop_duplicates(keep='first')
    
    removed = original_len - len(df_clean)
    
    print(f"\n✅ Duplicates removed!")
    print(f"  - Original rows: {original_len:,}")
    print(f"  - Cleaned rows: {len(df_clean):,}")
    print(f"  - Removed: {removed:,} ({removed/original_len*100:.2f}%)")
    
    return df_clean


# ============================================================================
# STEP 4: DETECT OUTLIERS
# ============================================================================

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 3 * IQR  # Using 3*IQR for less aggressive removal
    upper_bound = Q3 + 3 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return outliers, lower_bound, upper_bound


def check_outliers(df):
    """Comprehensive outlier detection"""
    
    print("\n" + "="*80)
    print("🔍 STEP 4: DETECTING OUTLIERS")
    print("-" * 80)
    
    # Key features to check for outliers
    key_features = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    
    # Filter features that exist
    key_features = [f for f in key_features if f in df.columns]
    
    print(f"\n📊 Checking outliers in key features: {key_features}")
    
    outlier_summary = []
    
    for col in key_features:
        outliers, lower, upper = detect_outliers_iqr(df, col)
        
        outlier_summary.append({
            'Feature': col,
            'Outlier_Count': len(outliers),
            'Outlier_Percentage': (len(outliers) / len(df)) * 100,
            'Lower_Bound': lower,
            'Upper_Bound': upper,
            'Min_Value': df[col].min(),
            'Max_Value': df[col].max()
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    
    print(f"\n📊 Outlier Summary:\n")
    print(outlier_df.to_string(index=False))
    
    # Visualize outliers with box plots
    print(f"\n📊 Creating outlier visualizations...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, col in enumerate(key_features):
        if idx < len(axes):
            df.boxplot(column=col, ax=axes[idx])
            axes[idx].set_title(f'{col.upper()} - Outliers')
            axes[idx].set_ylabel('Value')
    
    # Hide unused subplots
    for idx in range(len(key_features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('cleaning_reports/outliers_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot: cleaning_reports/outliers_boxplot.png")
    plt.close()
    
    return outlier_df


def handle_outliers(df):
    """Handle outliers - CONSERVATIVE approach for AQI data"""
    
    print("\n" + "-"*80)
    print("🔧 HANDLING OUTLIERS")
    print("-" * 80)
    
    print("\n⚠️  IMPORTANT: For AQI data, extreme values may be REAL!")
    print("   (e.g., PM2.5 can legitimately reach 500+ during severe pollution)")
    print("\n   Strategy: Only remove IMPOSSIBLE values, not just high values")
    
    df_clean = df.copy()
    
    # Define REASONABLE limits for AQI features (based on EPA standards)
    limits = {
        'aqi': (0, 500),      # AQI max is 500
        'pm25': (0, 1000),    # PM2.5 can be very high in severe pollution
        'pm10': (0, 1000),    # PM10 similar
        'o3': (0, 500),       # O3 in µg/m³
        'no2': (0, 1000),     # NO2 in µg/m³
        'so2': (0, 1000),     # SO2 in µg/m³
        'co': (0, 50000)      # CO in µg/m³ (can be high)
    }
    
    removed_count = 0
    
    for col, (min_val, max_val) in limits.items():
        if col in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= min_val) & (df_clean[col] <= max_val)]
            after = len(df_clean)
            removed = before - after
            
            if removed > 0:
                print(f"   {col}: Removed {removed} impossible values (< {min_val} or > {max_val})")
                removed_count += removed
    
    print(f"\n✅ Outlier handling complete!")
    print(f"   Total rows removed: {removed_count}")
    print(f"   Remaining rows: {len(df_clean):,}")
    
    return df_clean


# ============================================================================
# STEP 5: DATA QUALITY CHECKS
# ============================================================================

def data_quality_checks(df):
    """Additional data quality checks"""
    
    print("\n" + "="*80)
    print("🔍 STEP 5: DATA QUALITY CHECKS")
    print("-" * 80)
    
    print("\n1️⃣  Checking for infinite values...")
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"   Infinite values: {inf_count}")
    
    if inf_count > 0:
        print("   ⚠️  Replacing infinite values with NaN...")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()
    
    print("\n2️⃣  Checking data types...")
    print(f"   Timestamp column type: {df['timestamp'].dtype}")
    
    if df['timestamp'].dtype == 'object':
        print("   🔧 Converting timestamp to datetime...")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("\n3️⃣  Checking for negative values in key features...")
    key_features = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    
    for col in key_features:
        if col in df.columns:
            negative = (df[col] < 0).sum()
            if negative > 0:
                print(f"   ⚠️  {col}: {negative} negative values found")
                df[col] = df[col].clip(lower=0)  # Replace negative with 0
    
    print("\n✅ Quality checks complete!")
    
    return df


# ============================================================================
# STEP 6: GENERATE CLEANING REPORT
# ============================================================================

def generate_cleaning_report(df_original, df_clean):
    """Generate comprehensive cleaning report"""
    
    print("\n" + "="*80)
    print("📊 STEP 6: GENERATING CLEANING REPORT")
    print("-" * 80)
    
    report = []
    report.append("="*80)
    report.append("DATA CLEANING REPORT - KARACHI AQI PREDICTION SYSTEM")
    report.append("="*80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n" + "-"*80)
    
    # Original data
    report.append("\nORIGINAL DATASET:")
    report.append(f"  - Rows: {len(df_original):,}")
    report.append(f"  - Columns: {len(df_original.columns)}")
    report.append(f"  - Missing values: {df_original.isnull().sum().sum():,}")
    report.append(f"  - Duplicates: {df_original.duplicated().sum():,}")
    
    # Cleaned data
    report.append("\nCLEANED DATASET:")
    report.append(f"  - Rows: {len(df_clean):,}")
    report.append(f"  - Columns: {len(df_clean.columns)}")
    report.append(f"  - Missing values: {df_clean.isnull().sum().sum():,}")
    report.append(f"  - Duplicates: {df_clean.duplicated().sum():,}")
    
    # Changes
    rows_removed = len(df_original) - len(df_clean)
    report.append("\nCHANGES:")
    report.append(f"  - Rows removed: {rows_removed:,} ({rows_removed/len(df_original)*100:.2f}%)")
    report.append(f"  - Rows retained: {len(df_clean):,} ({len(df_clean)/len(df_original)*100:.2f}%)")
    
    # Statistics
    report.append("\nKEY STATISTICS (CLEANED DATA):")
    report.append(f"\n  AQI:")
    report.append(f"    - Mean: {df_clean['aqi'].mean():.2f}")
    report.append(f"    - Median: {df_clean['aqi'].median():.2f}")
    report.append(f"    - Min: {df_clean['aqi'].min():.2f}")
    report.append(f"    - Max: {df_clean['aqi'].max():.2f}")
    report.append(f"    - Std: {df_clean['aqi'].std():.2f}")
    
    if 'pm25' in df_clean.columns:
        report.append(f"\n  PM2.5:")
        report.append(f"    - Mean: {df_clean['pm25'].mean():.2f}")
        report.append(f"    - Median: {df_clean['pm25'].median():.2f}")
        report.append(f"    - Min: {df_clean['pm25'].min():.2f}")
        report.append(f"    - Max: {df_clean['pm25'].max():.2f}")
    
    report.append("\n" + "="*80)
    report.append("DATASET IS CLEAN AND READY FOR FEATURE STORE!")
    report.append("="*80)
    
    # Save report
    os.makedirs('cleaning_reports', exist_ok=True)
    report_path = 'cleaning_reports/cleaning_report.txt'
    
    try:
        # Try UTF-8 first
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print(f"\n✅ Report saved: {report_path}")
    except Exception as e:
        # Fallback to ASCII-safe version
        print(f"\n⚠️  UTF-8 encoding failed, saving ASCII version...")
        with open(report_path, 'w', encoding='ascii', errors='ignore') as f:
            f.write('\n'.join(report))
        print(f"✅ Report saved (ASCII): {report_path}")
    
    # Print report
    print("\n" + '\n'.join(report))


# ============================================================================
# STEP 7: SAVE CLEANED DATA
# ============================================================================

def save_cleaned_data(df, city="karachi"):
    """Save cleaned dataset"""
    
    print("\n" + "="*80)
    print("💾 STEP 7: SAVING CLEANED DATA")
    print("-" * 80)
    
    # Create output directory
    os.makedirs('cleaned_data', exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save with timestamp
    timestamped_file = f"cleaned_data/{city}_cleaned_{timestamp}.csv"
    df.to_csv(timestamped_file, index=False, encoding='utf-8')
    print(f"✅ Saved: {timestamped_file}")
    
    # Save as main cleaned file (for easy access)
    main_file = f"cleaned_data/{city}_cleaned.csv"
    df.to_csv(main_file, index=False, encoding='utf-8')
    print(f"✅ Saved: {main_file}")
    
    print(f"\n📊 File Information:")
    print(f"  - Size: {os.path.getsize(main_file) / 1024:.2f} KB")
    print(f"  - Rows: {len(df):,}")
    print(f"  - Columns: {len(df.columns)}")
    
    return main_file


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("🚀 STARTING DATA CLEANING PIPELINE")
    print("="*80)
    print("\n📋 Pipeline Steps:")
    print("  1️⃣  Load data")
    print("  2️⃣  Check missing values")
    print("  3️⃣  Check duplicates")
    print("  4️⃣  Detect outliers")
    print("  5️⃣  Data quality checks")
    print("  6️⃣  Generate report")
    print("  7️⃣  Save cleaned data")
    print("\n⏱️  Estimated time: 1-2 minutes")
    print("="*80 + "\n")
    
    # Load data
    df_original = load_data()
    
    if df_original is None:
        return
    
    # Store original for comparison
    df = df_original.copy()
    
    # Step 2: Missing values
    missing_df = check_missing_values(df)
    if missing_df is not None:
        df = handle_missing_values(df)
    
    # Step 3: Duplicates
    duplicates = check_duplicates(df)
    if duplicates and duplicates > 0:
        df = remove_duplicates(df)
    
    # Step 4: Outliers
    outlier_df = check_outliers(df)
    df = handle_outliers(df)
    
    # Step 5: Quality checks
    df = data_quality_checks(df)
    
    # Step 6: Generate report
    generate_cleaning_report(df_original, df)
    
    # Step 7: Save cleaned data
    cleaned_file = save_cleaned_data(df, city="karachi")
    
    print("\n" + "="*80)
    print("✅ ✅ ✅  DATA CLEANING COMPLETE!  ✅ ✅ ✅")
    print("="*80)
    print(f"\n📂 Your cleaned data is ready!")
    print(f"📁 Main file: {cleaned_file}")
    print(f"📊 Rows: {len(df):,}")
    print(f"📊 Columns: {len(df.columns)}")
    print(f"📊 Ready for EDA and Feature Store upload! ✅")
    print("\n🎯 Next Steps:")
    print("  1. Run EDA to identify important features")
    print("  2. Drop unimportant features")
    print("  3. Upload to Hopsworks Feature Store")
    print("  4. Train ML models")
    print("\n💡 View detailed report: cleaning_reports/cleaning_report.txt")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user!")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()