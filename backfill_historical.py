# fetch_historical_openweather.py
"""
Fetch 6-MONTH HISTORICAL AQI data from OpenWeatherMap
Saves to CSV for model training
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


def calculate_us_aqi_from_pm25(pm25):
    """
    Calculate US EPA AQI from PM2.5
    Converts to standard 0-500 AQI scale
    """
    if pd.isna(pm25) or pm25 is None:
        return None
    
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500)
    ]
    
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (pm25 - c_low) + aqi_low
            return int(round(aqi))
    
    if pm25 > 500.4:
        return 500
    
    return None


def fetch_historical_aqi(lat=24.8607, lon=67.0011, months_back=6):
    """
    Fetch HISTORICAL AQI data from OpenWeatherMap
    
    Args:
        lat: Latitude (Karachi: 24.8607)
        lon: Longitude (Karachi: 67.0011)
        months_back: Number of months of history to fetch
    
    Returns:
        DataFrame with historical AQI data
    """
    
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        raise ValueError("❌ OPENWEATHER_API_KEY not found in .env file!")
    
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    
    # Calculate date range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=months_back * 30)
    
    # Convert to Unix timestamps
    start_unix = int(start_time.timestamp())
    end_unix = int(end_time.timestamp())
    
    params = {
        'lat': lat,
        'lon': lon,
        'start': start_unix,
        'end': end_unix,
        'appid': api_key
    }
    
    print(f"\n📡 Fetching {months_back}-month historical AQI data...")
    print(f"   Location: Karachi ({lat}, {lon})")
    print(f"   From: {start_time.strftime('%Y-%m-%d')}")
    print(f"   To: {end_time.strftime('%Y-%m-%d')}")
    print(f"   Total days: {months_back * 30}")
    
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        if 'list' not in data:
            print("❌ No historical data returned")
            return None
        
        print(f"✅ Fetched {len(data['list'])} hourly data points!")
        
        # Convert to records
        records = []
        
        for item in data['list']:
            dt = datetime.fromtimestamp(item['dt'])
            components = item['components']
            pm25 = components.get('pm2_5', None)
            
            # Calculate US EPA AQI
            us_aqi = calculate_us_aqi_from_pm25(pm25)
            
            records.append({
                'timestamp': dt.isoformat(),
                'city': 'karachi',
                'aqi': us_aqi,  # US EPA AQI (0-500 scale)
                'aqi_openweather': item['main']['aqi'],  # OpenWeather scale (1-5)
                'dominant_pollutant': 'pm25',
                'pm25': pm25,
                'pm10': components.get('pm10', None),
                'o3': components.get('o3', None),
                'no2': components.get('no2', None),
                'so2': components.get('so2', None),
                'co': components.get('co', None),
                'no': components.get('no', None),
                'nh3': components.get('nh3', None),
                'temperature': None,  # Not provided by this API
                'humidity': None,
                'pressure': None,
                'wind_speed': None,
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\n✅ Converted to DataFrame!")
        print(f"📊 Shape: {df.shape}")
        print(f"📊 Columns: {len(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_historical_data(df):
    """Save historical data to CSV"""
    
    os.makedirs('historical_data', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = f'historical_data/karachi_6months_hourly_{timestamp}.csv'
    
    df.to_csv(filepath, index=False)
    
    print(f"\n💾 Saved to: {filepath}")
    print(f"📊 Total records: {len(df)}")
    print(f"📊 File size: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    return filepath


def save_to_checkpoint(df):
    """Save to checkpoint folder for feature extraction"""
    
    os.makedirs('checkpoints', exist_ok=True)
    
    filepath = 'checkpoints/karachi_checkpoint.csv'
    
    df.to_csv(filepath, index=False)
    
    print(f"💾 Saved checkpoint: {filepath}")
    
    return filepath


def print_summary(df):
    """Print data summary"""
    
    print("\n" + "="*60)
    print("📊 HISTORICAL DATA SUMMARY")
    print("="*60)
    
    print(f"\n🌍 City: KARACHI")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"📊 Total records: {len(df)}")
    print(f"📊 Total columns: {len(df.columns)}")
    
    print(f"\n📈 AQI Statistics (US EPA):")
    aqi_data = df['aqi'].dropna()
    if len(aqi_data) > 0:
        print(f"  Mean:   {aqi_data.mean():.1f}")
        print(f"  Median: {aqi_data.median():.1f}")
        print(f"  Min:    {aqi_data.min():.1f}")
        print(f"  Max:    {aqi_data.max():.1f}")
        print(f"  Std:    {aqi_data.std():.1f}")
    
    print(f"\n💨 PM2.5 Statistics (µg/m³):")
    pm25_data = df['pm25'].dropna()
    if len(pm25_data) > 0:
        print(f"  Mean:   {pm25_data.mean():.1f}")
        print(f"  Median: {pm25_data.median():.1f}")
        print(f"  Min:    {pm25_data.min():.1f}")
        print(f"  Max:    {pm25_data.max():.1f}")
    
    print(f"\n📋 Sample Data (First 5 rows):")
    print(df[['timestamp', 'aqi', 'pm25', 'pm10', 'o3', 'no2']].head())
    
    print(f"\n🔢 Missing Values:")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print("  ✅ No missing values!")
    
    print("="*60 + "\n")


def main():
    """Main execution"""
    
    print("\n" + "="*60)
    print("🌦️  FETCH 6-MONTH HISTORICAL AQI - OpenWeatherMap")
    print("="*60)
    print("\nFetches 6 months of hourly air quality data")
    print("Perfect for model training!")
    print("="*60)
    
    # Karachi coordinates
    KARACHI_LAT = 24.8607
    KARACHI_LON = 67.0011
    
    # Fetch 6 months of historical data
    print("\n⏱️  This may take 1-2 minutes...")
    
    df = fetch_historical_aqi(KARACHI_LAT, KARACHI_LON, months_back=6)
    
    if df is not None and not df.empty:
        # Print summary
        print_summary(df)
        
        # Save to historical_data folder
        historical_file = save_historical_data(df)
        
        # Save to checkpoint for feature extraction
        checkpoint_file = save_to_checkpoint(df)
        
        print("\n" + "="*60)
        print("✅ HISTORICAL DATA FETCH COMPLETE!")
        print("="*60)
        print(f"\n📂 Files saved:")
        print(f"  1. Historical: {historical_file}")
        print(f"  2. Checkpoint: {checkpoint_file}")
        print(f"\n🎯 Next Steps:")
        print(f"  1. Review the data quality")
        print(f"  2. Run feature extraction: python features_extraction.py")
        print(f"  3. Train your model!")
        print("="*60 + "\n")
        
    else:
        print("\n❌ Failed to fetch historical data")
        print("\n💡 Troubleshooting:")
        print("  1. Check OPENWEATHER_API_KEY in .env")
        print("  2. Verify API key at: https://home.openweathermap.org/api_keys")
        print("  3. Check API call limits (1000/day free)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()