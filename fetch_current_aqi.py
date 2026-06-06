# Fetch current AQI data from OpenWeatherMap and save to CSV for monitoring
"""
Fetch CURRENT AQI data from OpenWeatherMap
Saves to CSV for monitoring
"""

import os
import requests
import json
from datetime import datetime
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


def fetch_current_aqi(lat=24.8607, lon=67.0011):
    """
    Fetch current AQI data from OpenWeatherMap
    
    Args:
        lat: Latitude (Karachi: 24.8607)
        lon: Longitude (Karachi: 67.0011)
    
    Returns:
        DataFrame with current AQI data
    """
    
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        raise ValueError("❌ OPENWEATHER_API_KEY not found in .env file!")
    
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key
    }
    
    print(f"\n📡 Fetching current AQI from OpenWeatherMap...")
    print(f"   Location: Karachi ({lat}, {lon})")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'list' not in data or not data['list']:
            print("❌ No data returned")
            return None
        
        pollution = data['list'][0]
        
        # Extract timestamp
        dt = datetime.fromtimestamp(pollution['dt'])
        
        # Extract components
        components = pollution['components']
        pm25 = components.get('pm2_5', None)
        
        # Calculate US EPA AQI
        us_aqi = calculate_us_aqi_from_pm25(pm25)
        
        # Create data dictionary
        aqi_data = {
            'timestamp': dt.isoformat(),
            'city': 'karachi',
            'aqi': us_aqi,  # US EPA AQI (0-500 scale)
            'aqi_openweather': pollution['main']['aqi'],  # OpenWeather scale (1-5)
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
        }
        
        print(f"✅ Current AQI: {us_aqi} (US EPA)")
        print(f"✅ PM2.5: {pm25} µg/m³")
        print(f"✅ Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return pd.DataFrame([aqi_data])
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def save_current_data(df):
    """
    Save current data to CSV
    Appends to existing file or creates new
    """
    
    os.makedirs('current_data1', exist_ok=True)
    filepath = 'current_data1/karachi_current_aqi.csv'
    
    if os.path.exists(filepath):
        # Append to existing
        df_existing = pd.read_csv(filepath)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_csv(filepath, index=False)
        print(f"\n💾 Appended to: {filepath}")
        print(f"📊 Total records: {len(df_combined)}")
    else:
        # Create new file
        df.to_csv(filepath, index=False)
        print(f"\n💾 Created: {filepath}")
        print(f"📊 Records: {len(df)}")
    
    return filepath


def print_summary(df):
    """Print data summary"""
    
    print("\n" + "="*60)
    print("📊 CURRENT AQI DATA SUMMARY")
    print("="*60)
    
    row = df.iloc[0]
    
    print(f"\n🌍 City: {row['city'].upper()}")
    print(f"📅 Timestamp: {row['timestamp']}")
    print(f"📊 AQI (US EPA): {row['aqi']}")
    
    # AQI Category
    aqi = row['aqi']
    if pd.notna(aqi):
        if aqi <= 50:
            category = "Good 🟢"
        elif aqi <= 100:
            category = "Moderate 🟡"
        elif aqi <= 150:
            category = "Unhealthy for Sensitive 🟠"
        elif aqi <= 200:
            category = "Unhealthy 🔴"
        elif aqi <= 300:
            category = "Very Unhealthy 🟣"
        else:
            category = "Hazardous 🟤"
        
        print(f"🏷️  Category: {category}")
    
    print(f"\n💨 Main Pollutants (µg/m³):")
    print(f"  PM2.5: {row['pm25']}")
    print(f"  PM10:  {row['pm10']}")
    print(f"  O3:    {row['o3']}")
    print(f"  NO2:   {row['no2']}")
    print(f"  SO2:   {row['so2']}")
    print(f"  CO:    {row['co']}")
    
    print("="*60 + "\n")


def main():
    """Main execution"""
    
    print("\n" + "="*60)
    print("🌦️  FETCH CURRENT AQI - OpenWeatherMap")
    print("="*60)
    print("\nFetches current air quality data for Karachi")
    print("Saves to CSV for monitoring")
    print("="*60)
    
    # Karachi coordinates
    KARACHI_LAT = 24.8607
    KARACHI_LON = 67.0011
    
    # Fetch current data
    df = fetch_current_aqi(KARACHI_LAT, KARACHI_LON)
    
    if df is not None and not df.empty:
        # Print summary
        print_summary(df)
        
        # Save to CSV
        filepath = save_current_data(df)
        
        # Also save as JSON
        json_path = 'current_data/karachi_current_aqi.json'
        df.to_json(json_path, orient='records', indent=2)
        print(f"💾 Also saved JSON: {json_path}")
        
        print("\n" + "="*60)
        print("✅ CURRENT DATA FETCH COMPLETE!")
        print("="*60)
        print("\n💡 Run this hourly to build a monitoring dataset!")
        print("="*60 + "\n")
        
    else:
        print("\n❌ Failed to fetch current data")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()