# Quick check script
import pandas as pd

df = pd.read_csv('checkpoints/karachi_checkpoint.csv')

print("Checking for duplicates...")
print(f"Total rows: {len(df)}")
print(f"\nUnique AQI values: {df['aqi'].nunique()}")
print(f"Unique PM2.5 values: {df['pm25'].nunique()}")
print(f"\nAQI value counts:")
print(df['aqi'].value_counts().head(10))

# Check if all rows are identical
print(f"\nAre all AQI values the same? {df['aqi'].nunique() == 1}")
