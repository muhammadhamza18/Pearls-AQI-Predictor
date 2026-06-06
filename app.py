# app.py
"""
Karachi AQI Prediction Dashboard
=================================
Real-time 3-day Air Quality Index predictions using ML models
Data fetched live from Hopsworks Feature Store
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import pickle
import hopsworks
from datetime import datetime, timedelta
import os

# Page config
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main styling with Sky Blue Theme */
    .main-header {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.4rem;
        color: #2F80ED;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    /* Metric cards with Sky Blue gradient */
    .metric-card {
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(47, 128, 237, 0.3);
        color: white;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.8s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(47, 128, 237, 0.4);
    }
    
    /* Current AQI card */
    .current-aqi-card {
        background: linear-gradient(135deg, #E0F7FF 0%, #FFFFFF 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(86, 204, 242, 0.2);
        text-align: center;
        border: 3px solid #56CCF2;
        animation: pulse 2s infinite;
    }
    
    /* Prediction cards */
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(86, 204, 242, 0.15);
        text-align: center;
        margin: 10px 0;
        border-left: 5px solid;
        transition: all 0.3s ease;
        animation: slideInUp 0.6s;
    }
    
    .prediction-card:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 35px rgba(86, 204, 242, 0.3);
    }
    
    .day-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .aqi-value {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .aqi-category {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
    
    .aqi-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.3rem;
        font-style: italic;
    }
    
    /* Animated predict button with Sky Blue */
    .stButton>button {
        width: 100%;
        height: 4rem;
        font-size: 1.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        color: white;
        border: none;
        border-radius: 50px;
        box-shadow: 0 10px 30px rgba(47, 128, 237, 0.4);
        transition: all 0.3s ease;
        animation: glow 2s infinite;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(47, 128, 237, 0.6);
        background: linear-gradient(135deg, #2F80ED 0%, #56CCF2 100%);
    }
    
    /* Pollutant cards with Sky Blue variations */
    .pollutant-card {
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 20px rgba(47, 128, 237, 0.2);
        margin: 0.5rem 0;
    }
    
    .pollutant-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .pollutant-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Section headers with Sky Blue */
    .section-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #56CCF2;
        animation: fadeIn 1s;
    }
    
    /* Info boxes with Sky Blue */
    .info-box {
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(47, 128, 237, 0.2);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
            box-shadow: 0 10px 40px rgba(86, 204, 242, 0.2);
        }
        50% {
            transform: scale(1.02);
            box-shadow: 0 15px 50px rgba(86, 204, 242, 0.4);
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 10px 30px rgba(47, 128, 237, 0.4);
        }
        50% {
            box-shadow: 0 10px 40px rgba(86, 204, 242, 0.7);
        }
    }
    
    /* Metric cards in model performance with Sky Blue */
    .performance-metric {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(86, 204, 242, 0.15);
        border-top: 4px solid #56CCF2;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2F80ED;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        font-weight: 600;
    }
    
    /* Top banner effect */
    .top-banner {
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 5px 20px rgba(47, 128, 237, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Constants
HOPSWORKS_API_KEY = "o4ON0hNP9EFs7lCy.vUKteKkiiS0Ud8csAPyieAOLLLDnigZ86XRTlVT237H5h8JhcvtDtMjgqj8ZasPA"
OPENWEATHER_API_KEY = "9dfaa9c3f4e58af2ff0837fa3e4c866b"
PROJECT_NAME = "karachi_aqipred"
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

# Selected features (must match training)
SELECTED_FEATURES = [
    'aqi_rolling_max_24h',
    'pm10',
    'pm25',
    'aqi',
    'aqi_rolling_mean_3h',
    'aqi_lag_1h',
    'aqi_rolling_mean_6h',
    'co',
    'aqi_rolling_mean_12h',
    'aqi_lag_3h',
    'o3',
    'aqi_lag_6h',
]

# AQI Categories
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "🟢", "#00e400"
    elif aqi <= 100:
        return "Moderate", "🟡", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "🔴", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "#8f3f97"
    else:
        return "Hazardous", "🟤", "#7e0023"

# Cache functions
@st.cache_data(ttl=3600)
def fetch_current_aqi():
    """Fetch current AQI from OpenWeather API"""
    try:
        url = "http://api.openweathermap.org/data/2.5/air_pollution"
        params = {
            "lat": KARACHI_LAT,
            "lon": KARACHI_LON,
            "appid": OPENWEATHER_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        components = data["list"][0]["components"]
        aqi_index = data["list"][0]["main"]["aqi"]
        
        # Map to US AQI scale
        aqi_mapping = {1: 50, 2: 100, 3: 150, 4: 200, 5: 300}
        aqi_value = aqi_mapping.get(aqi_index, 150)
        
        return {
            "aqi": aqi_value,
            "pm10": components.get("pm10", 0),
            "pm25": components.get("pm2_5", 0),
            "co": components.get("co", 0),
            "o3": components.get("o3", 0),
            "timestamp": datetime.now()
        }
    except Exception as e:
        st.error(f"Error fetching current AQI: {str(e)}")
        return None

@st.cache_resource
def connect_hopsworks():
    """Connect to Hopsworks"""
    try:
        project = hopsworks.login(
            project=PROJECT_NAME,
            api_key_value=HOPSWORKS_API_KEY
        )
        return project
    except Exception as e:
        st.error(f"Error connecting to Hopsworks: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def fetch_historical_data(_project):
    """Fetch last 24 hours from Hopsworks"""
    try:
        fs = _project.get_feature_store()
        fg = fs.get_feature_group(name="karachi_aqi_raw", version=1)
        
        df = fg.read()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Get last 24 rows
        df_recent = df.tail(24).copy()
        
        return df_recent
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return None

def engineer_features(df):
    """Create engineered features from raw data"""
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Rolling features
    df['aqi_rolling_max_24h'] = df['aqi'].rolling(window=24, min_periods=1).max()
    df['aqi_rolling_mean_3h'] = df['aqi'].rolling(window=3, min_periods=1).mean()
    df['aqi_rolling_mean_6h'] = df['aqi'].rolling(window=6, min_periods=1).mean()
    df['aqi_rolling_mean_12h'] = df['aqi'].rolling(window=12, min_periods=1).mean()
    
    # Lag features
    df['aqi_lag_1h'] = df['aqi'].shift(1)
    df['aqi_lag_3h'] = df['aqi'].shift(3)
    df['aqi_lag_6h'] = df['aqi'].shift(6)
    
    # Fill NaN with forward fill
    df = df.ffill().bfill()
    
    return df

@st.cache_data(ttl=3600)
def get_model_metrics(_project, model_name):
    """Get model metrics from Hopsworks Model Registry"""
    try:
        mr = _project.get_model_registry()
        
        # Get latest version of the model
        models = mr.get_models(name="karachi_aqi_predictor")
        if not models or len(models) == 0:
            return None
        
        # Find the model matching the name
        for model in models:
            model_dir = model.download()
            
            # Load metrics
            metrics_path = os.path.join(model_dir, "metrics.json")
            if os.path.exists(metrics_path):
                import json
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                return metrics
        
        return None
    except Exception as e:
        st.error(f"Error loading model metrics: {str(e)}")
        return None

@st.cache_resource
def load_model(_project, model_name):
    """Load model from Hopsworks Model Registry"""
    try:
        mr = _project.get_model_registry()
        
        # Get latest model version
        models = mr.get_models(name="karachi_aqi_predictor")
        if not models or len(models) == 0:
            st.error("No trained models found in registry")
            return None
        
        # Get latest version
        latest_model = max(models, key=lambda m: m.version)
        model_dir = latest_model.download()
        
        # Load the model
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_3_days(model, features_df):
    """Predict 3-day AQI"""
    try:
        # Get latest row with engineered features
        latest_features = features_df[SELECTED_FEATURES].iloc[-1:].values
        
        # Predict
        predictions = model.predict(latest_features)
        
        return {
            "day1": float(predictions[0][0]),
            "day2": float(predictions[0][1]),
            "day3": float(predictions[0][2])
        }
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

# Main Dashboard
def main():
    # Header with banner
    st.markdown("""
    <div class="top-banner">
        🌬️ Real-Time Air Quality Monitoring System | Powered by Machine Learning & Hopsworks MLOps Platform
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">🌍 KARACHI AIR QUALITY PREDICTOR</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">🔮 Advanced 3-Day AQI Forecasting with Live Data Analytics</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        # Sidebar header with cloud icon
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%); border-radius: 15px; margin-bottom: 2rem;">
            <div style="font-size: 4rem; margin-bottom: 0.5rem;">☁️</div>
            <h2 style="color: white; margin: 0; font-size: 1.5rem;">Air Quality</h2>
            <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9;">Monitoring System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Settings")
        
        model_choice = st.selectbox(
            "🤖 Select Model",
            ["CatBoost", "XGBoost", "RandomForest"],
            help="Choose the ML model for predictions"
        )
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("""
        This dashboard provides:
        - Current AQI for Karachi
        - 3-day AQI predictions
        - Model performance metrics
        - Real-time data from Hopsworks
        """)
        
        st.markdown("---")
        st.markdown("**Data Source:** Hopsworks Feature Store")
        st.markdown("**API:** OpenWeatherMap")
    
    # Connect to Hopsworks
    with st.spinner("🔗 Connecting to Hopsworks..."):
        project = connect_hopsworks()
    
    if project is None:
        st.error("Failed to connect to Hopsworks. Please check your API key.")
        return
    
    # Fetch current AQI
    st.markdown('<p class="section-header">📍 Current Air Quality</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        current_data = fetch_current_aqi()
        
        if current_data:
            category, emoji, color = get_aqi_category(current_data["aqi"])
            st.markdown(f"""
            <div class="current-aqi-card">
                <p style="font-size: 1.2rem; color: #666; font-weight: 600; margin-bottom: 1rem;">CURRENT AQI - KARACHI</p>
                <div style="font-size: 5rem; margin: 1rem 0;">{emoji}</div>
                <h1 style="color: {color}; font-size: 4.5rem; margin: 1rem 0; font-weight: bold;">{int(current_data['aqi'])}</h1>
                <div style="background: {color}; color: white; padding: 0.7rem 1.5rem; border-radius: 25px; font-size: 1.3rem; font-weight: bold; margin: 1rem auto; display: inline-block;">
                    {category}
                </div>
                <p style="color: #999; font-size: 0.9rem; margin-top: 1rem;">Last updated: {current_data['timestamp'].strftime('%H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🌡️ Particulate Matter")
        if current_data:
            st.markdown(f"""
            <div class="pollutant-card" style="background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);">
                <div class="pollutant-label">PM2.5</div>
                <div class="pollutant-value">{current_data['pm25']:.1f}</div>
                <div class="pollutant-label">μg/m³</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="pollutant-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="pollutant-label">PM10</div>
                <div class="pollutant-value">{current_data['pm10']:.1f}</div>
                <div class="pollutant-label">μg/m³</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### 💨 Gases")
        if current_data:
            st.markdown(f"""
            <div class="pollutant-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <div class="pollutant-label">Carbon Monoxide</div>
                <div class="pollutant-value">{current_data['co']:.1f}</div>
                <div class="pollutant-label">μg/m³</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="pollutant-card" style="background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);">
                <div class="pollutant-label">Ozone (O₃)</div>
                <div class="pollutant-value">{current_data['o3']:.1f}</div>
                <div class="pollutant-label">μg/m³</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Prediction Section
    st.markdown("---")
    st.markdown('<p class="section-header">🔮 3-Day AQI Forecast</p>', unsafe_allow_html=True)
    
    predict_button = st.button("🚀 PREDICT 3-DAY AQI", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("🔄 Fetching data and making predictions..."):
            # Fetch historical data
            historical_df = fetch_historical_data(project)
            
            if historical_df is None or len(historical_df) == 0:
                st.error("No historical data available")
                return
            
            # Engineer features
            features_df = engineer_features(historical_df)
            
            # Load model
            model = load_model(project, model_choice)
            
            if model is None:
                st.error("Failed to load model")
                return
            
            # Predict
            predictions = predict_3_days(model, features_df)
            
            if predictions is None:
                st.error("Failed to make predictions")
                return
            
            # Display predictions
            st.success("✅ Predictions generated successfully!")
            st.markdown("<br>", unsafe_allow_html=True)
            
            pred_values = [predictions["day1"], predictions["day2"], predictions["day3"]]
            
            col1, col2, col3 = st.columns(3)
            
            days_info = [
                ("DAY 1 - TOMORROW", "Tomorrow's AQI", predictions["day1"]),
                ("DAY 2 - DAY AFTER", "Day After Tomorrow", predictions["day2"]),
                ("DAY 3 - 3 DAYS AHEAD", "3 Days from Now", predictions["day3"])
            ]
            
            for col, (day_title, day_subtitle, pred) in zip([col1, col2, col3], days_info):
                with col:
                    category, emoji, color = get_aqi_category(pred)
                    st.markdown(f"""
                    <div class="prediction-card" style="border-left-color: {color};">
                        <div class="day-label" style="color: {color};">{day_title}</div>
                        <div style="font-size: 0.9rem; color: #999; margin-bottom: 1rem;">{day_subtitle}</div>
                        <div style="font-size: 4rem; margin: 1rem 0;">{emoji}</div>
                        <div class="aqi-value" style="color: {color};">{int(pred)}</div>
                        <div class="aqi-label">AQI Value</div>
                        <div class="aqi-category" style="background: {color}; color: white;">
                            {category}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Visualizations
            st.markdown('<p class="section-header">📊 Interactive Visualizations</p>', unsafe_allow_html=True)
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Bar chart
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=["Day 1", "Day 2", "Day 3"],
                        y=pred_values,
                        marker_color=[get_aqi_category(p)[2] for p in pred_values],
                        text=[f"{int(p)}" for p in pred_values],
                        textposition='outside'
                    )
                ])
                fig_bar.update_layout(
                    title="3-Day AQI Forecast",
                    yaxis_title="AQI",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with viz_col2:
                # Gauge chart for current AQI
                if current_data:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=current_data["aqi"],
                        delta={'reference': pred_values[0]},
                        title={'text': "Current AQI vs Tomorrow"},
                        gauge={
                            'axis': {'range': [0, 300]},
                            'bar': {'color': get_aqi_category(current_data["aqi"])[2]},
                            'steps': [
                                {'range': [0, 50], 'color': "#00e400"},
                                {'range': [50, 100], 'color': "#ffff00"},
                                {'range': [100, 150], 'color': "#ff7e00"},
                                {'range': [150, 200], 'color': "#ff0000"},
                                {'range': [200, 300], 'color': "#8f3f97"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': pred_values[0]
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=400)
                    st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Line chart showing trend
            st.markdown("### 📈 AQI Trend")
            
            # Combine current + predictions
            trend_data = pd.DataFrame({
                'Day': ['Current', 'Day 1', 'Day 2', 'Day 3'],
                'AQI': [current_data["aqi"]] + pred_values if current_data else pred_values
            })
            
            fig_line = px.line(
                trend_data,
                x='Day',
                y='AQI',
                markers=True,
                title='AQI Trend (Current → 3-Day Forecast)'
            )
            fig_line.update_traces(line_color='#1f77b4', line_width=3)
            fig_line.update_layout(height=400)
            st.plotly_chart(fig_line, use_container_width=True)
            
            st.markdown("---")
            
            # Model Performance
            st.markdown('<p class="section-header">🎯 Model Performance Metrics</p>', unsafe_allow_html=True)
            
            metrics = get_model_metrics(project, model_choice)
            
            if metrics:
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                
                with met_col1:
                    st.markdown(f"""
                    <div class="performance-metric">
                        <div class="metric-label">Test R² Score</div>
                        <div class="metric-value">{metrics.get('avg_r2', 0):.4f}</div>
                        <div style="font-size: 0.8rem; color: #999;">Model Accuracy</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with met_col2:
                    st.markdown(f"""
                    <div class="performance-metric">
                        <div class="metric-label">Mean Absolute Error</div>
                        <div class="metric-value">{metrics.get('avg_mae', 0):.2f}</div>
                        <div style="font-size: 0.8rem; color: #999;">AQI Units</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with met_col3:
                    train_r2 = metrics.get('train_r2', 0)
                    st.markdown(f"""
                    <div class="performance-metric">
                        <div class="metric-label">Train R² Score</div>
                        <div class="metric-value">{train_r2:.4f}</div>
                        <div style="font-size: 0.8rem; color: #999;">Training Accuracy</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with met_col4:
                    test_r2 = metrics.get('avg_r2', 0)
                    overfit = train_r2 - test_r2
                    overfit_color = "#00e400" if overfit < 0.05 else "#ffff00" if overfit < 0.1 else "#ff0000"
                    overfit_status = "✅ Excellent" if overfit < 0.05 else "⚠️ Good" if overfit < 0.1 else "❌ High"
                    st.markdown(f"""
                    <div class="performance-metric">
                        <div class="metric-label">Overfitting</div>
                        <div class="metric-value" style="color: {overfit_color};">{overfit:.4f}</div>
                        <div style="font-size: 0.8rem; color: #999;">{overfit_status}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Data info
            st.markdown("---")
            st.markdown('<p class="section-header">📦 Data & Model Information</p>', unsafe_allow_html=True)
            
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.markdown(f"""
                <div class="info-box">
                    <h4 style="margin-bottom: 1rem;">📊 Data Source</h4>
                    <p><strong>Feature Store:</strong> Hopsworks</p>
                    <p><strong>Feature Group:</strong> karachi_aqi_raw (v1)</p>
                    <p><strong>Total Rows:</strong> {len(historical_df) if historical_df is not None else 0}</p>
                    <p><strong>Last Updated:</strong> {historical_df['timestamp'].max().strftime('%Y-%m-%d %H:%M') if historical_df is not None else 'N/A'}</p>
                    <p><strong>Update Frequency:</strong> Hourly via CI/CD</p>
                </div>
                """, unsafe_allow_html=True)
            
            with info_col2:
                st.markdown(f"""
                <div class="info-box">
                    <h4 style="margin-bottom: 1rem;">🤖 Model Details</h4>
                    <p><strong>Selected Model:</strong> {model_choice}</p>
                    <p><strong>Features Used:</strong> 12 engineered features</p>
                    <p><strong>Prediction Targets:</strong> 3-day ahead (1d, 2d, 3d)</p>
                    <p><strong>Training Method:</strong> Automated daily</p>
                    <p><strong>Model Registry:</strong> Hopsworks MLOps</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()