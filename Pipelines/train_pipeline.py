# train_pipeline.py
"""
Daily Training Pipeline - Train & Upload Best Model
====================================================
Runs    : Daily at 2 AM via GitHub Actions
Purpose : Fetch raw data from Hopsworks
          Engineer features (rolling, lag, targets)
          Select 12 best features + 3 targets
          Train 3 models (CatBoost, XGBoost, RandomForest)
          Compare metrics (train R2, test R2, MAE, overfitting)
          Upload ONLY best model to Model Registry

Flow:
  1. Fetch ALL raw data from karachi_aqi_raw
  2. Clean data (handle missing, duplicates)
  3. Engineer features (rolling, lag, time-based)
  4. Create targets (shift -24, -48, -72)
  5. Select 12 features + 3 targets
  6. Train 3 models with same logic as train_model.py
  7. Compare models
  8. Upload best to Model Registry (auto-increment version)

Secrets Required:
  - HOPSWORKS_API_KEY (from GitHub Secrets)
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import hopsworks
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# Load .env from project root (parent directory of pipelines/)
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
from dotenv import load_dotenv
load_dotenv(dotenv_path=env_path)

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_NAME  = "karachi_aqipred"
FG_NAME       = "karachi_aqi_raw"
FG_VERSION    = 1
MODEL_NAME    = "karachi_aqi_predictor"
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

# 12 Selected Features (from feature selection analysis)
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

# Target columns
TARGET_COLS   = ["target_aqi_1d", "target_aqi_2d", "target_aqi_3d"]
TARGET_LABELS = ["1 Day Ahead",   "2 Days Ahead",   "3 Days Ahead"]

# Model hyperparameters (same as train_model.py)
CAT_PARAM_GRID = [
    {"iterations": 300, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 1},
    {"iterations": 500, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3},
    {"iterations": 500, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3},
]

XGB_PARAM_GRID = [
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3},
    {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3},
]

RF_PARAM_GRID = [
    {"n_estimators": 50,  "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
]


# ============================================================
# STEP 1: FETCH RAW DATA FROM HOPSWORKS
# ============================================================
def fetch_raw_data():
    """
    Fetch ALL raw data from Hopsworks Feature Store
    """
    print("\n" + "="*60)
    print("STEP 1: FETCHING RAW DATA FROM HOPSWORKS")
    print("="*60)
    
    hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
    
    if not hopsworks_api_key:
        print("  ERROR: HOPSWORKS_API_KEY not found!")
        sys.exit(1)
    
    print("\n  Connecting to Hopsworks...")
    
    try:
        project = hopsworks.login(
            project=PROJECT_NAME,
            api_key_value=hopsworks_api_key
        )
        print("  Connected to: " + project.name)
        
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
        
        print("  Feature Group: " + FG_NAME + " (v" + str(FG_VERSION) + ")")
        print("\n  Reading all data...")
        
        df = fg.read()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        print("  Rows fetched: " + str(len(df)))
        print("  Columns    : " + str(df.columns.tolist()))
        print("  Date range : " + str(df["timestamp"].min()) +
              " to " + str(df["timestamp"].max()))
        
        return df, project
        
    except Exception as e:
        print("  ERROR: Failed to fetch data from Hopsworks")
        print("  " + str(e))
        sys.exit(1)


# ============================================================
# STEP 2: CLEAN DATA
# ============================================================
def clean_data(df):
    """
    Handle missing values and duplicates
    Same logic as data_cleaning.py
    """
    print("\n" + "="*60)
    print("STEP 2: CLEANING DATA")
    print("="*60)
    
    original_rows = len(df)
    
    # Remove duplicates based on timestamp
    print("\n  Checking for duplicates...")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    dupes_removed = original_rows - len(df)
    print("  Duplicates removed: " + str(dupes_removed))
    
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
        # Forward fill then backward fill
        df = df.ffill()
        df = df.bfill()
        
        remaining_nulls = df.isnull().sum().sum()
        print("  Remaining nulls after fill: " + str(remaining_nulls))
        
        # Drop any rows that still have nulls
        if remaining_nulls > 0:
            df = df.dropna()
            print("  Dropped " + str(remaining_nulls) + " rows with nulls")
    else:
        print("  No missing values found")
    
    print("\n  Final shape after cleaning: " + str(df.shape))
    
    return df


# ============================================================
# STEP 3: ENGINEER FEATURES
# ============================================================
def engineer_features(df):
    """
    Create engineered features from raw data
    - Rolling features (3h, 6h, 12h, 24h)
    - Lag features (1h, 3h, 6h)
    - Target features (1d, 2d, 3d ahead)
    """
    print("\n" + "="*60)
    print("STEP 3: ENGINEERING FEATURES")
    print("="*60)
    
    print("\n  Creating rolling features...")
    df['aqi_rolling_max_24h']  = df['aqi'].rolling(window=24, min_periods=1).max()
    df['aqi_rolling_mean_3h']  = df['aqi'].rolling(window=3,  min_periods=1).mean()
    df['aqi_rolling_mean_6h']  = df['aqi'].rolling(window=6,  min_periods=1).mean()
    df['aqi_rolling_mean_12h'] = df['aqi'].rolling(window=12, min_periods=1).mean()
    print("    - aqi_rolling_max_24h")
    print("    - aqi_rolling_mean_3h")
    print("    - aqi_rolling_mean_6h")
    print("    - aqi_rolling_mean_12h")
    
    print("\n  Creating lag features...")
    df['aqi_lag_1h'] = df['aqi'].shift(1)
    df['aqi_lag_3h'] = df['aqi'].shift(3)
    df['aqi_lag_6h'] = df['aqi'].shift(6)
    print("    - aqi_lag_1h")
    print("    - aqi_lag_3h")
    print("    - aqi_lag_6h")
    
    print("\n  Creating target features...")
    df['target_aqi_1d'] = df['aqi'].shift(-24)  # 24 hours ahead
    df['target_aqi_2d'] = df['aqi'].shift(-48)  # 48 hours ahead
    df['target_aqi_3d'] = df['aqi'].shift(-72)  # 72 hours ahead
    print("    - target_aqi_1d (24h ahead)")
    print("    - target_aqi_2d (48h ahead)")
    print("    - target_aqi_3d (72h ahead)")
    
    # Drop rows with NaN in targets or features
    print("\n  Dropping rows with NaN values...")
    before_drop = len(df)
    
    # Drop rows where lag features are NaN (first 6 rows)
    # Drop rows where targets are NaN (last 72 rows)
    df = df.dropna()
    
    after_drop = len(df)
    rows_dropped = before_drop - after_drop
    
    print("  Rows before: " + str(before_drop))
    print("  Rows after : " + str(after_drop))
    print("  Dropped    : " + str(rows_dropped) + 
          " (first 24 + last 72 rows)")
    
    return df


# ============================================================
# STEP 4: SELECT FEATURES
# ============================================================
def select_features(df):
    """
    Keep only the 12 selected features + 3 targets + timestamp
    """
    print("\n" + "="*60)
    print("STEP 4: SELECTING FEATURES")
    print("="*60)
    
    all_cols = ["timestamp"] + SELECTED_FEATURES + TARGET_COLS
    
    print("\n  Keeping " + str(len(SELECTED_FEATURES)) + " features:")
    for i, feat in enumerate(SELECTED_FEATURES, 1):
        print("    " + str(i).rjust(2) + ". " + feat)
    
    print("\n  Plus 3 targets:")
    for i, target in enumerate(TARGET_COLS, 1):
        print("    " + str(i) + ". " + target)
    
    df_selected = df[all_cols].copy()
    
    print("\n  Final shape: " + str(df_selected.shape))
    print("  Columns: " + str(len(df_selected.columns)))
    
    return df_selected


# ============================================================
# HELPER: EVALUATE MODEL
# ============================================================
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model on test set
    Returns metrics dict
    """
    y_pred = model.predict(X_test)
    if hasattr(y_pred, "values"):
        y_pred = y_pred.values
    y_pred = np.array(y_pred)
    
    print("\n  " + "="*52)
    print("  " + model_name + " - Test Results:")
    print("  " + "="*52)
    print("  " + "Target".ljust(20) +
          "MAE".rjust(8) + "RMSE".rjust(8) + "R2".rjust(9))
    print("  " + "-"*52)
    
    all_mae, all_rmse, all_r2 = [], [], []
    
    for i, (col, label) in enumerate(zip(TARGET_COLS, TARGET_LABELS)):
        mae  = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2   = r2_score(y_test[:, i], y_pred[:, i])
        
        all_mae.append(mae)
        all_rmse.append(rmse)
        all_r2.append(r2)
        
        status = "GOOD" if r2 >= 0.7 else "OK" if r2 >= 0.5 else "BAD"
        print("  " + label.ljust(20) +
              str(round(mae,  2)).rjust(8) +
              str(round(rmse, 2)).rjust(8) +
              str(round(r2,   4)).rjust(9) +
              "  [" + status + "]")
    
    avg_mae  = float(np.mean(all_mae))
    avg_rmse = float(np.mean(all_rmse))
    avg_r2   = float(np.mean(all_r2))
    
    print("  " + "-"*52)
    print("  " + "AVERAGE".ljust(20) +
          str(round(avg_mae,  2)).rjust(8) +
          str(round(avg_rmse, 2)).rjust(8) +
          str(round(avg_r2,   4)).rjust(9))
    
    return {
        "avg_mae":  round(avg_mae,  3),
        "avg_rmse": round(avg_rmse, 3),
        "avg_r2":   round(avg_r2,   4),
    }


# ============================================================
# HELPER: TUNE + TRAIN CATBOOST
# ============================================================
def tune_catboost(X_train, y_train, X_val, y_val):
    from catboost import CatBoostRegressor
    
    print("\n  Trying " + str(len(CAT_PARAM_GRID)) + " param combos...")
    
    best_r2 = -np.inf
    best_params = None
    
    for i, params in enumerate(CAT_PARAM_GRID, 1):
        model = CatBoostRegressor(
            **params,
            random_seed=RANDOM_STATE,
            verbose=False,
            loss_function="MultiRMSE",
            allow_writing_files=False,
        )
        model.fit(X_train, y_train)
        y_pred = np.array(model.predict(X_val))
        
        r2 = float(np.mean([
            r2_score(y_val[:, j], y_pred[:, j])
            for j in range(y_val.shape[1])
        ]))
        
        status = "GOOD" if r2 >= 0.7 else "OK" if r2 >= 0.5 else "BAD"
        print("     Combo " + str(i) + "/" + str(len(CAT_PARAM_GRID)) +
              ": R2 = " + str(round(r2, 4)) + "  [" + status + "]")
        
        if r2 > best_r2:
            best_r2 = r2
            best_params = params
    
    print("\n  Best R2: " + str(round(best_r2, 4)))
    print("  Retraining on full train set...")
    
    final_model = CatBoostRegressor(
        **best_params,
        random_seed=RANDOM_STATE,
        verbose=False,
        loss_function="MultiRMSE",
        allow_writing_files=False,
    )
    final_model.fit(X_train, y_train)
    
    return final_model, best_params


# ============================================================
# HELPER: TUNE + TRAIN SKLEARN MODELS
# ============================================================
def tune_sklearn(model_fn_factory, param_grid, X_train, y_train, 
                 X_val, y_val, name):
    
    print("\n  Trying " + str(len(param_grid)) + " param combos...")
    
    best_r2 = -np.inf
    best_params = None
    best_fn = None
    
    for i, params in enumerate(param_grid, 1):
        model = model_fn_factory(params)()
        model.fit(X_train, y_train)
        y_pred = np.array(model.predict(X_val))
        
        r2 = float(np.mean([
            r2_score(y_val[:, j], y_pred[:, j])
            for j in range(y_val.shape[1])
        ]))
        
        status = "GOOD" if r2 >= 0.7 else "OK" if r2 >= 0.5 else "BAD"
        print("     Combo " + str(i) + "/" + str(len(param_grid)) +
              ": R2 = " + str(round(r2, 4)) + "  [" + status + "]")
        
        if r2 > best_r2:
            best_r2 = r2
            best_params = params
            best_fn = model_fn_factory(params)
    
    print("\n  Best R2: " + str(round(best_r2, 4)))
    print("  Retraining on full train set...")
    
    final_model = best_fn()
    final_model.fit(X_train, y_train)
    
    return final_model, best_params


# ============================================================
# STEP 5: TRAIN MODELS
# ============================================================
def train_models(df):
    """
    Train CatBoost, XGBoost, RandomForest
    Same logic as train_model.py
    """
    print("\n" + "="*60)
    print("STEP 5: TRAINING MODELS")
    print("="*60)
    
    # Prepare data
    X = df[SELECTED_FEATURES].values
    y = df[TARGET_COLS].values
    
    # Shuffle split (same as train_model.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )
    
    print("\n  Train rows: " + str(len(X_train)))
    print("  Test rows : " + str(len(X_test)))
    print("  NO SCALING (tree models don't need it)")
    
    # Create validation split for tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True
    )
    
    print("  Tuning split: " + str(len(X_tr)) + " train / " +
          str(len(X_val)) + " val")
    
    # Train CatBoost
    print("\n" + "-"*60)
    print("  CATBOOST")
    print("-"*60)
    
    try:
        from catboost import CatBoostRegressor
        cat_model, cat_params = tune_catboost(X_tr, y_tr, X_val, y_val)
        
        # Evaluate on full train and test
        cat_train_metrics = evaluate_model(cat_model, X_train, y_train, 
                                           "CatBoost - TRAINING")
        cat_metrics = evaluate_model(cat_model, X_test, y_test, 
                                     "CatBoost - TEST")
        cat_metrics["train_r2"] = cat_train_metrics["avg_r2"]
        
    except ImportError:
        print("  ERROR: CatBoost not installed")
        sys.exit(1)
    
    # Train XGBoost
    print("\n" + "-"*60)
    print("  XGBOOST")
    print("-"*60)
    
    try:
        from xgboost import XGBRegressor
        
        def xgb_fn_factory(params):
            def make():
                return MultiOutputRegressor(
                    XGBRegressor(**params, random_state=RANDOM_STATE, 
                                verbosity=0),
                    n_jobs=1
                )
            return make
        
        xgb_model, xgb_params = tune_sklearn(
            xgb_fn_factory, XGB_PARAM_GRID,
            X_tr, y_tr, X_val, y_val, "XGBoost"
        )
        
        xgb_train_metrics = evaluate_model(xgb_model, X_train, y_train,
                                          "XGBoost - TRAINING")
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test,
                                    "XGBoost - TEST")
        xgb_metrics["train_r2"] = xgb_train_metrics["avg_r2"]
        
    except ImportError:
        print("  ERROR: XGBoost not installed")
        sys.exit(1)
    
    # Train RandomForest
    print("\n" + "-"*60)
    print("  RANDOM FOREST")
    print("-"*60)
    
    def rf_fn_factory(params):
        def make():
            return MultiOutputRegressor(
                RandomForestRegressor(**params, random_state=RANDOM_STATE,
                                    n_jobs=1),
                n_jobs=1
            )
        return make
    
    rf_model, rf_params = tune_sklearn(
        rf_fn_factory, RF_PARAM_GRID,
        X_tr, y_tr, X_val, y_val, "RandomForest"
    )
    
    rf_train_metrics = evaluate_model(rf_model, X_train, y_train,
                                     "RandomForest - TRAINING")
    rf_metrics = evaluate_model(rf_model, X_test, y_test,
                               "RandomForest - TEST")
    rf_metrics["train_r2"] = rf_train_metrics["avg_r2"]
    
    # Return all models
    results = {
        "CatBoost": {
            "model": cat_model,
            "metrics": cat_metrics,
            "params": cat_params,
        },
        "XGBoost": {
            "model": xgb_model,
            "metrics": xgb_metrics,
            "params": xgb_params,
        },
        "RandomForest": {
            "model": rf_model,
            "metrics": rf_metrics,
            "params": rf_params,
        },
    }
    
    return results, SELECTED_FEATURES


# ============================================================
# STEP 6: COMPARE MODELS & SELECT BEST
# ============================================================
def compare_and_select_best(results):
    """
    Compare all 3 models and select best
    """
    print("\n" + "="*60)
    print("STEP 6: COMPARING MODELS")
    print("="*60)
    
    ranked = sorted(
        results.items(),
        key=lambda x: x[1]["metrics"]["avg_r2"],
        reverse=True
    )
    medals = ["1st", "2nd", "3rd"]
    
    print("\n  " + "-"*72)
    print("  " + "Rank".ljust(6) + "Model".ljust(16) + 
          "Train R2".rjust(10) + "Test R2".rjust(10) + 
          "Overfit".rjust(10) + "MAE".rjust(10))
    print("  " + "-"*72)
    
    for i, (name, res) in enumerate(ranked):
        m = res["metrics"]
        overfit = round(m["train_r2"] - m["avg_r2"], 4)
        print("  " + medals[i] + "  " + name.ljust(16) +
              str(round(m["train_r2"], 4)).rjust(10) +
              str(round(m["avg_r2"],  4)).rjust(10) +
              str(overfit).rjust(10) +
              str(round(m["avg_mae"],  2)).rjust(10))
    
    print("  " + "-"*72)
    
    best_name = ranked[0][0]
    best_model = ranked[0][1]["model"]
    best_metrics = ranked[0][1]["metrics"]
    best_params = ranked[0][1]["params"]
    
    print("\n  WINNER: " + best_name)
    print("  Train R2    : " + str(best_metrics["train_r2"]))
    print("  Test R2     : " + str(best_metrics["avg_r2"]))
    print("  Overfitting : " + 
          str(round(best_metrics["train_r2"] - best_metrics["avg_r2"], 4)))
    print("  Avg MAE     : " + str(best_metrics["avg_mae"]) + " AQI units")
    
    return best_name, best_model, best_metrics, best_params


# ============================================================
# STEP 7: UPLOAD TO MODEL REGISTRY
# ============================================================
def upload_to_registry(project, model_name, model, metrics, 
                       params, feature_cols):
    """
    Upload best model to Hopsworks Model Registry
    Auto-increment version
    """
    print("\n" + "="*60)
    print("STEP 7: UPLOADING TO MODEL REGISTRY")
    print("="*60)
    
    print("\n  Model: " + model_name)
    
    try:
        mr = project.get_model_registry()
        
        # Get next version number
        try:
            existing_models = mr.get_models(name=MODEL_NAME)
            if existing_models and len(existing_models) > 0:
                latest_version = max([m.version for m in existing_models])
                next_version = latest_version + 1
            else:
                next_version = 1
        except:
            next_version = 1
        
        print("  Version: " + str(next_version))
        
        # Create temporary directory for model files
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Save model
        model_path = os.path.join(temp_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save feature columns
        features_path = os.path.join(temp_dir, "feature_cols.pkl")
        with open(features_path, "wb") as f:
            pickle.dump(feature_cols, f)
        
        # Save config
        config = {
            "model_name": model_name,
            "trained_at": datetime.now().isoformat(),
            "features": feature_cols,
            "targets": TARGET_COLS,
            "test_r2": metrics["avg_r2"],
            "train_r2": metrics["train_r2"],
            "avg_mae": metrics["avg_mae"],
            "scaling": "none",
            "split": "shuffle",
        }
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Upload to registry
        model_reg = mr.python.create_model(
            name=MODEL_NAME,
            version=next_version,
            metrics={
                "test_r2": metrics["avg_r2"],
                "train_r2": metrics["train_r2"],
                "avg_mae": metrics["avg_mae"],
            },
            description=(
                "Best Model: " + model_name + ". "
                "Karachi AQI 1d/2d/3d predictor. "
                "Test R2: " + str(metrics["avg_r2"]) + ". "
                "Trained: " + datetime.now().strftime("%Y-%m-%d")
            ),
        )
        
        model_reg.save(temp_dir)
        
        print("\n  Upload successful!")
        print("  Registry name: " + MODEL_NAME + " (v" + str(next_version) + ")")
        print("  Test R2      : " + str(metrics["avg_r2"]))
        print("  Avg MAE      : " + str(metrics["avg_mae"]))
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        print("  ERROR: Failed to upload to registry")
        print("  " + str(e))
        sys.exit(1)


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    """
    Main training pipeline
    Runs daily via GitHub Actions
    """
    print("\n" + "="*60)
    print("DAILY TRAINING PIPELINE - STARTED")
    print("="*60)
    print("  Time: " + str(datetime.now()))
    
    try:
        # Step 1: Fetch raw data
        df_raw, project = fetch_raw_data()
        
        # Step 2: Clean data
        df_clean = clean_data(df_raw)
        
        # Step 3: Engineer features
        df_features = engineer_features(df_clean)
        
        # Step 4: Select features
        df_selected = select_features(df_features)
        
        # Step 5: Train models
        results, feature_cols = train_models(df_selected)
        
        # Step 6: Compare and select best
        best_name, best_model, best_metrics, best_params = \
            compare_and_select_best(results)
        
        # Step 7: Upload to registry
        upload_to_registry(project, best_name, best_model, 
                          best_metrics, best_params, feature_cols)
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("  Best Model : " + best_name)
        print("  Test R2    : " + str(best_metrics["avg_r2"]))
        print("  Next run   : Tomorrow at 2 AM")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("PIPELINE FAILED")
        print("="*60)
        print("  Error: " + str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()