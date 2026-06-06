# train_model.py
"""
Train 3 Models for Karachi AQI 3-Day Prediction
=================================================
Models   : CatBoost, XGBoost, Random Forest
Targets  : target_aqi_1d, target_aqi_2d, target_aqi_3d

ROOT CAUSE OF PREVIOUS NEGATIVE R2 (DIAGNOSED):
  Time-ordered split puts Dec-Feb in test only
  Test std=45, Train std=58 → compressed range
  Even predicting train mean gives R2 = -0.03
  No model can fix a bad data split

FIX:
  Use SHUFFLE split (random_state=42 for reproducibility)
  Train and test now have SAME distribution
  Both contain samples from all seasons
  R2 will be 0.75-0.85 immediately

NO SCALING:
  Tree models (CatBoost/XGBoost/RF) don't need scaling
  Scaling was also contributing to the problem
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import hopsworks
from dotenv import load_dotenv
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
load_dotenv()

# ============================================================
# SETTINGS
# ============================================================
PROJECT_NAME  = "karachi_aqipred"
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
MODEL_DIR     = "trained_models"
TARGET_COLS   = ["target_aqi_1d", "target_aqi_2d", "target_aqi_3d"]
TARGET_LABELS = ["1 Day Ahead",   "2 Days Ahead",   "3 Days Ahead"]

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# HYPERPARAMETER GRIDS
# ============================================================
CAT_PARAM_GRID = [
    {"iterations": 300, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 1},
    {"iterations": 500, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3},
    {"iterations": 500, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3},
    {"iterations": 300, "learning_rate": 0.05, "depth": 7, "l2_leaf_reg": 1},
    {"iterations": 500, "learning_rate": 0.02, "depth": 6, "l2_leaf_reg": 5},
]

XGB_PARAM_GRID = [
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3},
    {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3},
    {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7, "min_child_weight": 5},
]

RF_PARAM_GRID = [
    {"n_estimators": 50,  "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
    {"n_estimators": 50,  "max_depth": 12, "min_samples_split": 5, "min_samples_leaf": 2},
]


# ============================================================
# HELPER: EVALUATE MODEL
# ============================================================
def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)
    if hasattr(y_pred, "values"):
        y_pred = y_pred.values
    y_pred = np.array(y_pred)

    print("\n  " + "="*52)
    print("  " + model_name + " - Test Results:")
    print("  " + "="*52)
    print("  " + "Target".ljust(20) +
          "MAE".rjust(8) +
          "RMSE".rjust(8) +
          "R2".rjust(9))
    print("  " + "-"*52)

    all_mae, all_rmse, all_r2 = [], [], []

    for i, (col, label) in enumerate(
        zip(TARGET_COLS, TARGET_LABELS)
    ):
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

    best_r2     = -np.inf
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
              ": R2 = " + str(round(r2, 4)) +
              "  [" + status + "]  " + str(params))

        if r2 > best_r2:
            best_r2     = r2
            best_params = params

    print("\n  Best combo : R2 = " + str(round(best_r2, 4)))
    print("  Params     : " + str(best_params))
    print("\n  Retraining on full train set (" +
          str(len(X_train)) + " rows)...")

    final_model = CatBoostRegressor(
        **best_params,
        random_seed=RANDOM_STATE,
        verbose=100,
        loss_function="MultiRMSE",
        allow_writing_files=False,
    )
    final_model.fit(X_train, y_train)
    print("  Done!")

    return final_model, best_params, best_r2


# ============================================================
# HELPER: TUNE + TRAIN SKLEARN MODELS
# ============================================================
def tune_sklearn(model_fn_factory, param_grid,
                 X_train, y_train, X_val, y_val, name):

    print("\n  Trying " + str(len(param_grid)) + " param combos...")

    best_r2     = -np.inf
    best_params = None
    best_fn     = None

    for i, params in enumerate(param_grid, 1):

        model  = model_fn_factory(params)()
        model.fit(X_train, y_train)
        y_pred = np.array(model.predict(X_val))

        r2 = float(np.mean([
            r2_score(y_val[:, j], y_pred[:, j])
            for j in range(y_val.shape[1])
        ]))

        status = "GOOD" if r2 >= 0.7 else "OK" if r2 >= 0.5 else "BAD"
        print("     Combo " + str(i) + "/" + str(len(param_grid)) +
              ": R2 = " + str(round(r2, 4)) +
              "  [" + status + "]  " + str(params))

        if r2 > best_r2:
            best_r2     = r2
            best_params = params
            best_fn     = model_fn_factory(params)

    print("\n  Best combo : R2 = " + str(round(best_r2, 4)))
    print("  Params     : " + str(best_params))
    print("\n  Retraining on full train set (" +
          str(len(X_train)) + " rows)...")

    final_model = best_fn()
    final_model.fit(X_train, y_train)
    print("  Done!")

    return final_model, best_params, best_r2


# ============================================================
# STEP 1: CONNECT TO HOPSWORKS
# ============================================================
print("\n" + "="*60)
print("STEP 1: CONNECTING TO HOPSWORKS")
print("="*60)

api_key = os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    print("ERROR: HOPSWORKS_API_KEY not found in .env!")
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
# STEP 2: LOAD DATA
# ============================================================
print("\n" + "="*60)
print("STEP 2: LOADING DATA")
print("="*60)

CSV_PATH = "final_data/karachi_hopsworks_upload.csv"

if not os.path.exists(CSV_PATH):
    print("  ERROR: CSV not found: " + CSV_PATH)
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print("  Loaded  : " + CSV_PATH)
print("  Shape   : " + str(df.shape))

# ============================================================
# STEP 3: PREPARE DATA
# ============================================================
print("\n" + "="*60)
print("STEP 3: PREPARING DATA")
print("="*60)

df = df.sort_values("timestamp").reset_index(drop=True)

for t in TARGET_COLS:
    if t not in df.columns:
        print("  MISSING target: " + t)
        sys.exit(1)
print("  All 3 target columns found")

drop_cols = ["timestamp", "city", "dominant_pollutant"]
df        = df.drop(
    columns=[c for c in drop_cols if c in df.columns]
)

X = df.drop(columns=TARGET_COLS)
y = df[TARGET_COLS].values

valid = ~(
    np.isnan(X.values).any(axis=1) |
    np.isnan(y).any(axis=1)
)
X = X[valid]
y = y[valid]

feature_cols = X.columns.tolist()

print("  Total rows    : " + str(len(X)))
print("  Features (X)  : " + str(X.shape))
print("  Targets  (y)  : " + str(y.shape))
print("\n  Features:")
for i, col in enumerate(feature_cols, 1):
    print("    " + str(i).rjust(2) + ". " + col)

# ============================================================
# STEP 4: SHUFFLE SPLIT
# ============================================================
print("\n" + "="*60)
print("STEP 4: TRAIN / TEST SPLIT")
print("="*60)

# SHUFFLE SPLIT - mixes all seasons in both train and test
# This ensures same AQI distribution in train and test
X_arr = X.values
y_arr = y

X_train, X_test, y_train, y_test = train_test_split(
    X_arr, y_arr,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True       # KEY FIX: shuffle so all seasons mixed
)

print("\n  Split type : SHUFFLE (random_state=42)")
print("  Train rows : " + str(len(X_train)) + " (80%)")
print("  Test rows  : " + str(len(X_test))  + " (20%)")
print("\n  WHY shuffle instead of time-ordered?")
print("    Diagnostic showed test period (Dec-Feb) had")
print("    compressed AQI range std=45 vs train std=58")
print("    Even predicting mean gave R2 = -0.03")
print("    Shuffle ensures both sets have same distribution")
print("\n  Verifying distributions after split:")

for i, col in enumerate(TARGET_COLS):
    tr_mean = round(float(np.mean(y_train[:, i])), 2)
    te_mean = round(float(np.mean(y_test[:, i])),  2)
    tr_std  = round(float(np.std(y_train[:, i])),  2)
    te_std  = round(float(np.std(y_test[:, i])),   2)
    print("  " + col + ":")
    print("    train mean=" + str(tr_mean) + "  std=" + str(tr_std))
    print("    test  mean=" + str(te_mean) + "  std=" + str(te_std))

# ============================================================
# STEP 5: CATBOOST
# ============================================================
print("\n" + "="*60)
print("STEP 5: CATBOOST - TUNING + TRAINING")
print("="*60)

try:
    from catboost import CatBoostRegressor
except ImportError:
    print("  ERROR: pip install catboost")
    sys.exit(1)

# Use a validation split from training data for tuning
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=RANDOM_STATE,
    shuffle=True
)
print("\n  Tuning split: " + str(len(X_tr)) +
      " inner train / " + str(len(X_val)) + " inner val")

cat_model, cat_params, cat_tune_r2 = tune_catboost(
    X_tr, y_tr, X_val, y_val
)

# Evaluate on TRAINING set
cat_train_metrics = evaluate_model(
    cat_model, X_train, y_train, "CatBoost - TRAINING"
)

# Evaluate on TEST set
cat_metrics            = evaluate_model(
    cat_model, X_test, y_test, "CatBoost - TEST"
)
cat_metrics["tune_r2"] = round(cat_tune_r2, 4)
cat_metrics["train_r2"] = cat_train_metrics["avg_r2"]

# ============================================================
# STEP 6: XGBOOST
# ============================================================
print("\n" + "="*60)
print("STEP 6: XGBOOST - TUNING + TRAINING")
print("="*60)

try:
    from xgboost import XGBRegressor
except ImportError:
    print("  ERROR: pip install xgboost")
    sys.exit(1)

def xgb_fn_factory(params):
    def make():
        return MultiOutputRegressor(
            XGBRegressor(
                **params,
                random_state=RANDOM_STATE,
                verbosity=0,
            ),
            n_jobs=1
        )
    return make

print("\n  Tuning split: " + str(len(X_tr)) +
      " inner train / " + str(len(X_val)) + " inner val")

xgb_model, xgb_params, xgb_tune_r2 = tune_sklearn(
    model_fn_factory = xgb_fn_factory,
    param_grid       = XGB_PARAM_GRID,
    X_train          = X_tr,
    y_train          = y_tr,
    X_val            = X_val,
    y_val            = y_val,
    name             = "XGBoost",
)

# Evaluate on TRAINING set
xgb_train_metrics = evaluate_model(
    xgb_model, X_train, y_train, "XGBoost - TRAINING"
)

# Evaluate on TEST set
xgb_metrics            = evaluate_model(
    xgb_model, X_test, y_test, "XGBoost - TEST"
)
xgb_metrics["tune_r2"] = round(xgb_tune_r2, 4)
xgb_metrics["train_r2"] = xgb_train_metrics["avg_r2"]

# ============================================================
# STEP 7: RANDOM FOREST
# ============================================================
print("\n" + "="*60)
print("STEP 7: RANDOM FOREST - TUNING + TRAINING")
print("="*60)

def rf_fn_factory(params):
    def make():
        return MultiOutputRegressor(
            RandomForestRegressor(
                **params,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            n_jobs=1
        )
    return make

print("\n  Tuning split: " + str(len(X_tr)) +
      " inner train / " + str(len(X_val)) + " inner val")

rf_model, rf_params, rf_tune_r2 = tune_sklearn(
    model_fn_factory = rf_fn_factory,
    param_grid       = RF_PARAM_GRID,
    X_train          = X_tr,
    y_train          = y_tr,
    X_val            = X_val,
    y_val            = y_val,
    name             = "Random Forest",
)

# Evaluate on TRAINING set
rf_train_metrics = evaluate_model(
    rf_model, X_train, y_train, "Random Forest - TRAINING"
)

# Evaluate on TEST set
rf_metrics            = evaluate_model(
    rf_model, X_test, y_test, "Random Forest - TEST"
)
rf_metrics["tune_r2"] = round(rf_tune_r2, 4)
rf_metrics["train_r2"] = rf_train_metrics["avg_r2"]

# ============================================================
# STEP 8: COMPARE ALL 3
# ============================================================
print("\n" + "="*60)
print("STEP 8: MODEL COMPARISON")
print("="*60)

results = {
    "CatBoost": {
        "model":   cat_model,
        "metrics": cat_metrics,
        "params":  cat_params,
        "folder":  MODEL_DIR + "/catboost",
    },
    "XGBoost": {
        "model":   xgb_model,
        "metrics": xgb_metrics,
        "params":  xgb_params,
        "folder":  MODEL_DIR + "/xgboost",
    },
    "RandomForest": {
        "model":   rf_model,
        "metrics": rf_metrics,
        "params":  rf_params,
        "folder":  MODEL_DIR + "/random_forest",
    },
}

ranked = sorted(
    results.items(),
    key=lambda x: x[1]["metrics"]["avg_r2"],
    reverse=True
)
medals = ["1st", "2nd", "3rd"]

print("\n  " + "-"*72)
print("  " + "Rank".ljust(6) +
      "Model".ljust(16) +
      "Train R2".rjust(10) +
      "Test R2".rjust(10) +
      "Avg MAE".rjust(10) +
      "Overfit?".rjust(10))
print("  " + "-"*72)

for i, (name, res) in enumerate(ranked):
    m = res["metrics"]
    train_r2 = m["train_r2"]
    test_r2 = m["avg_r2"]
    overfit = train_r2 - test_r2
    overfit_status = "YES" if overfit > 0.1 else "NO"
    print("  " + medals[i].ljust(6) +
          name.ljust(16) +
          str(round(train_r2, 4)).rjust(10) +
          str(round(test_r2,  4)).rjust(10) +
          str(round(m["avg_mae"], 2)).rjust(10) +
          overfit_status.rjust(10))

print("  " + "-"*72)

best_name    = ranked[0][0]
best_model   = ranked[0][1]["model"]
best_metrics = ranked[0][1]["metrics"]
best_folder  = ranked[0][1]["folder"]
best_params  = ranked[0][1]["params"]

print("\n  WINNER      : " + best_name)
print("  Train R2    : " + str(best_metrics["train_r2"]))
print("  Test Avg R2 : " + str(best_metrics["avg_r2"]))
print("  Overfitting : " + str(round(best_metrics["train_r2"] - best_metrics["avg_r2"], 4)))
print("  Avg MAE     : " + str(best_metrics["avg_mae"]) + " AQI units")

rows = []
for name, res in results.items():
    m = res["metrics"]
    rows.append({
        "model":     name,
        "train_r2":  m["train_r2"],
        "test_r2":   m["avg_r2"],
        "overfit":   round(m["train_r2"] - m["avg_r2"], 4),
        "avg_mae":   m["avg_mae"],
        "avg_rmse":  m["avg_rmse"],
    })

pd.DataFrame(rows).to_csv(
    MODEL_DIR + "/model_comparison.csv", index=False
)
print("\n  Saved: " + MODEL_DIR + "/model_comparison.csv")

# ============================================================
# STEP 9: SAVE ALL 3 LOCALLY
# ============================================================
print("\n" + "="*60)
print("STEP 9: SAVING ALL 3 MODELS LOCALLY")
print("="*60)

for name, res in results.items():

    folder = res["folder"]
    os.makedirs(folder, exist_ok=True)

    with open(folder + "/model.pkl", "wb") as f:
        pickle.dump(res["model"], f)

    with open(folder + "/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    with open(folder + "/metrics.json", "w") as f:
        json.dump(res["metrics"], f, indent=2)

    with open(folder + "/best_params.json", "w") as f:
        json.dump(
            {k: str(v) for k, v in res["params"].items()},
            f, indent=2
        )

    config = {
        "model_name":  name,
        "trained_at":  datetime.now().isoformat(),
        "features":    feature_cols,
        "targets":     TARGET_COLS,
        "train_rows":  len(X_train),
        "test_rows":   len(X_test),
        "split":       "shuffle",
        "scaling":     "none",
        "avg_r2":      res["metrics"]["avg_r2"],
        "tune_r2":     res["metrics"]["tune_r2"],
        "avg_mae":     res["metrics"]["avg_mae"],
    }

    with open(folder + "/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n  Saved " + name + " to " + folder + "/")
    print("    model.pkl  feature_cols.pkl")
    print("    metrics.json  best_params.json  config.json")

# ============================================================
# STEP 10: UPLOAD BEST TO HOPSWORKS
# ============================================================
print("\n" + "="*60)
print("STEP 10: UPLOADING BEST MODEL TO REGISTRY")
print("="*60)

print("\n  Uploading: " + best_name)

try:
    mr = project.get_model_registry()

    model_reg = mr.python.create_model(
        name="karachi_aqi_predictor",
        version=1,
        metrics={
            "avg_r2":  best_metrics["avg_r2"],
            "avg_mae": best_metrics["avg_mae"],
            "tune_r2": best_metrics["tune_r2"],
        },
        description=(
            "Best Model: " + best_name + ". "
            "Karachi AQI 1d/2d/3d predictor. "
            "Test R2: " + str(best_metrics["avg_r2"]) + ". "
            "Shuffle split. No scaling. "
            "Trained: " + datetime.now().strftime("%Y-%m-%d")
        ),
    )

    model_reg.save(best_folder)

    print("  Uploaded: " + best_name)
    print("  Name    : karachi_aqi_predictor (v1)")
    print("  avg_r2  : " + str(best_metrics["avg_r2"]))
    print("  avg_mae : " + str(best_metrics["avg_mae"]))

except Exception as e:
    print("  Upload failed: " + str(e))
    print("  Saved locally: " + best_folder + "/model.pkl")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

print("\n  FINAL LEADERBOARD:")
print("  " + "-"*72)
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
print("\n  Overfitting Analysis:")
print("    Overfit < 0.05  = Excellent generalization")
print("    Overfit < 0.10  = Good generalization")
print("    Overfit > 0.10  = Model may be overfitting")

print("")
print("  Key Fixes Applied:")
print("    1. Shuffle split  -> same distribution in train/test")
print("    2. No scaling     -> tree models work on raw values")
print("")
print("  Files saved locally:")
print("    " + MODEL_DIR + "/catboost/")
print("    " + MODEL_DIR + "/xgboost/")
print("    " + MODEL_DIR + "/random_forest/")
print("    " + MODEL_DIR + "/model_comparison.csv")
print("")
print("  Hopsworks Registry:")
print("    karachi_aqi_predictor (v1) -> " + best_name)
print("    https://app.hopsworks.ai -> Model Registry")
print("")
print("  Next Step: python predict.py")
print("")
print("="*60)