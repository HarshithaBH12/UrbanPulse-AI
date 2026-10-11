from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["hour", "is_weekend", "temperature", "rainfall", "aqi", "event_factor"]
TARGET_COLUMN = "traffic_index"

def train_or_load_model(city, df):
    # Train reproducibly on the generated dataset for each app session.
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=140, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, "Random Forest · synthetic training data"

def evaluate_model(model, df):
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS], df[TARGET_COLUMN], test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred))
    }

def predict_traffic(model, values):
    row = {
        "hour": values.get("hour", 12),
        "is_weekend": values.get("is_weekend", 0),
        "temperature": values.get("temperature", 27),
        "rainfall": values.get("rainfall", 0),
        "aqi": values.get("aqi") if values.get("aqi") is not None else 50,
        "event_factor": values.get("event_factor", 0)
    }
    import pandas as pd
    frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return float(np.clip(model.predict(frame)[0], 0, 100))
