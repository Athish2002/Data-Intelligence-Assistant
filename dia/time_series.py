"""
dia/time_series.py
──────────────────
Automated Time-Series & Temporal Forecasting Engine.
Detects temporal trends, generates lag/rolling features, and trains
autoregressive forecast models with prediction intervals.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

log = logging.getLogger("dia.time_series")


def detect_time_series_column(df: pd.DataFrame) -> str | None:
    """
    Identifies a datetime column or monotonically increasing temporal column.
    """
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        # Try parsing strings as date if sample matches ISO/Date patterns
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(20)
            try:
                converted = pd.to_datetime(sample, errors="coerce")
                if converted.notna().sum() > len(sample) * 0.8:
                    return col
            except Exception:  # noqa: S110 — routine "is this parseable as a date" probe, not a failure
                pass
    return None


def generate_time_series_features(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    max_lags: int = 7,
    rolling_windows: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Creates temporal lag, rolling statistics, and cyclical calendar features.
    """
    windows = rolling_windows or [3, 7]
    df_ts = df.copy()

    # Convert and sort chronologically
    df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors="coerce")
    df_ts = df_ts.sort_values(by=date_col).reset_index(drop=True)

    # Calendar features
    df_ts["ts_dayofweek"] = df_ts[date_col].dt.dayofweek
    df_ts["ts_month"] = df_ts[date_col].dt.month
    df_ts["ts_day"] = df_ts[date_col].dt.day
    df_ts["ts_is_weekend"] = df_ts["ts_dayofweek"].isin([5, 6]).astype(int)

    feature_cols = ["ts_dayofweek", "ts_month", "ts_day", "ts_is_weekend"]

    # Autoregressive Lags
    for lag in range(1, max_lags + 1):
        col_name = f"lag_{lag}"
        df_ts[col_name] = df_ts[target_col].shift(lag)
        feature_cols.append(col_name)

    # Rolling window aggregations
    for w in windows:
        col_mean = f"rolling_mean_{w}"
        col_std = f"rolling_std_{w}"
        df_ts[col_mean] = df_ts[target_col].shift(1).rolling(window=w, min_periods=1).mean()
        df_ts[col_std] = df_ts[target_col].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
        feature_cols.extend([col_mean, col_std])

    # Drop NaNs produced by shifting
    df_clean = df_ts.dropna(subset=feature_cols).reset_index(drop=True)
    return df_clean, feature_cols


def train_time_series_forecaster(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    forecast_horizon: int = 14,
) -> dict[str, Any]:
    """
    Trains a temporal autoregressive forecaster and generates forward projections
    with upper and lower uncertainty bounds.
    """
    df_ts, feature_cols = generate_time_series_features(df, date_col, target_col)

    if len(df_ts) < 20:
        return {
            "status": "insufficient_data",
            "message": "At least 20 chronological observations required for time-series forecasting.",
        }

    # Chronological Train-Test Split (Last 20% for test)
    split_idx = int(len(df_ts) * 0.8)
    train_df = df_ts.iloc[:split_idx]
    test_df = df_ts.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    r2 = r2_score(y_test, test_preds)
    std_residuals = float(np.std(y_test - test_preds))

    # Multi-step future recursive forecasting
    future_forecasts = []
    last_known_features = df_ts[feature_cols].iloc[-1].values.copy()
    last_date = df_ts[date_col].iloc[-1]

    # Infer frequency
    time_diff = df_ts[date_col].diff().median()
    if pd.isna(time_diff) or time_diff.total_seconds() == 0:
        step_delta = pd.Timedelta(days=1)
    else:
        step_delta = time_diff

    current_date = last_date
    current_features = last_known_features.copy()

    for step in range(1, forecast_horizon + 1):
        current_date += step_delta
        y_pred = float(model.predict(current_features.reshape(1, -1))[0])

        # Uncertainty intervals (95% confidence bounds = +/- 1.96 * sigma)
        uncertainty = 1.96 * std_residuals * np.sqrt(1 + 0.05 * step)

        future_forecasts.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "forecast_value": round(y_pred, 2),
            "lower_bound_95": round(y_pred - uncertainty, 2),
            "upper_bound_95": round(y_pred + uncertainty, 2),
        })

        # Shift lags forward
        current_features[0] = current_date.dayofweek
        current_features[1] = current_date.month
        current_features[2] = current_date.day
        current_features[3] = 1 if current_date.dayofweek in [5, 6] else 0
        # shift lags: lag_1 becomes y_pred, lag_2 becomes previous lag_1, etc.
        # lag columns start at index 4
        current_features[4] = y_pred

    return {
        "status": "success",
        "date_column": date_col,
        "target_column": target_col,
        "evaluation": {
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "r2_score": round(float(r2), 3),
            "residual_std": round(std_residuals, 3),
        },
        "forecast_horizon_steps": forecast_horizon,
        "future_projections": future_forecasts,
        "historical_dates": df_ts[date_col].dt.strftime("%Y-%m-%d").tolist(),
        "historical_actuals": df_ts[target_col].tolist(),
    }
