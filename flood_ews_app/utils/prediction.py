import json
import math
import os
import warnings

import numpy as np
import pandas as pd

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

_pipeline = None
_model_meta = {}
_sigmoid_params = None
_city_profiles = None
_model_loaded = False


def _load_model():
    global _pipeline, _model_meta, _sigmoid_params, _model_loaded
    if _model_loaded:
        return

    # Try known joblib filenames
    for fname in ("flood_model.joblib", "flood_ews_model.joblib"):
        joblib_path = os.path.join(_MODEL_DIR, fname)
        if os.path.exists(joblib_path):
            import joblib
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = joblib.load(joblib_path)
            if isinstance(data, dict):
                _pipeline = data["model"]
                _model_meta = data
            else:
                _pipeline = data
                _model_meta = {}
            _model_loaded = True
            return

    # Fallback: analytical sigmoid
    coeffs_path = os.path.join(_MODEL_DIR, "model_coefficients.json")
    with open(coeffs_path) as f:
        _sigmoid_params = json.load(f)
    _model_loaded = True


def _load_city_profiles() -> pd.DataFrame:
    global _city_profiles
    if _city_profiles is None:
        path = os.path.join(_DATA_DIR, "city_profiles.csv")
        _city_profiles = pd.read_csv(path).set_index("city")
    return _city_profiles


def _base_probability(rainfall_mm: float, antecedent_3day: float = 0.0) -> float:
    _load_model()
    if _pipeline is not None:
        features = _model_meta.get("features", ["Rainfall_mm"])
        if len(features) == 2:
            X = pd.DataFrame([[rainfall_mm, antecedent_3day]], columns=features)
        else:
            X = pd.DataFrame([[rainfall_mm]], columns=features)
        return float(_pipeline.predict_proba(X)[0][1])
    # Sigmoid fallback (single-feature)
    midpoint = _sigmoid_params["midpoint"]
    k = _sigmoid_params["steepness_k"]
    return 1.0 / (1.0 + math.exp(-k * (rainfall_mm - midpoint)))


def predict_flood_probability(
    rainfall_mm: float, city: str, antecedent_3day: float = 0.0
) -> float:
    """Return adjusted flood probability [0,1] for a given rainfall and city."""
    profiles = _load_city_profiles()
    base_prob = _base_probability(rainfall_mm, antecedent_3day)
    adjustment = (
        float(profiles.loc[city, "adjustment_factor"]) if city in profiles.index else 1.0
    )
    return min(max(base_prob * adjustment, 0.0), 1.0)


def get_city_info(city: str) -> dict:
    profiles = _load_city_profiles()
    if city in profiles.index:
        row = profiles.loc[city]
        return {
            "adjustment_factor": float(row["adjustment_factor"]),
            "description": row["description"],
        }
    return {"adjustment_factor": 1.0, "description": "No profile available."}


def get_optimal_threshold() -> float:
    """Return optimal rainfall trigger threshold in mm/day."""
    _load_model()
    if _model_meta:
        return float(_model_meta.get("optimal_threshold_mm", 23.5))
    return float(_sigmoid_params.get("optimal_threshold_mm", 21.5))


def sigmoid_curve_data(
    city: str | None = None, antecedent_3day: float = 0.0
) -> tuple[list[float], list[float]]:
    """Return (x_values, y_values) for the probability curve over 0–100 mm.

    antecedent_3day is held fixed so the curve shows only rainfall sensitivity.
    """
    x = list(range(0, 101))
    if city:
        y = [predict_flood_probability(xi, city, antecedent_3day) for xi in x]
    else:
        _load_model()
        y = [_base_probability(xi, antecedent_3day) for xi in x]
    return x, y


def is_using_fallback() -> bool:
    _load_model()
    return _pipeline is None


def get_model_features() -> list[str]:
    _load_model()
    if _model_meta:
        return _model_meta.get("features", ["Rainfall_mm"])
    return ["Rainfall_mm"]
