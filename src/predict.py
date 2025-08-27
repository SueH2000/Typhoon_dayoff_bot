
"""
Predictor that mirrors the training pipeline:
- Impute only the 13 'imputer columns'
- Insert imputed values back into the full feature frame
- Scale the 24 'scaler columns' in the exact order
- Predict proba with the trained classifier

"""
import os
import joblib
import numpy as np
import pandas as pd

# Expected columns for each stage, aligned to your stored artifacts
IMPUTER_COLS = [
    'Precp','RH','StnHeight','StnPres','T.Max','T.Min','Temperature',
    'WDGust_vector_x','WDGust_vector_y','WD_vector_x','WD_vector_y','lat','lon'
]
SCALER_COLS = [
    'Dayoff','Precp','RH','StnHeight','StnPres','T.Max','T.Min','Temperature',
    'TyWS','WDGust_vector_x','WDGust_vector_y','WD_vector_x','WD_vector_y',
    'X10_radius','X7_radius','alert_num','born_spotE','born_spotN','hpa',
    'lat','lon','route_--','route_2','route_3'
]

def _resolve_path(env_key: str, default_in_models: str, legacy_root: str) -> str:
    """Resolve artifact path with backward-compatible fallback.
    Order:
      1) env var if set
      2) default in models/ if exists
      3) legacy root filename if exists
    else return default in models/ (to produce a clear error later).
    """
    p = os.getenv(env_key)
    if p:
        return p
    if os.path.exists(default_in_models):
        return default_in_models
    if os.path.exists(legacy_root):
        return legacy_root
    return default_in_models

KNN_IMPUTER_PATH = _resolve_path("KNN_IMPUTER_PATH", "models/kNN_imputer.joblib", "kNN_imputer.joblib")
MINMAX_SCALER_PATH = _resolve_path("MINMAX_SCALER_PATH", "models/MMscaler.joblib", "MMscaler.joblib")
MODEL_PATH = _resolve_path("MODEL_PATH", "models/rf_model.joblib", "rf_model.joblib")

class Predictor:
    def __init__(self):
        for p in [KNN_IMPUTER_PATH, MINMAX_SCALER_PATH, MODEL_PATH]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing artifact: {p}. Set env vars KNN_IMPUTER_PATH, MINMAX_SCALER_PATH, MODEL_PATH or place files under models/.")
        self.imputer = joblib.load(KNN_IMPUTER_PATH)
        self.scaler = joblib.load(MINMAX_SCALER_PATH)
        self.model = joblib.load(MODEL_PATH)

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in set(IMPUTER_COLS + SCALER_COLS):
            if c not in out.columns:
                out[c] = np.nan
        # Numeric casting
        for c in set(IMPUTER_COLS + SCALER_COLS):
            out[c] = pd.to_numeric(out[c], errors='coerce')
        return out

    def predict_proba_dayoff(self, df_features: pd.DataFrame) -> float:
        """Return P(class=1) using the aligned pipeline."""
        df = self._ensure_columns(df_features)

        # 1) Impute on the 13-column subset
        X_imp = df[IMPUTER_COLS].copy()
        X_imp_filled = self.imputer.transform(X_imp)

        # 2) Insert imputed values back into the full feature frame
        X_full = df[SCALER_COLS].copy()
        X_full[IMPUTER_COLS] = X_imp_filled

        # 3) Scale in the exact scaler order
        X_scaled = self.scaler.transform(X_full[SCALER_COLS])

        # 4) Predict probability of positive class
        proba = self.model.predict_proba(X_scaled)[0][1]
        return float(proba)
