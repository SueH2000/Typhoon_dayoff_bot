
"""
train_model.py
=================================

Key details
---------------------------------------
- Imputer (`kNN_imputer.joblib`) input columns (13):
    ['Precp','RH','StnHeight','StnPres','T.Max','T.Min','Temperature',
     'WDGust_vector_x','WDGust_vector_y','WD_vector_x','WD_vector_y','lat','lon']

- Scaler (`MMscaler.joblib`) input columns (24) **and their order**:
    ['Dayoff','Precp','RH','StnHeight','StnPres','T.Max','T.Min','Temperature',
     'TyWS','WDGust_vector_x','WDGust_vector_y','WD_vector_x','WD_vector_y',
     'X10_radius','X7_radius','alert_num','born_spotE','born_spotN','hpa',
     'lat','lon','route_--','route_2','route_3']

  ⚠️ Note the first column is **Dayoff** (today) — it was used as a feature.
     The label to predict is typically **TmrDayoff** (tomorrow).

- Model (`rf_model.joblib`) parameters:
    n_estimators=420, max_depth=45, min_samples_split=6, class_weight='balanced',
    criterion='gini', max_features='auto', ...

What the script does
--------------------
1) Loads XLSX/CSV (auto-detected) and **engineers features** from your columns:
   - Builds one-hots from 'route' → route_3, route_2, route_--.
   - Ensures *all* imputer and scaler columns exist.
   - Orders columns **exactly** like the artifacts expect.

2) Builds a pipeline **matching the existing one**:
   - KNNImputer on the 13 imputed columns (others pass through unchanged).
   - MinMaxScaler on the 24 scaled columns (matched order).
   - RandomForestClassifier with the **same hyperparameters** as your current model.

3) Trains/validates (train/test split), prints metrics, and saves new artifacts:
   - kNN_imputer.joblib
   - MMscaler.joblib
   - rf_model.joblib

Usage
-----
python scripts/train_align_to_existing.py \\
  --data /path/to/data_ver_4_DCT.xlsx \\
  --label TmrDayoff \\
  --output-dir .

If your label is 'Dayoff', set --label Dayoff. The scaler expects 'Dayoff' as a **feature**,
so when predicting 'TmrDayoff' that is perfectly consistent with your old pipeline.
"""

import argparse
import json
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score, classification_report
)


# -----------------------------
# 1) The exact feature schema
# -----------------------------

# Columns imputed by your existing KNNImputer (and the order it expects)
IMPUTER_COLS = [
    'Precp','RH','StnHeight','StnPres','T.Max','T.Min','Temperature',
    'WDGust_vector_x','WDGust_vector_y','WD_vector_x','WD_vector_y','lat','lon'
]

# Columns scaled by your existing MinMaxScaler (and the order it expects)
SCALER_COLS = [
    'Dayoff','Precp','RH','StnHeight','StnPres','T.Max','T.Min','Temperature',
    'TyWS','WDGust_vector_x','WDGust_vector_y','WD_vector_x','WD_vector_y',
    'X10_radius','X7_radius','alert_num','born_spotE','born_spotN','hpa',
    'lat','lon','route_--','route_2','route_3'
]

# Sanity: which columns must be derivable from the raw XLSX?
DERIVED_FROM_ROUTE = ['route_--','route_2','route_3']


def read_any(path: str) -> pd.DataFrame:
    """Read XLSX/CSV by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    elif ext in ('.csv', '.txt'):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def derive_route_onehots(df: pd.DataFrame) -> pd.DataFrame:
    """Create route_3, route_2, route_-- from 'route' column if present."""
    out = df.copy()
    if 'route' in out.columns:
        def _is(v, target):
            try:
                if isinstance(v, str):
                    return v.strip() == str(target)
                return int(v) == int(target)
            except Exception:
                return False
        out['route_3']  = out['route'].apply(lambda v: 1.0 if _is(v, 3) else 0.0)
        out['route_2']  = out['route'].apply(lambda v: 1.0 if _is(v, 2) else 0.0)
        out['route_--'] = out['route'].apply(lambda v: 1.0 if not (_is(v, 3) or _is(v, 2)) else 0.0)
    return out


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Ensure all columns in `cols` exist; if missing, add as NaN."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def build_matrices(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    From the raw dataframe, produce:
    - y label series (binary int)
    - X_imp:  the 2D matrix for the imputer (IMPUTER_COLS, exact order)
    - X_scl:  the 2D matrix for the scaler  (SCALER_COLS, exact order)

    This mirrors your original pipeline: impute a subset, scale the full set.
    """
    # Derive one-hot columns
    df = derive_route_onehots(df)

    # Ensure existence of required columns for both stages
    df = ensure_columns(df, IMPUTER_COLS + SCALER_COLS)

    # Cast numerics (robustly); leave NaN if not coercible
    for c in set(IMPUTER_COLS + SCALER_COLS):
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Label → int {0,1}
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data. Available: {list(df.columns)[:15]} ...")
    y = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int)

    # Extract matrices in the exact order
    X_imp = df[IMPUTER_COLS].copy()
    X_scl = df[SCALER_COLS].copy()

    return df, y, X_imp, X_scl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to data_ver_4_DCT.xlsx or CSV')
    parser.add_argument('--label', default='TmrDayoff', help="Which column to predict (default: TmrDayoff)" )
    parser.add_argument('--test-size', type=float, default=0.2, help='Test fraction for holdout (default: 0.2)')
    parser.add_argument('--output-dir', default='.', help='Where to write joblib artifacts and metrics')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split/model (default: 42)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Read data
    df_raw = read_any(args.data)

    # 2) Build aligned matrices
    df, y, X_imp, X_scl = build_matrices(df_raw, label_col=args.label)

    # 3) Split
    X_imp_train, X_imp_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=args.test_size,
        stratify=y if getattr(y, 'nunique', lambda: 2)() > 1 else None,
        random_state=args.seed
    )
    # Important: we must apply the SAME row split to the scaler input matrix
    X_scl_train = df.loc[X_imp_train.index, SCALER_COLS].copy()
    X_scl_test  = df.loc[X_imp_test.index,  SCALER_COLS].copy()

    # 4) Fit the **same** preprocessor shapes as your old pipeline
    #    - KNNImputer on the 13 imputed columns
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_imp_train_filled = imputer.fit_transform(X_imp_train)
    X_imp_test_filled  = imputer.transform(X_imp_test)

    #    Now we need to REPLACE the imputed columns back into the full scaler matrix
    #    because your original scaler expects the 24 columns in SCALER_COLS order,
    #    which include the 13 imputed columns plus 11 others.
    X_scl_train_filled = X_scl_train.copy()
    X_scl_train_filled[IMPUTER_COLS] = X_imp_train_filled

    X_scl_test_filled = X_scl_test.copy()
    X_scl_test_filled[IMPUTER_COLS] = X_imp_test_filled

    # 5) Fit the **same** scaler (MinMaxScaler on the 24 columns, same order)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_scl_train_filled[SCALER_COLS])
    X_test_scaled  = scaler.transform(X_scl_test_filled[SCALER_COLS])

    # 6) Fit the **same** model hyperparameters (RandomForest in your artifact)
    model = RandomForestClassifier(
        n_estimators=420,
        max_depth=45,
        min_samples_split=6,
        class_weight='balanced',
        criterion='gini',
        max_features='sqrt',  # modern sklearn replacement for 'auto' in classifiers
        random_state=args.seed,
    )
    model.fit(X_train_scaled, y_train)

    # 7) Evaluate
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'avg_precision': float(average_precision_score(y_test, y_proba)),
        'f1': float(f1_score(y_test, y_pred)),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'imputer_columns': IMPUTER_COLS,
        'scaler_columns': SCALER_COLS,
        'label': args.label,
    }

    with open(os.path.join(args.output_dir, 'rf_metrics_aligned.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Write metadata for reproducibility
    try:
        import platform, sklearn
        meta = {
            'imputer_columns': IMPUTER_COLS,
            'scaler_columns': SCALER_COLS,
            'label': args.label,
            'seed': args.seed,
            'versions': {
                'python': platform.python_version(),
                'pandas': pd.__version__,
                'numpy': np.__version__,
                'sklearn': sklearn.__version__,
            }
        }
        with open(os.path.join(args.output_dir, 'artifacts_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 8) Save artifacts with the **same filenames** your runtime expects
    joblib.dump(imputer, os.path.join(args.output_dir, 'kNN_imputer.joblib'))
    joblib.dump(scaler,  os.path.join(args.output_dir, 'MMscaler.joblib'))
    joblib.dump(model,   os.path.join(args.output_dir, 'rf_model.joblib'))

    # 9) Print a concise summary
    print('[OK] Saved artifacts: kNN_imputer.joblib, MMscaler.joblib, rf_model.joblib')
    print(f"[Eval] AUC={metrics['roc_auc']:.3f}  AP={metrics['avg_precision']:.3f}  F1={metrics['f1']:.3f}")
    try:
        importances = model.feature_importances_
        pairs = sorted(zip(SCALER_COLS, importances), key=lambda t: t[1], reverse=True)[:15]
        print('Top features:')
        for name, imp in pairs:
            print(f"  {name:24s} {imp:.4f}")
    except Exception:
        pass


if __name__ == '__main__':
    main()
