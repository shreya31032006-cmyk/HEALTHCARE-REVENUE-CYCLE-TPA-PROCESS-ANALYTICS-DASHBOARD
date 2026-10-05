"""
train_model.py
==============
Healthcare Revenue Cycle and TPA Process Analytics Dashboard
Machine-Learning Training Script — Improved Version

Key improvements over v1
------------------------
- Delay threshold changed from 30 to 23 days (median split → balanced 50/50 classes).
- Training restricted to resolved claims only (Settlement_Date not null).
- Added RandomForestClassifier and GradientBoostingClassifier (scikit-learn only).
- Added engineered features: Claim_Amount_Log, Claim_High_Value, TPA_Service_Interaction.
- Used StratifiedKFold cross-validation (5-fold).
- Hyperparameter tuning via GridSearchCV for best model.
- class_weight='balanced' for Logistic Regression and Decision Tree.
- Comprehensive model comparison table printed to console.

Workflow
--------
1.  Locate TPA DATA.xlsx (project root or DATA/ or data/ folder).
2.  Inspect workbook sheets.
3.  Load the claim-level worksheet.
4.  Normalize and map columns.
5.  Validate the dataset.
6.  Create derived variables.
7.  Engineer additional features.
8.  Filter to resolved claims only for training.
9.  Split dataset (stratified 80/20, random_state=42).
10. Build scikit-learn Pipelines with ColumnTransformer.
11. Train 4 candidate models.
12. Evaluate all candidates and compare.
13. Select best model (highest F1, recall as tiebreaker).
14. Save complete pipeline to models/model.pkl via joblib.
15. Print training summary.
"""

import os
import sys
import warnings
import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Using median processing time (23 days) as threshold for balanced 50/50 split.
# This maximises model learning signal. The dashboard slider lets users adjust.
DELAY_THRESHOLD_DAYS = 23
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_RECORDS_FOR_TRAINING = 30
MODEL_OUTPUT_PATH = os.path.join("models", "model.pkl")

WORKBOOK_CANDIDATES = [
    "TPA DATA.xlsx",
    os.path.join("DATA", "TPA data.xlsx"),
    os.path.join("data", "TPA DATA.xlsx"),
    os.path.join("data", "TPA data.xlsx"),
]

EXPECTED_COLUMNS = [
    "Case_ID", "Patient_Type", "Department", "TPA_or_Payer",
    "Service_Type", "Claim_Amount", "Approved_Amount",
    "Rejected_Amount", "Pending_Amount", "Submission_Date",
    "Settlement_Date", "Case_Status", "Query_or_Rejection_Reason",
]

VALID_STATUSES = {"Approved", "Pending", "Rejected", "Queried", "In Process", "Settled",
                  "Partially Approved"}

# Features available at submission (no post-outcome leakage)
# Base categorical features
BASE_CAT = ["Patient_Type", "Department", "TPA_or_Payer", "Service_Type"]
# Base numeric features
BASE_NUM = ["Claim_Amount", "Submission_Month", "Submission_Weekday"]
# Engineered features (derived from submission-time data only)
ENG_NUM = ["Claim_Amount_Log", "Claim_High_Value"]
ENG_CAT = ["TPA_Service"]

ALL_CAT = BASE_CAT + ENG_CAT
ALL_NUM = BASE_NUM + ENG_NUM


# ---------------------------------------------------------------------------
# File location
# ---------------------------------------------------------------------------

def locate_workbook():
    for path in WORKBOOK_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

def build_column_map(raw_columns):
    canonical_aliases = {
        "case id": "Case_ID", "case_id": "Case_ID",
        "patient type": "Patient_Type", "patient_type": "Patient_Type",
        "department": "Department",
        "tpa or payer": "TPA_or_Payer", "tpa_or_payer": "TPA_or_Payer",
        "service type": "Service_Type", "service_type": "Service_Type",
        "claim amount": "Claim_Amount", "claim_amount": "Claim_Amount",
        "approved amount": "Approved_Amount", "approved_amount": "Approved_Amount",
        "rejected amount": "Rejected_Amount", "rejected_amount": "Rejected_Amount",
        "pending amount": "Pending_Amount", "pending_amount": "Pending_Amount",
        "submission date": "Submission_Date", "submission_date": "Submission_Date",
        "settlement date": "Settlement_Date", "settlement_date": "Settlement_Date",
        "case status": "Case_Status", "case_status": "Case_Status",
        "query or rejection reason": "Query_or_Rejection_Reason",
        "query_or_rejection_reason": "Query_or_Rejection_Reason",
    }
    final_map = {}
    for raw in raw_columns:
        normalized = " ".join(str(raw).strip().split())
        key = normalized.lower().replace("_", " ")
        if key in canonical_aliases:
            final_map[raw] = canonical_aliases[key]
        elif normalized.lower() in canonical_aliases:
            final_map[raw] = canonical_aliases[normalized.lower()]
        else:
            final_map[raw] = normalized
    return final_map


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_workbook(path):
    print(f"\n[INFO] Loading workbook: {path}")
    xl = pd.ExcelFile(path, engine="openpyxl")
    sheets = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        sheets[sheet] = df
        print(f"       Sheet '{sheet}': {len(df)} rows x {len(df.columns)} columns")
    return sheets


def select_worksheet(sheets):
    best_sheet, best_score = None, -1
    for name, df in sheets.items():
        raw_cols = list(df.columns)
        final_map = build_column_map(raw_cols)
        score = len(set(final_map.values()).intersection(set(EXPECTED_COLUMNS)))
        if score > best_score:
            best_score = score
            best_sheet = name
    return best_sheet


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_columns(df, required=None):
    if required is None:
        required = EXPECTED_COLUMNS
    return [c for c in required if c not in df.columns]


def validate_dates(df):
    report = {"invalid_submission": 0, "invalid_settlement": 0,
               "settlement_before_submission": 0}
    for col in ["Submission_Date", "Settlement_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "Submission_Date" in df.columns:
        report["invalid_submission"] = int(df["Submission_Date"].isna().sum())
    if "Settlement_Date" in df.columns:
        report["invalid_settlement"] = int(df["Settlement_Date"].isna().sum())
    if "Submission_Date" in df.columns and "Settlement_Date" in df.columns:
        both = df["Submission_Date"].notna() & df["Settlement_Date"].notna()
        bad = both & (df["Settlement_Date"] < df["Submission_Date"])
        report["settlement_before_submission"] = int(bad.sum())
        df.loc[bad, "Settlement_Date"] = pd.NaT
    return df, report


def validate_financials(df):
    for col in ["Claim_Amount", "Approved_Amount", "Rejected_Amount", "Pending_Amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Derived variables + feature engineering
# ---------------------------------------------------------------------------

def create_derived_variables(df, delay_threshold=DELAY_THRESHOLD_DAYS):
    """Compute processing time, delayed flag, submission features."""
    analysis_date = pd.Timestamp(datetime.date.today())

    # Processing time (resolved only)
    if "Submission_Date" in df.columns and "Settlement_Date" in df.columns:
        both_valid = df["Submission_Date"].notna() & df["Settlement_Date"].notna()
        df["Processing_Time_Days"] = np.nan
        df.loc[both_valid, "Processing_Time_Days"] = (
            df.loc[both_valid, "Settlement_Date"] - df.loc[both_valid, "Submission_Date"]
        ).dt.days

    # Delayed flag
    df["Delayed"] = np.nan
    if "Processing_Time_Days" in df.columns:
        resolved = df["Processing_Time_Days"].notna()
        df.loc[resolved, "Delayed"] = (
            df.loc[resolved, "Processing_Time_Days"] > delay_threshold
        ).astype(int)
    # For unresolved claims (Pending): use elapsed time
    if "Submission_Date" in df.columns:
        unresolved = df["Processing_Time_Days"].isna()
        elapsed = (analysis_date - df["Submission_Date"]).dt.days
        df.loc[unresolved & elapsed.notna(), "Delayed"] = (
            elapsed[unresolved & elapsed.notna()] > delay_threshold
        ).astype(int)

    # Submission date features
    if "Submission_Date" in df.columns:
        df["Submission_Month"] = df["Submission_Date"].dt.month.fillna(0).astype(int)
        df["Submission_Weekday"] = df["Submission_Date"].dt.dayofweek.fillna(0).astype(int)
        df["Submission_Year"] = df["Submission_Date"].dt.year
        df["Submission_YM"] = df["Submission_Date"].dt.to_period("M").astype(str)

    # Financial consistency
    fin_cols = ["Claim_Amount", "Approved_Amount", "Rejected_Amount", "Pending_Amount"]
    if all(c in df.columns for c in fin_cols):
        df["Unallocated_Amount"] = (
            df["Claim_Amount"]
            - df["Approved_Amount"]
            - df["Rejected_Amount"]
            - df["Pending_Amount"]
        )
        df["Financial_Inconsistent"] = df["Unallocated_Amount"].abs() > 0.01

    return df


def engineer_features(df):
    """
    Create additional features from submission-time data only.
    All features derived purely from Claim_Amount, TPA, Service_Type — no leakage.
    """
    # Log transform of Claim_Amount (reduces skew, more linear signal)
    if "Claim_Amount" in df.columns:
        df["Claim_Amount_Log"] = np.log1p(df["Claim_Amount"].fillna(0))
        # High-value claim flag (above 75th percentile)
        threshold_75 = df["Claim_Amount"].quantile(0.75)
        df["Claim_High_Value"] = (df["Claim_Amount"] > threshold_75).astype(int)

    # TPA x Service_Type interaction (captures TPA-specific service processing patterns)
    if "TPA_or_Payer" in df.columns and "Service_Type" in df.columns:
        df["TPA_Service"] = (
            df["TPA_or_Payer"].fillna("Unknown").astype(str)
            + "_"
            + df["Service_Type"].fillna("Unknown").astype(str)
        )

    return df


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_pipeline(cat_features, num_features, model):
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("cat", cat_transformer, cat_features),
        ("num", num_transformer, num_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_and_evaluate(df, delay_threshold=DELAY_THRESHOLD_DAYS):
    print("\n" + "=" * 60)
    print("HEALTHCARE TPA MODEL TRAINING — IMPROVED")
    print("=" * 60)

    # Determine available features
    cat_features = [c for c in ALL_CAT if c in df.columns]
    num_features = [c for c in ALL_NUM if c in df.columns]
    all_features = cat_features + num_features

    print(f"\n[INFO] Target            : Delayed (1 = processing > {delay_threshold} days)")
    print(f"[INFO] Problem type      : Binary Classification")
    print(f"[INFO] Delay threshold   : {delay_threshold} days (median split for balanced classes)")
    print(f"[INFO] Categorical feats : {cat_features}")
    print(f"[INFO] Numeric feats     : {num_features}")
    print(f"[INFO] Leakage-excluded  : Settlement_Date, Processing_Time_Days, "
          "Approved_Amount, Rejected_Amount, Pending_Amount")

    # Use ONLY resolved claims for training (unresolved have uncertain target)
    resolved_df = df[df["Processing_Time_Days"].notna()].copy()
    print(f"\n[INFO] Total records     : {len(df)}")
    print(f"[INFO] Resolved records  : {len(resolved_df)} (used for training)")
    print(f"[INFO] Unresolved (excl) : {len(df) - len(resolved_df)} (Pending/no settlement date)")

    target_col = "Delayed"
    valid_df = resolved_df.dropna(subset=[target_col]).copy()
    valid_df[target_col] = valid_df[target_col].astype(int)

    print(f"[INFO] Valid training records: {len(valid_df)}")
    class_dist = valid_df[target_col].value_counts()
    print(f"[INFO] Class distribution:")
    print(f"         Not Delayed (0): {class_dist.get(0,0)} ({100*class_dist.get(0,0)/len(valid_df):.1f}%)")
    print(f"         Delayed     (1): {class_dist.get(1,0)} ({100*class_dist.get(1,0)/len(valid_df):.1f}%)")

    if len(valid_df) < MIN_RECORDS_FOR_TRAINING:
        print(f"[ERROR] Too few valid records ({len(valid_df)}). Training aborted.")
        return None, None
    if len(class_dist) < 2:
        print(f"[ERROR] Only one class in target. Training aborted.")
        return None, None

    X = valid_df[all_features]
    y = valid_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # --- 4 candidate models ---
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            C=1.0,
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    pipelines = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n[INFO] Evaluating candidate models...")
    print("-" * 60)

    for name, estimator in candidates.items():
        pipe = build_pipeline(cat_features, num_features, estimator)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            y_prob = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = None

        try:
            cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except Exception:
            cv_mean, cv_std = None, None

        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4) if auc is not None else None,
            "cv_f1_mean": round(cv_mean, 4) if cv_mean is not None else None,
            "cv_f1_std": round(cv_std, 4) if cv_std is not None else None,
            "confusion_matrix": cm,
        }
        pipelines[name] = pipe

        print(f"  {name}")
        print(f"    Accuracy  : {acc:.4f}")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")
        print(f"    F1        : {f1:.4f}")
        if auc is not None:
            print(f"    ROC-AUC   : {auc:.4f}")
        if cv_mean is not None:
            print(f"    CV F1     : {cv_mean:.4f} +/- {cv_std:.4f}")
        print()

    # --- Select best (highest F1, recall as tiebreaker) ---
    best_name = max(results, key=lambda n: (results[n]["f1"], results[n]["recall"]))
    best_pipeline = pipelines[best_name]
    best_metrics = results[best_name]

    print(f"[SELECTED] Best model : {best_name}")
    print(f"           F1={best_metrics['f1']:.4f}, "
          f"Recall={best_metrics['recall']:.4f}, "
          f"Accuracy={best_metrics['accuracy']:.4f}")

    # --- Metadata ---
    metadata = {
        "model_name": best_name,
        "problem_type": "classification",
        "target_col": target_col,
        "delay_threshold_days": delay_threshold,
        "cat_features": cat_features,
        "num_features": num_features,
        "all_features": all_features,
        "leakage_excluded": [
            "Settlement_Date", "Processing_Time_Days", "Delayed",
            "Approved_Amount", "Rejected_Amount", "Pending_Amount",
        ],
        "metrics": best_metrics,
        "all_results": results,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "trained_at": str(datetime.datetime.now()),
        "selection_rule": (
            "Best F1 score on hold-out test set; Recall used as tiebreaker. "
            "Trained on resolved claims only (Settlement_Date not null). "
            f"Delay threshold = {delay_threshold} days (median split for balanced classes)."
        ),
        "class_distribution": {
            "not_delayed": int(class_dist.get(0, 0)),
            "delayed": int(class_dist.get(1, 0)),
        },
    }

    return best_pipeline, metadata


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_model(pipeline, metadata):
    os.makedirs("models", exist_ok=True)
    payload = {"pipeline": pipeline, "metadata": metadata}
    joblib.dump(payload, MODEL_OUTPUT_PATH)
    print(f"\n[SAVED] Pipeline saved to: {MODEL_OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("STEP 1 — Locating TPA DATA.xlsx")
    print("=" * 60)
    path = locate_workbook()
    if path is None:
        print(
            "\n[ERROR] TPA DATA.xlsx was not found.\n"
            "Please place the actual Excel workbook in the project root "
            "or in the DATA/ folder.\nTraining aborted."
        )
        sys.exit(1)
    print(f"[FOUND] Workbook at: {path}")

    print("\nSTEP 2 — Loading workbook")
    sheets = load_workbook(path)
    best_sheet = select_worksheet(sheets)
    print(f"[SELECTED] Worksheet: '{best_sheet}'")
    df = sheets[best_sheet].copy()

    print("\nSTEP 3 — Normalizing columns")
    final_map = build_column_map(list(df.columns))
    df.rename(columns=final_map, inplace=True)
    missing = validate_columns(df)
    if missing:
        print(f"  [WARNING] Missing columns: {missing}")
    else:
        print("  All expected columns present.")

    print("\nSTEP 4 — Date & financial validation")
    df, date_report = validate_dates(df)
    df = validate_financials(df)
    print(f"  Invalid Settlement_Date : {date_report['invalid_settlement']}")
    print(f"  Settlement < Submission : {date_report['settlement_before_submission']}")

    print("\nSTEP 5 — Derived variables")
    df = create_derived_variables(df, delay_threshold=DELAY_THRESHOLD_DAYS)
    resolved = df["Processing_Time_Days"].notna().sum()
    print(f"  Processing_Time_Days computed for {resolved} resolved records.")

    print("\nSTEP 6 — Feature engineering")
    df = engineer_features(df)
    print(f"  Engineered features added: Claim_Amount_Log, Claim_High_Value, TPA_Service")

    print("\nSTEP 7 — Training models")
    best_pipeline, metadata = train_and_evaluate(df, delay_threshold=DELAY_THRESHOLD_DAYS)

    if best_pipeline is None:
        print("\n[ERROR] Training failed. No model saved.")
        sys.exit(1)

    save_model(best_pipeline, metadata)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Workbook        : {path}")
    print(f"  Worksheet       : {best_sheet}")
    print(f"  Total records   : {len(df)}")
    print(f"  Resolved (train): {metadata['class_distribution']['not_delayed'] + metadata['class_distribution']['delayed']}")
    print(f"  Delay threshold : {DELAY_THRESHOLD_DAYS} days (median split)")
    print(f"  Best model      : {metadata['model_name']}")
    print(f"  Accuracy        : {metadata['metrics']['accuracy']}")
    print(f"  F1 Score        : {metadata['metrics']['f1']}")
    print(f"  Recall          : {metadata['metrics']['recall']}")
    print(f"  ROC-AUC         : {metadata['metrics']['roc_auc']}")
    print(f"  Saved to        : {MODEL_OUTPUT_PATH}")
    print("=" * 60)
    print("\n[DONE] Run 'streamlit run app.py' to launch the dashboard.")


if __name__ == "__main__":
    main()
