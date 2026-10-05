"""
train_model.py
==============
Healthcare Revenue Cycle and TPA Process Analytics Dashboard
Machine-Learning Training Script

Workflow
--------
1.  Locate TPA DATA.xlsx (project root or data/ folder).
2.  Inspect workbook sheets.
3.  Load the claim-level worksheet.
4.  Normalize and map columns.
5.  Validate the dataset (file, columns, dates, amounts, status, duplicates).
6.  Create derived variables (Processing_Time_Days, Delayed).
7.  Automatically identify target and problem type.
8.  Prevent data leakage by excluding post-outcome features.
9.  Split dataset (train/test, random_state=42).
10. Build scikit-learn Pipelines with ColumnTransformer.
11. Train candidate models (LogisticRegression vs DecisionTreeClassifier).
12. Evaluate all candidates and compare.
13. Select the best-performing model (highest F1 score, recall as tiebreaker).
14. Save the complete pipeline to models/model.pkl via joblib.
15. Print a clear training summary.
"""

import os
import sys
import warnings
import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DELAY_THRESHOLD_DAYS = 30
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
    "Case_ID",
    "Patient_Type",
    "Department",
    "TPA_or_Payer",
    "Service_Type",
    "Claim_Amount",
    "Approved_Amount",
    "Rejected_Amount",
    "Pending_Amount",
    "Submission_Date",
    "Settlement_Date",
    "Case_Status",
    "Query_or_Rejection_Reason",
]

VALID_STATUSES = {"Approved", "Pending", "Rejected", "Queried", "In Process", "Settled"}

# Features available at or near submission (no post-outcome leakage)
CLASSIFICATION_FEATURES_CAT = ["Patient_Type", "Department", "TPA_or_Payer", "Service_Type"]
CLASSIFICATION_FEATURES_NUM = ["Claim_Amount", "Submission_Month", "Submission_Weekday"]

# ---------------------------------------------------------------------------
# File location
# ---------------------------------------------------------------------------

def locate_workbook():
    """Return the relative path to TPA DATA.xlsx, or None if not found."""
    for path in WORKBOOK_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

def normalize_column_name(col):
    """Strip whitespace, collapse spaces, title-case, replace spaces with underscores."""
    col = str(col).strip()
    col = " ".join(col.split())   # collapse repeated spaces
    return col


def build_column_map(raw_columns):
    """
    Map raw column names to expected names.
    Returns (normalized_map, final_map) where:
      normalized_map: raw -> normalized
      final_map:      raw -> expected (if unambiguous mapping exists)
    """
    canonical_aliases = {
        "case id": "Case_ID",
        "case_id": "Case_ID",
        "patient type": "Patient_Type",
        "patient_type": "Patient_Type",
        "department": "Department",
        "tpa or payer": "TPA_or_Payer",
        "tpa_or_payer": "TPA_or_Payer",
        "service type": "Service_Type",
        "service_type": "Service_Type",
        "claim amount": "Claim_Amount",
        "claim_amount": "Claim_Amount",
        "approved amount": "Approved_Amount",
        "approved_amount": "Approved_Amount",
        "rejected amount": "Rejected_Amount",
        "rejected_amount": "Rejected_Amount",
        "pending amount": "Pending_Amount",
        "pending_amount": "Pending_Amount",
        "submission date": "Submission_Date",
        "submission_date": "Submission_Date",
        "settlement date": "Settlement_Date",
        "settlement_date": "Settlement_Date",
        "case status": "Case_Status",
        "case_status": "Case_Status",
        "query or rejection reason": "Query_or_Rejection_Reason",
        "query_or_rejection_reason": "Query_or_Rejection_Reason",
    }

    normalized_map = {}
    final_map = {}

    for raw in raw_columns:
        normalized = normalize_column_name(raw)
        normalized_map[raw] = normalized
        key = normalized.lower().replace("_", " ")
        if key in canonical_aliases:
            final_map[raw] = canonical_aliases[key]
        else:
            # try underscore form
            key2 = normalized.lower()
            if key2 in canonical_aliases:
                final_map[raw] = canonical_aliases[key2]
            else:
                final_map[raw] = normalized  # keep normalized as-is

    return normalized_map, final_map


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_workbook(path):
    """Load the workbook and return a dict of {sheet_name: DataFrame}."""
    print(f"\n[INFO] Loading workbook: {path}")
    xl = pd.ExcelFile(path, engine="openpyxl")
    sheets = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        sheets[sheet] = df
        print(f"       Sheet '{sheet}': {len(df)} rows × {len(df.columns)} columns")
    return sheets


def select_worksheet(sheets):
    """Return the sheet with the most expected column matches."""
    best_sheet = None
    best_score = -1
    for name, df in sheets.items():
        raw_cols = list(df.columns)
        _, final_map = build_column_map(raw_cols)
        mapped = set(final_map.values())
        score = len(mapped.intersection(set(EXPECTED_COLUMNS)))
        if score > best_score:
            best_score = score
            best_sheet = name
    return best_sheet


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_columns(df, required=None):
    """Return list of missing required columns."""
    if required is None:
        required = EXPECTED_COLUMNS
    missing = [c for c in required if c not in df.columns]
    return missing


def validate_dates(df):
    """
    Convert date columns, detect invalids, detect settlement < submission.
    Returns modified df and a report dict.
    """
    report = {
        "invalid_submission": 0,
        "invalid_settlement": 0,
        "settlement_before_submission": 0,
        "excluded_from_dates": 0,
    }

    for col in ["Submission_Date", "Settlement_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Submission_Date" in df.columns:
        report["invalid_submission"] = df["Submission_Date"].isna().sum()
    if "Settlement_Date" in df.columns:
        report["invalid_settlement"] = df["Settlement_Date"].isna().sum()

    if "Submission_Date" in df.columns and "Settlement_Date" in df.columns:
        both_valid = df["Submission_Date"].notna() & df["Settlement_Date"].notna()
        bad_order = both_valid & (df["Settlement_Date"] < df["Submission_Date"])
        report["settlement_before_submission"] = bad_order.sum()
        df.loc[bad_order, "Settlement_Date"] = pd.NaT
        report["excluded_from_dates"] = report["settlement_before_submission"]

    return df, report


def validate_financials(df):
    """Check numeric types and non-negative constraints. Return report dict."""
    report = {"non_numeric": [], "negative_counts": {}, "over_claim": {}}
    amount_cols = ["Claim_Amount", "Approved_Amount", "Rejected_Amount", "Pending_Amount"]
    for col in amount_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        neg = (df[col] < 0).sum()
        if neg > 0:
            report["negative_counts"][col] = int(neg)
    for col in ["Approved_Amount", "Rejected_Amount", "Pending_Amount"]:
        if col in df.columns and "Claim_Amount" in df.columns:
            over = (df[col] > df["Claim_Amount"]).sum()
            if over > 0:
                report["over_claim"][col] = int(over)
    return df, report


def validate_status(df):
    """Return unexpected status values."""
    if "Case_Status" not in df.columns:
        return []
    unexpected = df[~df["Case_Status"].isin(VALID_STATUSES)]["Case_Status"].unique().tolist()
    return unexpected


def validate_duplicates(df):
    """Return duplicate count on Case_ID."""
    if "Case_ID" not in df.columns:
        return 0
    return int(df["Case_ID"].duplicated().sum())


# ---------------------------------------------------------------------------
# Derived variables
# ---------------------------------------------------------------------------

def create_derived_variables(df, delay_threshold=DELAY_THRESHOLD_DAYS):
    """Add Processing_Time_Days, Delayed columns."""
    analysis_date = pd.Timestamp(datetime.date.today())

    # Processing time (only for resolved cases with valid dates)
    if "Submission_Date" in df.columns and "Settlement_Date" in df.columns:
        both_valid = df["Submission_Date"].notna() & df["Settlement_Date"].notna()
        df["Processing_Time_Days"] = np.nan
        df.loc[both_valid, "Processing_Time_Days"] = (
            df.loc[both_valid, "Settlement_Date"] - df.loc[both_valid, "Submission_Date"]
        ).dt.days

    # Delayed flag
    df["Delayed"] = 0

    if "Processing_Time_Days" in df.columns:
        resolved = df["Processing_Time_Days"].notna()
        df.loc[resolved & (df["Processing_Time_Days"] > delay_threshold), "Delayed"] = 1

    # Unresolved: check elapsed days from submission
    if "Submission_Date" in df.columns:
        unresolved = df.get("Processing_Time_Days", pd.Series(dtype=float)).isna()
        if "Submission_Date" in df.columns:
            elapsed = (analysis_date - df["Submission_Date"]).dt.days
            df.loc[unresolved & (elapsed > delay_threshold), "Delayed"] = 1

    # Submission month and weekday (for features)
    if "Submission_Date" in df.columns:
        df["Submission_Month"] = df["Submission_Date"].dt.month.fillna(0).astype(int)
        df["Submission_Weekday"] = df["Submission_Date"].dt.dayofweek.fillna(0).astype(int)

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


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_classification_pipeline(cat_features, num_features, model):
    """Build a full scikit-learn Pipeline for classification."""
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
    """
    Train classification models predicting Delayed.
    Returns best pipeline and metadata dict.
    """
    print("\n" + "="*60)
    print("HEALTHCARE TPA MODEL TRAINING")
    print("="*60)

    # --- Feature / target setup ---
    target_col = "Delayed"
    cat_features = [c for c in CLASSIFICATION_FEATURES_CAT if c in df.columns]
    num_features = [c for c in CLASSIFICATION_FEATURES_NUM if c in df.columns]
    all_features = cat_features + num_features

    print(f"\n[INFO] Target            : {target_col} (1 = delayed > {delay_threshold} days)")
    print(f"[INFO] Problem type      : Binary Classification")
    print(f"[INFO] Categorical feats : {cat_features}")
    print(f"[INFO] Numeric feats     : {num_features}")
    print(f"[INFO] Leakage-excluded  : Settlement_Date, Processing_Time_Days, "
          "Approved_Amount, Rejected_Amount, Pending_Amount")

    # --- Target availability ---
    if target_col not in df.columns:
        print("[ERROR] Target column 'Delayed' missing. Training aborted.")
        return None, None

    valid_df = df.dropna(subset=[target_col]).copy()
    print(f"[INFO] Records after dropping missing target : {len(valid_df)}")

    if len(valid_df) < MIN_RECORDS_FOR_TRAINING:
        print(f"[ERROR] Too few valid records ({len(valid_df)} < {MIN_RECORDS_FOR_TRAINING}). "
              "Training aborted.")
        return None, None

    classes = valid_df[target_col].unique()
    if len(classes) < 2:
        print(f"[ERROR] Target has only one class: {classes}. Training aborted.")
        return None, None

    X = valid_df[all_features]
    y = valid_df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # --- Candidate models ---
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
    }

    results = {}
    pipelines = {}

    print("\n[INFO] Evaluating candidate models...")
    print("-" * 60)

    for name, estimator in candidates.items():
        pipe = build_classification_pipeline(cat_features, num_features, estimator)
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
            cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="f1")
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
            print(f"    CV F1     : {cv_mean:.4f} ± {cv_std:.4f}")
        print()

    # --- Model selection (highest F1, recall as tiebreaker) ---
    best_name = max(results, key=lambda n: (results[n]["f1"], results[n]["recall"]))
    best_pipeline = pipelines[best_name]
    best_metrics = results[best_name]

    print(f"[SELECTED] Best model : {best_name} (F1={best_metrics['f1']:.4f}, "
          f"Recall={best_metrics['recall']:.4f})")

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
            "Settlement_Date",
            "Processing_Time_Days",
            "Delayed",
            "Approved_Amount",
            "Rejected_Amount",
            "Pending_Amount",
        ],
        "metrics": best_metrics,
        "all_results": results,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "trained_at": str(datetime.datetime.now()),
        "selection_rule": (
            "Best F1 score on hold-out test set; "
            "Recall used as secondary tiebreaker."
        ),
    }

    return best_pipeline, metadata


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_model(pipeline, metadata):
    """Save pipeline + metadata to models/model.pkl using joblib."""
    os.makedirs("models", exist_ok=True)
    payload = {"pipeline": pipeline, "metadata": metadata}
    joblib.dump(payload, MODEL_OUTPUT_PATH)
    print(f"\n[SAVED] Pipeline saved to: {MODEL_OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("STEP 1 — Locating TPA DATA.xlsx")
    print("="*60)
    path = locate_workbook()
    if path is None:
        print(
            "\n[ERROR] TPA DATA.xlsx was not found.\n"
            "Please place the actual Excel workbook in the project root "
            "directory or in the data folder.\n"
            "Training aborted."
        )
        sys.exit(1)
    print(f"[FOUND] Workbook at: {path}")

    # --- Load ---
    print("\nSTEP 2 — Loading workbook sheets")
    sheets = load_workbook(path)

    # --- Select worksheet ---
    best_sheet = select_worksheet(sheets)
    print(f"\n[SELECTED] Worksheet for analysis: '{best_sheet}'")
    df = sheets[best_sheet].copy()

    # --- Column normalization ---
    print("\nSTEP 3 — Normalizing column names")
    raw_cols = list(df.columns)
    normalized_map, final_map = build_column_map(raw_cols)
    df.rename(columns=final_map, inplace=True)
    print(f"  Original  : {raw_cols}")
    print(f"  Mapped to : {list(df.columns)}")

    # --- Column validation ---
    print("\nSTEP 4 — Column validation")
    missing_cols = validate_columns(df)
    if missing_cols:
        print(f"  [WARNING] Missing expected columns: {missing_cols}")
    else:
        print("  All expected columns present.")

    # --- Date validation ---
    print("\nSTEP 5 — Date validation")
    df, date_report = validate_dates(df)
    print(f"  Invalid Submission_Date  : {date_report['invalid_submission']}")
    print(f"  Invalid Settlement_Date  : {date_report['invalid_settlement']}")
    print(f"  Settlement < Submission  : {date_report['settlement_before_submission']} "
          "(excluded from date calculations)")

    # --- Financial validation ---
    print("\nSTEP 6 — Financial validation")
    df, fin_report = validate_financials(df)
    if fin_report["negative_counts"]:
        print(f"  Negative amounts: {fin_report['negative_counts']}")
    else:
        print("  No negative amounts detected.")
    if fin_report["over_claim"]:
        print(f"  Amounts exceeding Claim_Amount: {fin_report['over_claim']}")
    else:
        print("  No amounts exceed Claim_Amount.")

    # --- Status validation ---
    print("\nSTEP 7 — Status validation")
    unexpected = validate_status(df)
    if unexpected:
        print(f"  Unexpected Case_Status values: {unexpected}")
    else:
        print("  All Case_Status values are valid.")

    # --- Duplicate validation ---
    print("\nSTEP 8 — Duplicate validation")
    dup_count = validate_duplicates(df)
    if dup_count > 0:
        print(f"  [WARNING] {dup_count} duplicate Case_ID(s) found. "
              "Duplicates are NOT silently removed.")
    else:
        print("  No duplicate Case_IDs.")

    # --- Missing values ---
    print("\nSTEP 9 — Missing value summary")
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    if mv.empty:
        print("  No missing values.")
    else:
        print(mv.to_string())

    # --- Derived variables ---
    print("\nSTEP 10 — Creating derived variables")
    df = create_derived_variables(df, delay_threshold=DELAY_THRESHOLD_DAYS)
    if "Processing_Time_Days" in df.columns:
        resolved = df["Processing_Time_Days"].notna().sum()
        print(f"  Processing_Time_Days computed for {resolved} resolved records.")
    if "Delayed" in df.columns:
        delayed = df["Delayed"].sum()
        print(f"  Delayed (>{DELAY_THRESHOLD_DAYS} days): {delayed} records "
              f"({100*delayed/len(df):.1f}%)")

    # --- Train models ---
    best_pipeline, metadata = train_and_evaluate(df, delay_threshold=DELAY_THRESHOLD_DAYS)

    if best_pipeline is None:
        print("\n[ERROR] Model training failed. No model saved.")
        sys.exit(1)

    # --- Save ---
    save_model(best_pipeline, metadata)

    # --- Summary ---
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"  Workbook       : {path}")
    print(f"  Worksheet      : {best_sheet}")
    print(f"  Total records  : {len(df)}")
    print(f"  Model type     : Classification (Binary)")
    print(f"  Target         : Delayed (>{DELAY_THRESHOLD_DAYS} days)")
    print(f"  Best model     : {metadata['model_name']}")
    print(f"  F1 Score       : {metadata['metrics']['f1']}")
    print(f"  Recall         : {metadata['metrics']['recall']}")
    print(f"  Accuracy       : {metadata['metrics']['accuracy']}")
    print(f"  Saved to       : {MODEL_OUTPUT_PATH}")
    print("="*60)
    print("\n[DONE] Run 'streamlit run app.py' to launch the dashboard.")


if __name__ == "__main__":
    main()
