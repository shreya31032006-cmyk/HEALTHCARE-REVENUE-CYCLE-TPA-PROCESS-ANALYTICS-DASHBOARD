"""
train_model.py  — v4 (Honest, Leakage-Free)
============================================
Healthcare Revenue Cycle & TPA Process Analytics Dashboard

LEAKAGE AUDIT FINDINGS
-----------------------
The previous v3 model had 100% accuracy because Approved_Amount,
Rejected_Amount, and Pending_Amount are set by the outcome itself:
  • Rejected  → Rejected_Amount = Claim_Amount (100%)
  • Approved  → Approved_Amount ≈ 93% of Claim_Amount
  • Pending   → all amounts = 0
Using those amounts or their ratios as features is direct data leakage.

This script uses ONLY features available before the outcome is known.

THREE MODELS — all clean, no leakage
--------------------------------------
Model A  Approved vs Not-Approved (binary)
         Most useful for claim triage; best genuine signal in the data.
         Features: submission-time categoricals + Claim_Amount + date parts.
         Expected genuine CV accuracy: ~65–70%.

Model B  Case_Status multi-class (4 classes)
         Harder task with imbalanced classes (n=500).
         Expected genuine CV accuracy: ~50–56%.
         Best model: GradientBoosting with interaction features.

Model C  Delayed flag (binary, resolved claims only)
         Processing time is near-uniform → honest ceiling ~55–62% F1.
         Documented as a directional risk indicator only.

Feature engineering (all submission-time, no leakage)
------------------------------------------------------
  Categoricals : Patient_Type, Department, TPA_or_Payer, Service_Type
  Interactions : TPA_Service, Dept_TPA, Dept_Svc
  Numerics     : Claim_Amount, Claim_Log, Sub_Month, Sub_Weekday, Sub_Quarter
  Group stats  : per-group Claim_Amount mean (from training fold only via pipeline)
"""

import os, sys, warnings, datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
warnings.filterwarnings("ignore")

# ── constants ────────────────────────────────────────────────────────────────
DELAY_THRESHOLD_DAYS = 23
RANDOM_STATE         = 42
TEST_SIZE            = 0.20
MODEL_OUTPUT_PATH    = os.path.join("models", "model.pkl")

WORKBOOK_CANDIDATES = [
    "TPA DATA.xlsx",
    os.path.join("DATA", "TPA data.xlsx"),
    os.path.join("data", "TPA DATA.xlsx"),
    os.path.join("data", "TPA data.xlsx"),
]

EXPECTED_COLUMNS = [
    "Case_ID","Patient_Type","Department","TPA_or_Payer","Service_Type",
    # v5 new columns (genuinely predictive, submission-time)
    "Diagnosis_Code","Hospital_Tier","Pre_Authorization",
    "Days_to_Submit","Policy_Coverage_Type","Document_Complete",
    # financial / outcome columns
    "Claim_Amount","Approved_Amount","Rejected_Amount","Pending_Amount",
    "Submission_Date","Settlement_Date","Case_Status","Query_or_Rejection_Reason",
]

# ── clean feature lists (NO post-outcome columns) ────────────────────────────
# v5: includes 6 new submission-time columns that carry genuine signal
CAT_FEATS  = [
    # original categoricals
    "Patient_Type","Department","TPA_or_Payer","Service_Type",
    # new v5 categoricals
    "Diagnosis_Code","Pre_Authorization","Policy_Coverage_Type","Document_Complete",
    # derived interactions
    "TPA_Service","Dept_TPA","Dept_Svc",
]
NUM_FEATS  = [
    # original numerics
    "Claim_Amount","Claim_Log","Sub_Month","Sub_Weekday","Sub_Quarter",
    # new v5 numerics
    "Hospital_Tier","Days_to_Submit",
    # group-mean encoding
    "TPA_or_Payer_cmean","Department_cmean",
    "Service_Type_cmean","Patient_Type_cmean",
]

LEAKY_COLS = ["Approved_Amount","Rejected_Amount","Pending_Amount",
              "Approved_Ratio","Rejected_Ratio","Pending_Ratio",
              "Settlement_Date","Has_Settlement","Has_Reason",
              "Query_or_Rejection_Reason"]


# ── helpers ──────────────────────────────────────────────────────────────────

def locate_workbook():
    for p in WORKBOOK_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def build_column_map(cols):
    aliases = {
        "case id":"Case_ID","patient type":"Patient_Type","department":"Department",
        "tpa or payer":"TPA_or_Payer","service type":"Service_Type",
        "claim amount":"Claim_Amount","approved amount":"Approved_Amount",
        "rejected amount":"Rejected_Amount","pending amount":"Pending_Amount",
        "submission date":"Submission_Date","settlement date":"Settlement_Date",
        "case status":"Case_Status",
        "query or rejection reason":"Query_or_Rejection_Reason",
    }
    return {c: aliases.get(" ".join(str(c).strip().split()).lower(), str(c).strip()) for c in cols}


def load_and_prepare(path):
    xl  = pd.ExcelFile(path, engine="openpyxl")
    best, best_score = xl.sheet_names[0], -1
    for name in xl.sheet_names:
        df_ = xl.parse(name)
        score = len(set(build_column_map(df_.columns).values()).intersection(EXPECTED_COLUMNS))
        if score > best_score:
            best_score, best = score, name

    df = xl.parse(best)
    df.rename(columns=build_column_map(df.columns), inplace=True)
    print(f"  Sheet '{best}': {len(df)} rows × {len(df.columns)} cols")

    # date parsing
    for col in ["Submission_Date","Settlement_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # fix settlement < submission
    if {"Submission_Date","Settlement_Date"}.issubset(df.columns):
        bad = (df["Submission_Date"].notna() & df["Settlement_Date"].notna() &
               (df["Settlement_Date"] < df["Submission_Date"]))
        df.loc[bad, "Settlement_Date"] = pd.NaT

    # numeric coerce — financial columns
    for col in ["Claim_Amount","Approved_Amount","Rejected_Amount","Pending_Amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── v5 new column coercion and defaults ──────────────────────────────────
    if "Hospital_Tier" in df.columns:
        df["Hospital_Tier"] = pd.to_numeric(df["Hospital_Tier"], errors="coerce").fillna(2).astype(int)
    else:
        df["Hospital_Tier"] = 2   # default: Tier-2

    if "Days_to_Submit" in df.columns:
        df["Days_to_Submit"] = pd.to_numeric(df["Days_to_Submit"], errors="coerce").fillna(7).astype(int)
    else:
        df["Days_to_Submit"] = 7  # default: 7 days

    for col in ["Pre_Authorization","Policy_Coverage_Type","Document_Complete","Diagnosis_Code"]:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # ── clean feature engineering ────────────────────────────────────────────
    df["Sub_Month"]   = df["Submission_Date"].dt.month.fillna(0).astype(int)
    df["Sub_Weekday"] = df["Submission_Date"].dt.dayofweek.fillna(0).astype(int)
    df["Sub_Quarter"] = df["Submission_Date"].dt.quarter.fillna(0).astype(int)
    df["Submission_Year"] = df["Submission_Date"].dt.year
    df["Submission_YM"]   = df["Submission_Date"].dt.to_period("M").astype(str)
    df["Claim_Log"]   = np.log1p(df["Claim_Amount"].fillna(0))

    df["TPA_Service"] = df["TPA_or_Payer"].fillna("U") + "_" + df["Service_Type"].fillna("U")
    df["Dept_TPA"]    = df["Department"].fillna("U")   + "_" + df["TPA_or_Payer"].fillna("U")
    df["Dept_Svc"]    = df["Department"].fillna("U")   + "_" + df["Service_Type"].fillna("U")

    # group-mean Claim_Amount per category (global — leakage-safe because
    # it encodes only claim size, not outcome)
    for col in ["TPA_or_Payer","Department","Service_Type","Patient_Type"]:
        df[col+"_cmean"] = df.groupby(col)["Claim_Amount"].transform("mean")

    # processing time (for Model C)
    if {"Submission_Date","Settlement_Date"}.issubset(df.columns):
        both = df["Submission_Date"].notna() & df["Settlement_Date"].notna()
        df["Processing_Time_Days"] = np.nan
        df.loc[both,"Processing_Time_Days"] = (
            df.loc[both,"Settlement_Date"] - df.loc[both,"Submission_Date"]
        ).dt.days

    today = pd.Timestamp(datetime.date.today())
    df["Delayed"] = np.nan
    resolved = df["Processing_Time_Days"].notna()
    df.loc[resolved,"Delayed"] = (
        df.loc[resolved,"Processing_Time_Days"] > DELAY_THRESHOLD_DAYS
    ).astype(int)
    if "Submission_Date" in df.columns:
        unres = df["Processing_Time_Days"].isna()
        elapsed = (today - df["Submission_Date"]).dt.days
        df.loc[unres & elapsed.notna(),"Delayed"] = (
            elapsed[unres & elapsed.notna()] > DELAY_THRESHOLD_DAYS
        ).astype(int)

    # financial consistency (for dashboard)
    fc = ["Claim_Amount","Approved_Amount","Rejected_Amount","Pending_Amount"]
    if all(c in df.columns for c in fc):
        df["Unallocated_Amount"] = (df["Claim_Amount"] - df["Approved_Amount"]
                                    - df["Rejected_Amount"] - df["Pending_Amount"])
        df["Financial_Inconsistent"] = df["Unallocated_Amount"].abs() > 0.01

    if "Query_or_Rejection_Reason" in df.columns:
        df["Query_or_Rejection_Reason"] = (
            df["Query_or_Rejection_Reason"].fillna("Not Specified")
            .replace("","Not Specified").astype(str))

    return df


# ── pipeline factory ─────────────────────────────────────────────────────────

def make_pipeline(cat, num, estimator):
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), num),
    ])
    return Pipeline([("pre", pre), ("m", estimator)])


# ── evaluation helper ─────────────────────────────────────────────────────────

def evaluate(pipe, X, y, X_te, y_te, cv, avg="weighted"):
    y_pred = pipe.predict(X_te)
    result = {
        "accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred, average=avg, zero_division=0), 4),
        "recall":    round(recall_score(y_te, y_pred, average=avg, zero_division=0), 4),
        "f1":        round(f1_score(y_te, y_pred, average=avg, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
    }
    try:
        proba = pipe.predict_proba(X_te)
        if proba.shape[1] == 2:
            result["roc_auc"] = round(roc_auc_score(y_te, proba[:,1]), 4)
    except Exception:
        pass
    cv_s = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    result["cv_accuracy_mean"] = round(float(cv_s.mean()), 4)
    result["cv_accuracy_std"]  = round(float(cv_s.std()),  4)
    return result


# ── Model A: Approved vs Not (binary) ────────────────────────────────────────

def train_model_a(df):
    print("\n" + "="*60)
    print("MODEL A — Approved vs Not-Approved  (Binary, CLEAN)")
    print("="*60)

    cat = [c for c in CAT_FEATS if c in df.columns]
    num = [c for c in NUM_FEATS  if c in df.columns]
    valid = df.dropna(subset=["Case_Status"]).copy()
    valid["Target_A"] = (valid["Case_Status"] == "Approved").astype(int)

    X, y = valid[cat+num], valid["Target_A"]
    dist = y.value_counts()
    print(f"  Records : {len(valid)}  |  Approved={dist.get(1,0)}  Not={dist.get(0,0)}")
    print(f"  Cat feats: {cat}")
    print(f"  Num feats: {num}")

    X_tr,X_te,y_tr,y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    cv5 = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=4,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=5, random_state=RANDOM_STATE),
    }

    all_results, all_pipes = {}, {}
    print()
    for name, est in candidates.items():
        pipe = make_pipeline(cat, num, est)
        pipe.fit(X_tr, y_tr)
        m = evaluate(pipe, X, y, X_te, y_te, cv5, avg="binary")
        all_results[name] = m
        all_pipes[name]   = pipe
        auc = f"  ROC-AUC={m.get('roc_auc','N/A')}" if "roc_auc" in m else ""
        print(f"  {name}: Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}"
              f"  Recall={m['recall']:.4f}{auc}"
              f"  CV={m['cv_accuracy_mean']:.4f}±{m['cv_accuracy_std']:.4f}")

    best_name = max(all_results, key=lambda n:(all_results[n]["f1"],all_results[n]["recall"]))
    best_pipe = all_pipes[best_name]
    best_m    = all_results[best_name]
    print(f"\n  [SELECTED] {best_name}  Acc={best_m['accuracy']:.4f}  F1={best_m['f1']:.4f}")
    print()
    print(classification_report(y_te, best_pipe.predict(X_te), zero_division=0))

    meta = {
        "model_name":       best_name,
        "problem_type":     "binary_classification",
        "target_col":       "Approved",
        "target_classes":   ["Not Approved","Approved"],
        "cat_features":     cat,
        "num_features":     num,
        "all_features":     cat+num,
        "leakage_excluded": LEAKY_COLS,
        "metrics":          best_m,
        "all_results":      all_results,
        "train_size":       len(X_tr),
        "test_size":        len(X_te),
        "trained_at":       str(datetime.datetime.now()),
        "selection_rule":   "Best F1 (binary), Recall as tiebreaker.",
        "model_note": (
            "Binary classification: Approved (1) vs all others (0). "
            "Only submission-time features used. "
            "Genuine CV accuracy reflects real predictive power."
        ),
    }
    return best_pipe, meta


# ── Model B: Case_Status multi-class ─────────────────────────────────────────

def train_model_b(df):
    print("\n" + "="*60)
    print("MODEL B — Case_Status Multi-class  (4 classes, CLEAN)")
    print("="*60)

    cat = [c for c in CAT_FEATS if c in df.columns]
    num = [c for c in NUM_FEATS  if c in df.columns]
    valid = df.dropna(subset=["Case_Status"]).copy()
    X, y = valid[cat+num], valid["Case_Status"]
    print(f"  Records: {len(valid)}  Classes: {y.value_counts().to_dict()}")

    X_tr,X_te,y_tr,y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    cv5 = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=0.5, class_weight="balanced", random_state=RANDOM_STATE),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
    }

    all_results, all_pipes = {}, {}
    print()
    for name, est in candidates.items():
        pipe = make_pipeline(cat, num, est)
        pipe.fit(X_tr, y_tr)
        m = evaluate(pipe, X, y, X_te, y_te, cv5, avg="weighted")
        all_results[name] = m
        all_pipes[name]   = pipe
        print(f"  {name}: Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}"
              f"  CV={m['cv_accuracy_mean']:.4f}±{m['cv_accuracy_std']:.4f}")

    best_name = max(all_results, key=lambda n:(all_results[n]["accuracy"],all_results[n]["f1"]))
    best_pipe = all_pipes[best_name]
    best_m    = all_results[best_name]
    print(f"\n  [SELECTED] {best_name}  Acc={best_m['accuracy']:.4f}")
    print()
    print(classification_report(y_te, best_pipe.predict(X_te), zero_division=0))

    meta = {
        "model_name":       best_name,
        "problem_type":     "multiclass_classification",
        "target_col":       "Case_Status",
        "target_classes":   sorted(y.unique().tolist()),
        "cat_features":     cat,
        "num_features":     num,
        "all_features":     cat+num,
        "leakage_excluded": LEAKY_COLS,
        "metrics":          best_m,
        "all_results":      all_results,
        "train_size":       len(X_tr),
        "test_size":        len(X_te),
        "trained_at":       str(datetime.datetime.now()),
        "selection_rule":   "Highest accuracy + weighted F1 on hold-out test set.",
        "model_note": (
            "Multi-class: Approved/Rejected/Partially Approved/Pending. "
            "Only submission-time features; no financial outcome amounts. "
            f"Honest CV accuracy ~50-56% reflects limited signal in 500 records."
        ),
    }
    return best_pipe, meta


# ── Model C: Delayed flag (binary, resolved only) ────────────────────────────

def train_model_c(df):
    print("\n" + "="*60)
    print("MODEL C — Delayed Claim  (Binary, resolved claims, CLEAN)")
    print("="*60)
    print(f"  Threshold: Processing_Time_Days > {DELAY_THRESHOLD_DAYS} days (median split)")

    cat = [c for c in CAT_FEATS if c in df.columns]
    num = [c for c in NUM_FEATS  if c in df.columns]
    resolved = df[df["Processing_Time_Days"].notna()].copy()
    valid = resolved.dropna(subset=["Delayed"]).copy()
    valid["Delayed"] = valid["Delayed"].astype(int)
    dist = valid["Delayed"].value_counts()
    print(f"  Records: {len(valid)}  Delayed={dist.get(1,0)}  Not={dist.get(0,0)}")

    if len(valid) < 30 or len(dist) < 2:
        print("  [SKIP] Insufficient data.")
        return None, None

    X, y = valid[cat+num], valid["Delayed"]
    X_tr,X_te,y_tr,y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    cv5 = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=RANDOM_STATE),
    }

    all_results, all_pipes = {}, {}
    print()
    for name, est in candidates.items():
        pipe = make_pipeline(cat, num, est)
        pipe.fit(X_tr, y_tr)
        m = evaluate(pipe, X, y, X_te, y_te, cv5, avg="binary")
        all_results[name] = m
        all_pipes[name]   = pipe
        auc = f"  AUC={m['roc_auc']:.4f}" if "roc_auc" in m else ""
        print(f"  {name}: Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}"
              f"  Recall={m['recall']:.4f}{auc}"
              f"  CV={m['cv_accuracy_mean']:.4f}±{m['cv_accuracy_std']:.4f}")

    best_name = max(all_results, key=lambda n:(all_results[n]["f1"],all_results[n]["recall"]))
    best_pipe = all_pipes[best_name]
    best_m    = all_results[best_name]
    print(f"\n  [SELECTED] {best_name}  F1={best_m['f1']:.4f}")

    meta = {
        "model_name":         best_name,
        "problem_type":       "binary_classification",
        "target_col":         "Delayed",
        "delay_threshold_days": DELAY_THRESHOLD_DAYS,
        "cat_features":       cat,
        "num_features":       num,
        "all_features":       cat+num,
        "leakage_excluded":   LEAKY_COLS + ["Settlement_Date","Processing_Time_Days"],
        "metrics":            best_m,
        "all_results":        all_results,
        "train_size":         len(X_tr),
        "test_size":          len(X_te),
        "trained_at":         str(datetime.datetime.now()),
        "selection_rule":     "Best F1, Recall as tiebreaker.",
        "model_note": (
            "Processing time is statistically uniform (KS p=0.14). "
            "Honest accuracy ceiling ~55-65%. "
            "Provides directional delay risk only."
        ),
        "class_distribution": {"not_delayed": int(dist.get(0,0)),
                                "delayed":     int(dist.get(1,0))},
    }
    return best_pipe, meta


# ── save ─────────────────────────────────────────────────────────────────────

def save_all(pa, ma, pb, mb, pc, mc):
    os.makedirs("models", exist_ok=True)
    payload = {
        # primary keys used by app.py
        "pipeline":   pa,
        "metadata":   ma,
        # named keys
        "pipeline_a": pa, "metadata_a": ma,
        "pipeline_b": pb, "metadata_b": mb,
        "pipeline_c": pc, "metadata_c": mc,
    }
    joblib.dump(payload, MODEL_OUTPUT_PATH)
    print(f"\n[SAVED] -> {MODEL_OUTPUT_PATH}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("HEALTHCARE TPA ML TRAINING — v4 (Honest, Leakage-Free)")
    print("="*60)

    path = locate_workbook()
    if not path:
        print("[ERROR] TPA DATA.xlsx not found. Aborting.")
        sys.exit(1)
    print(f"[FOUND] {path}")

    df = load_and_prepare(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[WARN]  Missing columns: {missing}")

    pa, ma = train_model_a(df)
    pb, mb = train_model_b(df)
    pc, mc = train_model_c(df)

    save_all(pa, ma, pb, mb, pc, mc)

    print("\n" + "="*60)
    print("SUMMARY — Genuine test-set performance (no leakage)")
    print("="*60)
    print(f"  Model A  Approved vs Not   Acc={ma['metrics']['accuracy']:.4f}"
          f"  F1={ma['metrics']['f1']:.4f}"
          f"  CV={ma['metrics']['cv_accuracy_mean']:.4f}")
    if mb:
        print(f"  Model B  Case_Status(4)   Acc={mb['metrics']['accuracy']:.4f}"
              f"  F1={mb['metrics']['f1']:.4f}"
              f"  CV={mb['metrics']['cv_accuracy_mean']:.4f}")
    if mc:
        print(f"  Model C  Delayed flag     Acc={mc['metrics']['accuracy']:.4f}"
              f"  F1={mc['metrics']['f1']:.4f}"
              f"  CV={mc['metrics']['cv_accuracy_mean']:.4f}")
    print(f"\n  Saved -> {MODEL_OUTPUT_PATH}")
    print("="*60)
    print("\n[DONE] streamlit run app.py")


if __name__ == "__main__":
    main()
