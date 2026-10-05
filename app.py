"""
app.py
======
Healthcare Revenue Cycle and TPA Process Analytics Dashboard
with Predictive Analysis

Sections
--------
1. Overview Dashboard
2. Claim Records
3. TPA Analysis
4. Department Analysis
5. Process Performance
6. Query and Rejection Analysis
7. Predictive Analytics
8. Data Quality and Model Validation

Run:
    streamlit run app.py
"""

import os
import datetime
import warnings

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DELAY_THRESHOLD_DAYS = 23   # median split — balanced 50/50 classes
MODEL_PATH = os.path.join("models", "model.pkl")

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

VALID_STATUSES = {"Approved", "Pending", "Rejected", "Queried", "In Process", "Settled"}

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare RCM & TPA Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helper: locate workbook
# ---------------------------------------------------------------------------

def locate_workbook():
    for p in WORKBOOK_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Column normalization (mirrors train_model.py)
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
# Data loading & preprocessing (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(delay_threshold=DELAY_THRESHOLD_DAYS):
    """
    Locate and load the workbook.
    Returns (df, audit_info) or (None, audit_info) on failure.
    """
    audit = {
        "workbook_path": None,
        "sheet_names": [],
        "selected_sheet": None,
        "total_rows": 0,
        "total_cols": 0,
        "original_columns": [],
        "mapped_columns": [],
        "missing_required": [],
        "missing_values": {},
        "invalid_dates": 0,
        "negative_amounts": {},
        "invalid_statuses": [],
        "duplicate_case_ids": 0,
        "financial_inconsistencies": 0,
        "settlement_before_submission": 0,
    }

    wb_path = locate_workbook()
    if wb_path is None:
        return None, audit

    audit["workbook_path"] = wb_path

    xl = pd.ExcelFile(wb_path, engine="openpyxl")
    audit["sheet_names"] = xl.sheet_names

    # Select best sheet
    best_sheet, best_score = xl.sheet_names[0], -1
    for sheet in xl.sheet_names:
        tmp = xl.parse(sheet)
        _, fmap = _build_final_map(tmp.columns.tolist())
        score = len(set(fmap.values()).intersection(set(EXPECTED_COLUMNS)))
        if score > best_score:
            best_score = score
            best_sheet = sheet

    audit["selected_sheet"] = best_sheet
    df = xl.parse(best_sheet)

    audit["original_columns"] = list(df.columns)
    final_map = build_column_map(list(df.columns))
    df.rename(columns=final_map, inplace=True)
    audit["mapped_columns"] = list(df.columns)

    audit["total_rows"] = len(df)
    audit["total_cols"] = len(df.columns)

    # Column check
    audit["missing_required"] = [c for c in EXPECTED_COLUMNS if c not in df.columns]

    # Dates
    for col in ["Submission_Date", "Settlement_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    inv_dates = 0
    if "Submission_Date" in df.columns:
        inv_dates += int(df["Submission_Date"].isna().sum())
    if "Settlement_Date" in df.columns:
        inv_dates += int(df["Settlement_Date"].isna().sum())
    audit["invalid_dates"] = inv_dates

    # Settlement before submission
    if "Submission_Date" in df.columns and "Settlement_Date" in df.columns:
        both = df["Submission_Date"].notna() & df["Settlement_Date"].notna()
        bad = both & (df["Settlement_Date"] < df["Submission_Date"])
        audit["settlement_before_submission"] = int(bad.sum())
        df.loc[bad, "Settlement_Date"] = pd.NaT

    # Financials
    amount_cols = ["Claim_Amount", "Approved_Amount", "Rejected_Amount", "Pending_Amount"]
    for col in amount_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            neg = int((df[col] < 0).sum())
            if neg > 0:
                audit["negative_amounts"][col] = neg

    # Status
    if "Case_Status" in df.columns:
        audit["invalid_statuses"] = (
            df[~df["Case_Status"].isin(VALID_STATUSES)]["Case_Status"].unique().tolist()
        )

    # Duplicates
    if "Case_ID" in df.columns:
        audit["duplicate_case_ids"] = int(df["Case_ID"].duplicated().sum())

    # Missing values
    audit["missing_values"] = {
        c: int(v) for c, v in df.isnull().sum().items() if v > 0
    }

    # Derived variables
    analysis_date = pd.Timestamp(datetime.date.today())

    if "Submission_Date" in df.columns and "Settlement_Date" in df.columns:
        both_valid = df["Submission_Date"].notna() & df["Settlement_Date"].notna()
        df["Processing_Time_Days"] = np.nan
        df.loc[both_valid, "Processing_Time_Days"] = (
            df.loc[both_valid, "Settlement_Date"] - df.loc[both_valid, "Submission_Date"]
        ).dt.days

    df["Delayed"] = np.nan
    if "Processing_Time_Days" in df.columns:
        resolved = df["Processing_Time_Days"].notna()
        df.loc[resolved, "Delayed"] = (
            df.loc[resolved, "Processing_Time_Days"] > delay_threshold
        ).astype(int)
    if "Submission_Date" in df.columns:
        unresolved_mask = df["Processing_Time_Days"].isna()
        elapsed = (analysis_date - df["Submission_Date"]).dt.days
        df.loc[unresolved_mask & elapsed.notna(), "Delayed"] = (
            elapsed[unresolved_mask & elapsed.notna()] > delay_threshold
        ).astype(int)

    # Submission date features (clean — no leakage)
    if "Submission_Date" in df.columns:
        df["Sub_Month"]   = df["Submission_Date"].dt.month.fillna(0).astype(int)
        df["Sub_Weekday"] = df["Submission_Date"].dt.dayofweek.fillna(0).astype(int)
        df["Sub_Quarter"] = df["Submission_Date"].dt.quarter.fillna(0).astype(int)
        df["Submission_Month"]   = df["Sub_Month"]
        df["Submission_Weekday"] = df["Sub_Weekday"]
        df["Submission_Year"]    = df["Submission_Date"].dt.year
        df["Submission_YM"]      = df["Submission_Date"].dt.to_period("M").astype(str)

    # Numeric claim features (clean)
    if "Claim_Amount" in df.columns:
        df["Claim_Amount"] = pd.to_numeric(df["Claim_Amount"], errors="coerce").fillna(0)
        df["Claim_Log"]    = np.log1p(df["Claim_Amount"])

    # Interaction features (clean — submission-time categoricals only)
    if "TPA_or_Payer" in df.columns and "Service_Type" in df.columns:
        df["TPA_Service"] = df["TPA_or_Payer"].fillna("U") + "_" + df["Service_Type"].fillna("U")
    if "Department" in df.columns and "TPA_or_Payer" in df.columns:
        df["Dept_TPA"] = df["Department"].fillna("U") + "_" + df["TPA_or_Payer"].fillna("U")
    if "Department" in df.columns and "Service_Type" in df.columns:
        df["Dept_Svc"] = df["Department"].fillna("U") + "_" + df["Service_Type"].fillna("U")

    # Group-mean Claim_Amount per category (clean — encodes claim size, not outcome)
    for col in ["TPA_or_Payer", "Department", "Service_Type", "Patient_Type"]:
        if col in df.columns:
            df[col+"_cmean"] = df.groupby(col)["Claim_Amount"].transform("mean")

    # Financial amounts — coerce for dashboard KPIs only (NOT used as model features)
    for col in ["Approved_Amount", "Rejected_Amount", "Pending_Amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    fin_cols = ["Claim_Amount", "Approved_Amount", "Rejected_Amount", "Pending_Amount"]
    if all(c in df.columns for c in fin_cols):
        df["Unallocated_Amount"] = (
            df["Claim_Amount"]
            - df["Approved_Amount"]
            - df["Rejected_Amount"]
            - df["Pending_Amount"]
        )
        df["Financial_Inconsistent"] = df["Unallocated_Amount"].abs() > 0.01
        audit["financial_inconsistencies"] = int(df["Financial_Inconsistent"].sum())

    if "Query_or_Rejection_Reason" in df.columns:
        df["Query_or_Rejection_Reason"] = (
            df["Query_or_Rejection_Reason"]
            .fillna("Not Specified")
            .replace("", "Not Specified")
            .astype(str)
        )

    return df, audit


def _build_final_map(raw_columns):
    """Internal helper mirroring build_column_map for sheet selection."""
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
    fmap = {}
    for raw in raw_columns:
        normalized = " ".join(str(raw).strip().split())
        key = normalized.lower().replace("_", " ")
        if key in canonical_aliases:
            fmap[raw] = canonical_aliases[key]
        elif normalized.lower() in canonical_aliases:
            fmap[raw] = canonical_aliases[normalized.lower()]
        else:
            fmap[raw] = normalized
    return None, fmap


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.isfile(MODEL_PATH):
        return None, None, None, None, None, None, "Model file not found at models/model.pkl. Run train_model.py first."
    try:
        payload = joblib.load(MODEL_PATH)
        pipeline_a = payload.get("pipeline_a") or payload.get("pipeline")
        metadata_a = payload.get("metadata_a") or payload.get("metadata")
        pipeline_b = payload.get("pipeline_b")
        metadata_b = payload.get("metadata_b")
        pipeline_c = payload.get("pipeline_c")
        metadata_c = payload.get("metadata_c")
        if pipeline_a is None:
            return None, None, None, None, None, None, "Saved file does not contain a valid pipeline."
        return pipeline_a, metadata_a, pipeline_b, metadata_b, pipeline_c, metadata_c, None
    except Exception as e:
        return None, None, None, None, None, None, f"Failed to load model: {str(e)}"


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

def apply_sidebar_filters(df, delay_threshold_key="delay_threshold"):
    st.sidebar.header("Filters")

    # Delay threshold slider
    threshold = st.sidebar.slider(
        "Delay Threshold (days)", min_value=7, max_value=90,
        value=DELAY_THRESHOLD_DAYS, step=1,
        key=delay_threshold_key,
        help="Claims with processing time above this threshold are considered delayed.",
    )

    filters = {}

    if "Submission_Date" in df.columns:
        min_date = df["Submission_Date"].dropna().min().date()
        max_date = df["Submission_Date"].dropna().max().date()
        date_range = st.sidebar.date_input(
            "Submission Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            filters["date_start"] = pd.Timestamp(date_range[0])
            filters["date_end"] = pd.Timestamp(date_range[1])

    if "Department" in df.columns:
        departments = sorted(df["Department"].dropna().unique().tolist())
        selected = st.sidebar.multiselect("Department", options=departments, default=[])
        if selected:
            filters["Department"] = selected

    if "TPA_or_Payer" in df.columns:
        tpas = sorted(df["TPA_or_Payer"].dropna().unique().tolist())
        selected = st.sidebar.multiselect("TPA / Payer", options=tpas, default=[])
        if selected:
            filters["TPA_or_Payer"] = selected

    if "Patient_Type" in df.columns:
        pts = sorted(df["Patient_Type"].dropna().unique().tolist())
        selected = st.sidebar.multiselect("Patient Type", options=pts, default=[])
        if selected:
            filters["Patient_Type"] = selected

    if "Case_Status" in df.columns:
        statuses = sorted(df["Case_Status"].dropna().unique().tolist())
        selected = st.sidebar.multiselect("Case Status", options=statuses, default=[])
        if selected:
            filters["Case_Status"] = selected

    if "Service_Type" in df.columns:
        services = sorted(df["Service_Type"].dropna().unique().tolist())
        selected = st.sidebar.multiselect("Service Type", options=services, default=[])
        if selected:
            filters["Service_Type"] = selected

    delayed_filter = st.sidebar.selectbox(
        "Delayed Status", options=["All", "Delayed Only", "Not Delayed"],
    )

    return filters, threshold, delayed_filter


def filter_dataframe(df, filters, delayed_filter, threshold):
    fdf = df.copy()

    if "date_start" in filters and "Submission_Date" in fdf.columns:
        fdf = fdf[fdf["Submission_Date"] >= filters["date_start"]]
    if "date_end" in filters and "Submission_Date" in fdf.columns:
        fdf = fdf[fdf["Submission_Date"] <= filters["date_end"]]

    for col in ["Department", "TPA_or_Payer", "Patient_Type", "Case_Status", "Service_Type"]:
        if col in filters:
            fdf = fdf[fdf[col].isin(filters[col])]

    # Recompute Delayed based on current threshold
    if "Processing_Time_Days" in fdf.columns:
        analysis_date = pd.Timestamp(datetime.date.today())
        fdf["Delayed"] = np.nan
        resolved = fdf["Processing_Time_Days"].notna()
        fdf.loc[resolved, "Delayed"] = (
            fdf.loc[resolved, "Processing_Time_Days"] > threshold
        ).astype(int)
        if "Submission_Date" in fdf.columns:
            unresolved_mask = fdf["Processing_Time_Days"].isna()
            elapsed = (analysis_date - fdf["Submission_Date"]).dt.days
            fdf.loc[unresolved_mask & elapsed.notna(), "Delayed"] = (
                elapsed[unresolved_mask & elapsed.notna()] > threshold
            ).astype(int)

    if delayed_filter == "Delayed Only":
        fdf = fdf[fdf["Delayed"] == 1]
    elif delayed_filter == "Not Delayed":
        fdf = fdf[fdf["Delayed"] == 0]

    return fdf


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def add_predictions(df, pipeline_a, metadata_a, pipeline_c, metadata_c):
    """
    Add predictions to df:
      Model A  Approved vs Not-Approved (binary, clean features)
      Model C  Delayed flag (binary, resolved claims, clean features)
    """
    def risk_cat(p):
        return "High" if p >= 0.70 else ("Medium" if p >= 0.40 else "Low")

    # Model A: Approved vs Not
    if pipeline_a is not None and metadata_a is not None:
        feats = metadata_a.get("all_features", [])
        if all(f in df.columns for f in feats):
            try:
                proba = pipeline_a.predict_proba(df[feats])
                df["Predicted_Approved_Prob"] = proba[:, 1]
                df["Predicted_Approved"]      = pipeline_a.predict(df[feats])
                df["Predicted_Approval_Risk"]  = df["Predicted_Approved_Prob"].apply(risk_cat)
            except Exception:
                pass

    # Model C: Delayed flag
    if pipeline_c is not None and metadata_c is not None:
        feats = metadata_c.get("all_features", [])
        if all(f in df.columns for f in feats):
            try:
                proba = pipeline_c.predict_proba(df[feats])
                df["Predicted_Delayed_Prob"] = proba[:, 1]
                df["Predicted_Delayed"]      = pipeline_c.predict(df[feats])
                df["Predicted_Risk"]          = df["Predicted_Delayed_Prob"].apply(risk_cat)
            except Exception:
                pass

    return df


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def safe_bar_chart(data, x, y, title, xlabel="", ylabel="", color=None, labels=None):
    if data.empty:
        st.info(f"No data available for: {title}")
        return
    kwargs = dict(x=x, y=y, title=title, labels=labels or {})
    if color:
        kwargs["color"] = color
    fig = px.bar(data, **kwargs)
    fig.update_xaxes(title_text=xlabel or x)
    fig.update_yaxes(title_text=ylabel or y)
    st.plotly_chart(fig, use_container_width=True)


def safe_pie_chart(data, names, values, title):
    if data.empty or data[values].sum() == 0:
        st.info(f"No data available for: {title}")
        return
    fig = px.pie(data, names=names, values=values, title=title)
    st.plotly_chart(fig, use_container_width=True)


def safe_line_chart(data, x, y, title, xlabel="", ylabel="", color=None, labels=None):
    if data.empty:
        st.info(f"No data available for: {title}")
        return
    kwargs = dict(x=x, y=y, title=title, labels=labels or {})
    if color:
        kwargs["color"] = color
    fig = px.line(data, **kwargs)
    fig.update_xaxes(title_text=xlabel or x)
    fig.update_yaxes(title_text=ylabel or y)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def section_overview(df, pipeline, metadata, threshold):
    st.header("Overview Dashboard")

    total = len(df)
    total_claim = df["Claim_Amount"].sum() if "Claim_Amount" in df.columns else 0
    total_approved = df["Approved_Amount"].sum() if "Approved_Amount" in df.columns else 0
    total_pending = df["Pending_Amount"].sum() if "Pending_Amount" in df.columns else 0
    total_rejected = df["Rejected_Amount"].sum() if "Rejected_Amount" in df.columns else 0
    approval_rate = (total_approved / total_claim * 100) if total_claim > 0 else 0
    rejection_rate = (total_rejected / total_claim * 100) if total_claim > 0 else 0
    delayed_count = int(df["Delayed"].sum()) if "Delayed" in df.columns else 0

    avg_proc = None
    if "Processing_Time_Days" in df.columns:
        avg_proc = df["Processing_Time_Days"].mean()

    pred_delayed = None
    avg_pred_prob = None
    if "Predicted_Delayed" in df.columns:
        pred_delayed = int(df["Predicted_Delayed"].sum())
    if "Predicted_Delayed_Prob" in df.columns:
        avg_pred_prob = df["Predicted_Delayed_Prob"].mean()

    # KPI row 1
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cases", f"{total:,}")
    col2.metric("Total Claim Amount", f"₹{total_claim:,.0f}")
    col3.metric("Approved Amount", f"₹{total_approved:,.0f}")
    col4.metric("Pending Amount", f"₹{total_pending:,.0f}")

    # KPI row 2
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Rejected Amount", f"₹{total_rejected:,.0f}")
    col6.metric("Approval Rate", f"{approval_rate:.1f}%")
    col7.metric("Rejection Rate", f"{rejection_rate:.1f}%")
    col8.metric("Delayed Cases", f"{delayed_count:,}")

    # KPI row 3
    col9, col10, col11, col12 = st.columns(4)
    col9.metric(
        "Avg Processing Time",
        f"{avg_proc:.1f} days" if avg_proc is not None and not np.isnan(avg_proc) else "N/A"
    )
    col10.metric(
        "Predicted Delayed",
        f"{pred_delayed:,}" if pred_delayed is not None else "Model not loaded"
    )
    col11.metric(
        "Avg Delay Probability",
        f"{avg_pred_prob:.1%}" if avg_pred_prob is not None else "N/A"
    )
    if metadata:
        col12.metric("ML Model", metadata.get("model_name", "N/A"))

    st.divider()

    # Charts row 1
    col_a, col_b = st.columns(2)
    with col_a:
        if "Case_Status" in df.columns:
            status_counts = df["Case_Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            safe_pie_chart(status_counts, names="Status", values="Count",
                           title="Cases by Status")
    with col_b:
        if "Case_Status" in df.columns and "Claim_Amount" in df.columns:
            claim_by_status = (
                df.groupby("Case_Status")["Claim_Amount"].sum().reset_index()
            )
            claim_by_status.columns = ["Status", "Claim Amount"]
            safe_bar_chart(claim_by_status, x="Status", y="Claim Amount",
                           title="Claim Amount by Status",
                           ylabel="Total Claim Amount (₹)")

    # Monthly volume
    if "Submission_YM" in df.columns:
        monthly = (
            df.groupby("Submission_YM").size().reset_index(name="Cases")
            .sort_values("Submission_YM")
        )
        safe_line_chart(monthly, x="Submission_YM", y="Cases",
                        title="Monthly Case Volume",
                        xlabel="Month", ylabel="Number of Cases")

    # Charts row 2
    col_c, col_d = st.columns(2)
    with col_c:
        if "TPA_or_Payer" in df.columns and "Processing_Time_Days" in df.columns:
            avg_proc_tpa = (
                df.groupby("TPA_or_Payer")["Processing_Time_Days"]
                .mean()
                .reset_index()
                .dropna()
            )
            avg_proc_tpa.columns = ["TPA / Payer", "Avg Processing Time (Days)"]
            avg_proc_tpa = avg_proc_tpa.sort_values("Avg Processing Time (Days)", ascending=False)
            safe_bar_chart(avg_proc_tpa, x="TPA / Payer", y="Avg Processing Time (Days)",
                           title="Average Processing Time by TPA",
                           ylabel="Days")
    with col_d:
        if "Department" in df.columns and "Pending_Amount" in df.columns:
            pend_dept = (
                df.groupby("Department")["Pending_Amount"]
                .sum()
                .reset_index()
                .sort_values("Pending_Amount", ascending=False)
            )
            pend_dept.columns = ["Department", "Pending Amount"]
            safe_bar_chart(pend_dept, x="Department", y="Pending Amount",
                           title="Pending Amount by Department",
                           ylabel="Total Pending Amount (₹)")

    # Delayed by department
    col_e, col_f = st.columns(2)
    with col_e:
        if "Department" in df.columns and "Delayed" in df.columns:
            del_dept = df.groupby("Department")["Delayed"].sum().reset_index()
            del_dept.columns = ["Department", "Delayed Cases"]
            del_dept = del_dept.sort_values("Delayed Cases", ascending=False)
            safe_bar_chart(del_dept, x="Department", y="Delayed Cases",
                           title="Delayed Cases by Department",
                           ylabel="Number of Delayed Cases")
    with col_f:
        if "Predicted_Risk" in df.columns:
            risk_dist = df["Predicted_Risk"].value_counts().reset_index()
            risk_dist.columns = ["Risk Category", "Count"]
            safe_pie_chart(risk_dist, names="Risk Category", values="Count",
                           title="Predicted Delay Risk Distribution")


def section_claim_records(df):
    st.header("Claim Records")

    search_id = st.text_input("Search by Case ID", placeholder="e.g. C001")

    display_cols = [c for c in [
        "Case_ID", "Patient_Type", "Department", "TPA_or_Payer",
        "Service_Type", "Claim_Amount", "Approved_Amount",
        "Rejected_Amount", "Pending_Amount",
        "Submission_Date", "Settlement_Date", "Case_Status",
        "Processing_Time_Days", "Delayed",
        "Predicted_Delayed_Prob", "Predicted_Risk",
    ] if c in df.columns]

    view = df[display_cols].copy()

    if search_id.strip():
        if "Case_ID" in view.columns:
            view = view[view["Case_ID"].astype(str).str.contains(
                search_id.strip(), case=False, na=False)]

    if "Submission_Date" in view.columns:
        view["Submission_Date"] = view["Submission_Date"].dt.strftime("%Y-%m-%d").fillna("")
    if "Settlement_Date" in view.columns:
        view["Settlement_Date"] = view["Settlement_Date"].dt.strftime("%Y-%m-%d").fillna("")

    if "Predicted_Delayed_Prob" in view.columns:
        view["Predicted_Delayed_Prob"] = view["Predicted_Delayed_Prob"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else ""
        )

    st.info(f"Showing {len(view):,} of {len(df):,} records")
    st.dataframe(view, use_container_width=True)


def section_tpa_analysis(df):
    st.header("TPA Analysis")

    if "TPA_or_Payer" not in df.columns:
        st.error("TPA_or_Payer column not found in the dataset.")
        return

    tpa_group = df.groupby("TPA_or_Payer")

    summary = tpa_group.agg(
        Cases=("Case_ID", "count"),
        Total_Claim=("Claim_Amount", "sum"),
        Total_Approved=("Approved_Amount", "sum"),
        Total_Pending=("Pending_Amount", "sum"),
        Total_Rejected=("Rejected_Amount", "sum"),
    ).reset_index()

    if "Processing_Time_Days" in df.columns:
        avg_pt = tpa_group["Processing_Time_Days"].mean().reset_index()
        avg_pt.columns = ["TPA_or_Payer", "Avg_Processing_Days"]
        summary = summary.merge(avg_pt, on="TPA_or_Payer", how="left")

    if "Delayed" in df.columns:
        del_pct = (
            tpa_group["Delayed"].mean().reset_index()
        )
        del_pct.columns = ["TPA_or_Payer", "Delayed_Pct"]
        del_pct["Delayed_Pct"] = del_pct["Delayed_Pct"] * 100
        summary = summary.merge(del_pct, on="TPA_or_Payer", how="left")

    if "Predicted_Delayed" in df.columns:
        pred_del = (
            tpa_group["Predicted_Delayed"].mean().reset_index()
        )
        pred_del.columns = ["TPA_or_Payer", "Pred_Delayed_Pct"]
        pred_del["Pred_Delayed_Pct"] = pred_del["Pred_Delayed_Pct"] * 100
        summary = summary.merge(pred_del, on="TPA_or_Payer", how="left")

    if summary["Total_Claim"].sum() > 0:
        summary["Approval_Rate"] = (
            summary["Total_Approved"] / summary["Total_Claim"] * 100
        ).round(1)
        summary["Rejection_Rate"] = (
            summary["Total_Rejected"] / summary["Total_Claim"] * 100
        ).round(1)

    # Warn for small samples
    small = summary[summary["Cases"] < 10]
    if not small.empty:
        st.warning(
            f"TPAs with fewer than 10 cases (unreliable comparison): "
            f"{', '.join(small['TPA_or_Payer'].tolist())}"
        )

    st.dataframe(summary, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        safe_bar_chart(summary.sort_values("Cases", ascending=False),
                       x="TPA_or_Payer", y="Cases",
                       title="Case Volume by TPA", ylabel="Number of Cases")
    with col2:
        safe_bar_chart(summary.sort_values("Total_Claim", ascending=False),
                       x="TPA_or_Payer", y="Total_Claim",
                       title="Total Claim Amount by TPA", ylabel="Claim Amount (₹)")

    col3, col4 = st.columns(2)
    with col3:
        if "Approval_Rate" in summary.columns:
            safe_bar_chart(summary.sort_values("Approval_Rate", ascending=False),
                           x="TPA_or_Payer", y="Approval_Rate",
                           title="Approval Rate by TPA (%)", ylabel="Approval Rate (%)")
    with col4:
        if "Delayed_Pct" in summary.columns:
            safe_bar_chart(summary.sort_values("Delayed_Pct", ascending=False),
                           x="TPA_or_Payer", y="Delayed_Pct",
                           title="Delayed Case % by TPA", ylabel="Delayed %")

    if "Avg_Processing_Days" in summary.columns:
        safe_bar_chart(summary.sort_values("Avg_Processing_Days", ascending=False),
                       x="TPA_or_Payer", y="Avg_Processing_Days",
                       title="Average Processing Time by TPA",
                       ylabel="Days")


def section_department_analysis(df):
    st.header("Department Analysis")

    if "Department" not in df.columns:
        st.error("Department column not found in the dataset.")
        return

    dept_group = df.groupby("Department")

    summary = dept_group.agg(
        Cases=("Case_ID", "count"),
        Total_Claim=("Claim_Amount", "sum"),
        Total_Pending=("Pending_Amount", "sum"),
        Total_Rejected=("Rejected_Amount", "sum"),
        Total_Approved=("Approved_Amount", "sum"),
    ).reset_index()

    if "Processing_Time_Days" in df.columns:
        avg_pt = dept_group["Processing_Time_Days"].mean().reset_index()
        avg_pt.columns = ["Department", "Avg_Processing_Days"]
        summary = summary.merge(avg_pt, on="Department", how="left")

    if "Delayed" in df.columns:
        del_cnt = dept_group["Delayed"].sum().reset_index()
        del_cnt.columns = ["Department", "Delayed_Cases"]
        summary = summary.merge(del_cnt, on="Department", how="left")

    if "Predicted_Delayed" in df.columns:
        pred_del = dept_group["Predicted_Delayed"].sum().reset_index()
        pred_del.columns = ["Department", "Pred_Delayed_Cases"]
        summary = summary.merge(pred_del, on="Department", how="left")

    if summary["Total_Claim"].sum() > 0:
        summary["Approval_Rate"] = (
            summary["Total_Approved"] / summary["Total_Claim"] * 100
        ).round(1)

    st.dataframe(summary, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        safe_bar_chart(summary.sort_values("Cases", ascending=False),
                       x="Department", y="Cases",
                       title="Case Volume by Department")
    with col2:
        safe_bar_chart(summary.sort_values("Total_Pending", ascending=False),
                       x="Department", y="Total_Pending",
                       title="Pending Amount by Department", ylabel="₹")

    col3, col4 = st.columns(2)
    with col3:
        if "Delayed_Cases" in summary.columns:
            safe_bar_chart(summary.sort_values("Delayed_Cases", ascending=False),
                           x="Department", y="Delayed_Cases",
                           title="Delayed Cases by Department")
    with col4:
        if "Approval_Rate" in summary.columns:
            safe_bar_chart(summary.sort_values("Approval_Rate", ascending=False),
                           x="Department", y="Approval_Rate",
                           title="Approval Rate by Department (%)", ylabel="%")

    # Common rejection reasons by department
    if "Query_or_Rejection_Reason" in df.columns and "Department" in df.columns:
        st.subheader("Common Query / Rejection Reasons by Department")
        top_reasons = (
            df[df["Query_or_Rejection_Reason"] != "Not Specified"]
            .groupby(["Department", "Query_or_Rejection_Reason"])
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )
        if not top_reasons.empty:
            fig = px.bar(
                top_reasons.head(30),
                x="Count", y="Department",
                color="Query_or_Rejection_Reason",
                orientation="h",
                title="Top Query / Rejection Reasons by Department",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No query or rejection reasons recorded.")


def section_process_performance(df):
    st.header("Process Performance")

    if "Submission_YM" not in df.columns:
        st.warning("Submission date data not available for process performance analysis.")
        return

    monthly = df.groupby("Submission_YM").agg(
        Cases=("Case_ID", "count"),
        Total_Claim=("Claim_Amount", "sum"),
        Total_Approved=("Approved_Amount", "sum"),
        Total_Pending=("Pending_Amount", "sum"),
        Total_Rejected=("Rejected_Amount", "sum"),
    ).reset_index().sort_values("Submission_YM")

    if "Processing_Time_Days" in df.columns:
        pt_monthly = (
            df.groupby("Submission_YM")["Processing_Time_Days"]
            .agg(["mean", "median"])
            .reset_index()
            .rename(columns={"mean": "Avg_PT", "median": "Median_PT"})
        )
        monthly = monthly.merge(pt_monthly, on="Submission_YM", how="left")

    if "Delayed" in df.columns:
        del_monthly = (
            df.groupby("Submission_YM")["Delayed"].mean().reset_index()
        )
        del_monthly.columns = ["Submission_YM", "Delayed_Pct"]
        del_monthly["Delayed_Pct"] = del_monthly["Delayed_Pct"] * 100
        monthly = monthly.merge(del_monthly, on="Submission_YM", how="left")

    col1, col2 = st.columns(2)
    with col1:
        safe_line_chart(monthly, x="Submission_YM", y="Cases",
                        title="Monthly Case Volume",
                        xlabel="Month", ylabel="Cases")
    with col2:
        safe_line_chart(monthly, x="Submission_YM", y="Total_Claim",
                        title="Monthly Claim Amount",
                        xlabel="Month", ylabel="₹")

    col3, col4 = st.columns(2)
    with col3:
        if "Avg_PT" in monthly.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly["Submission_YM"], y=monthly["Avg_PT"],
                mode="lines+markers", name="Average"
            ))
            if "Median_PT" in monthly.columns:
                fig.add_trace(go.Scatter(
                    x=monthly["Submission_YM"], y=monthly["Median_PT"],
                    mode="lines+markers", name="Median"
                ))
            fig.update_layout(
                title="Processing Time by Month (Days)",
                xaxis_title="Month", yaxis_title="Days",
            )
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        if "Delayed_Pct" in monthly.columns:
            safe_line_chart(monthly, x="Submission_YM", y="Delayed_Pct",
                            title="Delayed Case % by Month",
                            xlabel="Month", ylabel="Delayed %")

    # Processing time distribution
    if "Processing_Time_Days" in df.columns:
        pt_valid = df["Processing_Time_Days"].dropna()
        if not pt_valid.empty:
            fig = px.histogram(
                df.dropna(subset=["Processing_Time_Days"]),
                x="Processing_Time_Days",
                nbins=30,
                title="Processing Time Distribution (Days)",
                labels={"Processing_Time_Days": "Processing Time (Days)"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # By patient type
    col5, col6 = st.columns(2)
    with col5:
        if "Patient_Type" in df.columns and "Processing_Time_Days" in df.columns:
            pt_type = (
                df.dropna(subset=["Processing_Time_Days"])
                .groupby("Patient_Type")["Processing_Time_Days"]
                .mean()
                .reset_index()
            )
            pt_type.columns = ["Patient Type", "Avg Processing Time (Days)"]
            safe_bar_chart(pt_type, x="Patient Type", y="Avg Processing Time (Days)",
                           title="Avg Processing Time by Patient Type")
    with col6:
        if "Service_Type" in df.columns and "Processing_Time_Days" in df.columns:
            pt_svc = (
                df.dropna(subset=["Processing_Time_Days"])
                .groupby("Service_Type")["Processing_Time_Days"]
                .mean()
                .reset_index()
                .sort_values("Processing_Time_Days", ascending=False)
            )
            pt_svc.columns = ["Service Type", "Avg Processing Time (Days)"]
            safe_bar_chart(pt_svc, x="Service Type", y="Avg Processing Time (Days)",
                           title="Avg Processing Time by Service Type")


def section_query_rejection(df):
    st.header("Query and Rejection Analysis")

    if "Query_or_Rejection_Reason" not in df.columns:
        st.error("Query_or_Rejection_Reason column not available.")
        return

    reasons_df = df[df["Query_or_Rejection_Reason"] != "Not Specified"].copy()

    if reasons_df.empty:
        st.info("No query or rejection reasons recorded in the dataset.")
        return

    # Top reasons
    top = reasons_df["Query_or_Rejection_Reason"].value_counts().reset_index()
    top.columns = ["Reason", "Count"]
    safe_bar_chart(top.head(20), x="Reason", y="Count",
                   title="Top Query / Rejection Reasons",
                   ylabel="Number of Cases")

    col1, col2 = st.columns(2)
    with col1:
        if "TPA_or_Payer" in df.columns:
            tpa_reasons = (
                reasons_df.groupby(["TPA_or_Payer", "Query_or_Rejection_Reason"])
                .size()
                .reset_index(name="Count")
                .sort_values("Count", ascending=False)
                .head(30)
            )
            if not tpa_reasons.empty:
                fig = px.bar(
                    tpa_reasons,
                    x="Count", y="TPA_or_Payer",
                    color="Query_or_Rejection_Reason",
                    orientation="h",
                    title="Rejection Reasons by TPA",
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Rejected_Amount" in df.columns:
            amt_reason = (
                reasons_df.groupby("Query_or_Rejection_Reason")["Rejected_Amount"]
                .sum()
                .reset_index()
                .sort_values("Rejected_Amount", ascending=False)
                .head(15)
            )
            amt_reason.columns = ["Reason", "Rejected Amount"]
            safe_bar_chart(amt_reason, x="Reason", y="Rejected Amount",
                           title="Rejected Amount by Reason", ylabel="₹")

    # Pending amount by reason
    if "Pending_Amount" in df.columns:
        pend_reason = (
            reasons_df.groupby("Query_or_Rejection_Reason")["Pending_Amount"]
            .sum()
            .reset_index()
            .sort_values("Pending_Amount", ascending=False)
            .head(15)
        )
        pend_reason.columns = ["Reason", "Pending Amount"]
        safe_bar_chart(pend_reason, x="Reason", y="Pending Amount",
                       title="Pending Amount by Reason", ylabel="₹")

    # Trend over time
    if "Submission_YM" in df.columns:
        trend = (
            reasons_df.groupby("Submission_YM").size().reset_index(name="Queries/Rejections")
            .sort_values("Submission_YM")
        )
        safe_line_chart(trend, x="Submission_YM", y="Queries/Rejections",
                        title="Query / Rejection Trend Over Time",
                        xlabel="Month")

    # Predicted delay by reason
    if "Delayed" in df.columns:
        delay_reason = (
            reasons_df.groupby("Query_or_Rejection_Reason")["Delayed"]
            .mean()
            .reset_index()
        )
        delay_reason.columns = ["Reason", "Delayed Rate"]
        delay_reason["Delayed Rate (%)"] = delay_reason["Delayed Rate"] * 100
        delay_reason = delay_reason.sort_values("Delayed Rate (%)", ascending=False).head(15)
        safe_bar_chart(delay_reason, x="Reason", y="Delayed Rate (%)",
                       title="Delayed Rate by Query / Rejection Reason", ylabel="%")


def section_predictive_analytics(df, pipeline_a, metadata_a, pipeline_b=None, metadata_b=None, pipeline_c=None, metadata_c=None):
    st.header("Predictive Analytics")

    if pipeline_a is None or metadata_a is None:
        st.error("No trained model found. Run `python train_model.py` first.")
        return

    st.info(
        "Three leakage-free models are trained exclusively on features available "
        "at or before submission time. No post-outcome amounts "
        "(Approved_Amount, Rejected_Amount, Pending_Amount) are used as inputs."
    )

    # ── leakage disclosure ────────────────────────────────────────────────────
    with st.expander("Leakage Prevention — Excluded Columns", expanded=False):
        leaky = metadata_a.get("leakage_excluded", [])
        st.write("The following columns are **excluded** from all model features "
                 "because they are determined by the outcome (data leakage):")
        st.dataframe(pd.DataFrame({"Excluded Column": leaky}), use_container_width=True)
        st.write(
            "Approved_Amount / Rejected_Amount / Pending_Amount are set by the "
            "adjudication result itself (e.g. Rejected → Rejected_Amount = Claim_Amount). "
            "Using them as inputs would give the model direct access to the answer, "
            "producing spuriously perfect accuracy that would not generalise."
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A — Approved vs Not-Approved (binary)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Model A — Approved vs Not-Approved (Binary Classification)")
    ma = metadata_a
    mA1, mA2, mA3, mA4 = st.columns(4)
    mA1.metric("Algorithm", ma.get("model_name", "N/A"))
    mA2.metric("Task", "Binary Classification")
    mA3.metric("Target", "Approved (1) vs Not (0)")
    mA4.metric("Test Accuracy", f"{ma.get('metrics', {}).get('accuracy', 0):.2%}")

    st.caption(ma.get("model_note", ""))

    bm_a = ma.get("metrics", {})
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Accuracy",       f"{bm_a.get('accuracy',  0):.4f}")
    a2.metric("Precision",      f"{bm_a.get('precision', 0):.4f}")
    a3.metric("Recall",         f"{bm_a.get('recall',    0):.4f}")
    a4.metric("F1 Score",       f"{bm_a.get('f1',        0):.4f}")
    a5.metric("ROC-AUC",        f"{bm_a.get('roc_auc',   0):.4f}" if bm_a.get('roc_auc') else "N/A")

    cv_a = bm_a.get("cv_accuracy_mean")
    if cv_a:
        st.metric(
            "5-Fold CV Accuracy",
            f"{cv_a:.4f} ± {bm_a.get('cv_accuracy_std', 0):.4f}",
            help="Computed on the full labelled dataset with stratified splits."
        )

    with st.expander("Model A — Candidate Comparison", expanded=True):
        all_r_a = ma.get("all_results", {})
        if all_r_a:
            rows = []
            for name, m in all_r_a.items():
                rows.append({
                    "Model":       name,
                    "Accuracy":    m.get("accuracy"),
                    "Precision":   m.get("precision"),
                    "Recall":      m.get("recall"),
                    "F1":          m.get("f1"),
                    "ROC-AUC":     m.get("roc_auc"),
                    "CV Accuracy": m.get("cv_accuracy_mean"),
                    "CV Std":      m.get("cv_accuracy_std"),
                    "Selected":    "★" if name == ma.get("model_name") else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(f"Selection rule: {ma.get('selection_rule', '')}")

    cm_a = bm_a.get("confusion_matrix")
    classes_a = ma.get("target_classes", ["Not Approved", "Approved"])
    if cm_a and len(cm_a) == len(classes_a):
        st.subheader("Model A — Confusion Matrix (Test Set)")
        fig_cm_a = px.imshow(
            cm_a, text_auto=True,
            x=classes_a, y=classes_a,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            title="Approved vs Not — Confusion Matrix",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_cm_a, use_container_width=True)

    if "Predicted_Approved_Prob" in df.columns:
        st.subheader("Model A — Predicted Approval Probability Distribution")
        fig_pa = px.histogram(
            df, x="Predicted_Approved_Prob", nbins=25,
            title="Distribution of Predicted Approval Probability",
            labels={"Predicted_Approved_Prob": "P(Approved)"},
        )
        fig_pa.update_xaxes(title_text="P(Approved)")
        fig_pa.update_yaxes(title_text="Number of Claims")
        st.plotly_chart(fig_pa, use_container_width=True)

        risk_counts = df["Predicted_Approval_Risk"].value_counts().reset_index() if "Predicted_Approval_Risk" in df.columns else None
        if risk_counts is not None:
            risk_counts.columns = ["Risk Category", "Count"]
            safe_bar_chart(risk_counts, x="Risk Category", y="Count",
                           title="Model A — Predicted Approval Risk Distribution",
                           ylabel="Number of Claims")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL B — Case_Status multi-class
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Model B — Case Status Multi-class (4 Classes)")
    if pipeline_b is None or metadata_b is None:
        st.info("Model B not loaded. Re-run `python train_model.py` to include the multi-class model.")
    else:
        mb = metadata_b
        bB1, bB2, bB3, bB4 = st.columns(4)
        bB1.metric("Algorithm", mb.get("model_name", "N/A"))
        bB2.metric("Task", "Multi-class Classification")
        bB3.metric("Target", "Case_Status (4 classes)")
        bB4.metric("Test Accuracy", f"{mb.get('metrics', {}).get('accuracy', 0):.2%}")

        st.caption(mb.get("model_note", ""))

        bm_b = mb.get("metrics", {})
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Accuracy",       f"{bm_b.get('accuracy',  0):.4f}")
        b2.metric("Precision (wtd)",f"{bm_b.get('precision', 0):.4f}")
        b3.metric("Recall (wtd)",   f"{bm_b.get('recall',    0):.4f}")
        b4.metric("F1 (wtd)",       f"{bm_b.get('f1',        0):.4f}")

        cv_b = bm_b.get("cv_accuracy_mean")
        if cv_b:
            st.metric("5-Fold CV Accuracy",
                      f"{cv_b:.4f} ± {bm_b.get('cv_accuracy_std', 0):.4f}")

        with st.expander("Model B — Candidate Comparison", expanded=True):
            all_r_b = mb.get("all_results", {})
            if all_r_b:
                rows_b = []
                for name, m in all_r_b.items():
                    rows_b.append({
                        "Model":       name,
                        "Accuracy":    m.get("accuracy"),
                        "Precision":   m.get("precision"),
                        "Recall":      m.get("recall"),
                        "F1":          m.get("f1"),
                        "CV Accuracy": m.get("cv_accuracy_mean"),
                        "CV Std":      m.get("cv_accuracy_std"),
                        "Selected":    "★" if name == mb.get("model_name") else "",
                    })
                st.dataframe(pd.DataFrame(rows_b), use_container_width=True)
            st.caption(f"Selection rule: {mb.get('selection_rule', '')}")

        cm_b = bm_b.get("confusion_matrix")
        classes_b = mb.get("target_classes")
        if cm_b and classes_b and len(cm_b) == len(classes_b):
            st.subheader("Model B — Confusion Matrix (Test Set)")
            fig_cm_b = px.imshow(
                cm_b, text_auto=True,
                x=classes_b, y=classes_b,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                title="Case Status (4-class) — Confusion Matrix",
                color_continuous_scale="Purples",
            )
            st.plotly_chart(fig_cm_b, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL C — Delayed flag
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Model C — Delayed Claim Risk (Binary, Directional)")
    if pipeline_c is None or metadata_c is None:
        st.info("Model C not loaded. Re-run `python train_model.py`.")
    else:
        mc = metadata_c
        cC1, cC2, cC3, cC4 = st.columns(4)
        cC1.metric("Algorithm", mc.get("model_name", "N/A"))
        cC2.metric("Task", "Binary Classification")
        cC3.metric("Threshold", f"> {mc.get('delay_threshold_days', 23)} days")
        cC4.metric("Test F1", f"{mc.get('metrics', {}).get('f1', 0):.4f}")

        st.warning(
            "Processing time in this dataset is statistically near-uniform "
            "(Kolmogorov-Smirnov test p = 0.14). No submission-time feature "
            "strongly predicts processing duration. Honest accuracy ceiling is "
            "~55–65%. This model provides a directional risk indicator only — "
            "not a precise forecast."
        )
        st.caption(mc.get("model_note", ""))

        bm_c = mc.get("metrics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  f"{bm_c.get('accuracy',  0):.4f}")
        c2.metric("Precision", f"{bm_c.get('precision', 0):.4f}")
        c3.metric("Recall",    f"{bm_c.get('recall',    0):.4f}")
        c4.metric("F1",        f"{bm_c.get('f1',        0):.4f}")
        c5.metric("ROC-AUC",   f"{bm_c.get('roc_auc',  0):.4f}" if bm_c.get('roc_auc') else "N/A")

        cv_c = bm_c.get("cv_accuracy_mean")
        if cv_c:
            st.metric("5-Fold CV Accuracy",
                      f"{cv_c:.4f} ± {bm_c.get('cv_accuracy_std', 0):.4f}")

        with st.expander("Model C — Candidate Comparison", expanded=True):
            all_r_c = mc.get("all_results", {})
            if all_r_c:
                rows_c = []
                for name, m in all_r_c.items():
                    rows_c.append({
                        "Model":       name,
                        "Accuracy":    m.get("accuracy"),
                        "F1":          m.get("f1"),
                        "Recall":      m.get("recall"),
                        "ROC-AUC":     m.get("roc_auc"),
                        "CV Accuracy": m.get("cv_accuracy_mean"),
                        "CV Std":      m.get("cv_accuracy_std"),
                        "Selected":    "★" if name == mc.get("model_name") else "",
                    })
                st.dataframe(pd.DataFrame(rows_c), use_container_width=True)
            st.caption(f"Selection rule: {mc.get('selection_rule', '')}")

        if "Predicted_Delayed_Prob" in df.columns:
            st.subheader("Model C — Predicted Delay Probability Distribution")
            fig_pc = px.histogram(
                df, x="Predicted_Delayed_Prob", nbins=25,
                title="Distribution of Predicted Delay Probability",
                labels={"Predicted_Delayed_Prob": "P(Delayed)"},
            )
            fig_pc.update_xaxes(title_text="P(Delayed)")
            fig_pc.update_yaxes(title_text="Number of Claims")
            st.plotly_chart(fig_pc, use_container_width=True)

        if "Predicted_Risk" in df.columns:
            risk_c = df["Predicted_Risk"].value_counts().reset_index()
            risk_c.columns = ["Risk Category", "Count"]
            safe_bar_chart(risk_c, x="Risk Category", y="Count",
                           title="Model C — Delay Risk Category Distribution",
                           ylabel="Number of Claims")

    st.divider()

    # ── feature list ──────────────────────────────────────────────────────────
    with st.expander("Features Used by All Models", expanded=False):
        from train_model import CAT_FEATS, NUM_FEATS
        feat_df = pd.DataFrame({
            "Feature":  CAT_FEATS + NUM_FEATS,
            "Type":     ["Categorical"] * len(CAT_FEATS) + ["Numeric"] * len(NUM_FEATS),
        })
        st.dataframe(feat_df, use_container_width=True)
        st.write(
            "Categorical features are one-hot encoded (handle_unknown='ignore'). "
            "Numeric features are standard-scaled after median imputation. "
            "All preprocessing is fitted on the training split only (Pipeline)."
        )

    # ── ethical / limitations notice ─────────────────────────────────────────
    st.divider()
    st.subheader("Model Limitations and Ethical Notice")
    st.warning(
        "All three models are trained on 500 records. "
        "Performance metrics are honest (leakage-free) but the dataset is small. "
        "Results must not be used for clinical decisions, claim adjudication, "
        "or patient management without independent validation by a qualified "
        "healthcare revenue-cycle professional. "
        "This is a decision-support tool — human review is mandatory."
    )


def _derive_features(inputs, df):
    """
    Given a dict of raw user inputs, compute all derived submission-time features.
    Returns a single-row DataFrame ready for pipeline.predict().
    """
    claim = float(inputs.get("Claim_Amount", 1.0))
    inputs["Claim_Log"]   = float(np.log1p(claim))
    inputs["Sub_Quarter"] = ((inputs.get("Sub_Month", 1) - 1) // 3) + 1

    tpa  = inputs.get("TPA_or_Payer",  "Unknown")
    svc  = inputs.get("Service_Type",  "Unknown")
    dept = inputs.get("Department",    "Unknown")
    inputs["TPA_Service"] = f"{tpa}_{svc}"
    inputs["Dept_TPA"]    = f"{dept}_{tpa}"
    inputs["Dept_Svc"]    = f"{dept}_{svc}"

    # group-mean Claim_Amount from training distribution (approximate from df)
    for col in ["TPA_or_Payer", "Department", "Service_Type", "Patient_Type"]:
        key = col + "_cmean"
        val = inputs.get(col, None)
        if val and col in df.columns:
            gm = df.groupby(col)["Claim_Amount"].mean()
            inputs[key] = float(gm.get(val, df["Claim_Amount"].mean()))
        else:
            inputs[key] = float(df["Claim_Amount"].mean()) if "Claim_Amount" in df.columns else 0.0

    return inputs


def _build_form_inputs(cat_feats, num_feats, df, key_prefix):
    """
    Render Streamlit input widgets for the clean feature set.
    Returns (inputs_dict, sub_date) — sub_date is the selected date widget value.
    """
    # Derived interaction features — user does not enter these directly
    DERIVED_CATS = {"TPA_Service", "Dept_TPA", "Dept_Svc"}

    inputs = {}
    primary_cats = [f for f in cat_feats if f not in DERIVED_CATS]

    st.subheader("Claim Details (Submission-time only)")
    st.caption(
        "Enter only the information known at the time of claim submission. "
        "No post-outcome amounts are required or accepted."
    )

    col_left, col_right = st.columns(2)
    for i, feat in enumerate(primary_cats):
        col = col_left if i % 2 == 0 else col_right
        if feat in df.columns:
            options = sorted(df[feat].dropna().unique().tolist())
            inputs[feat] = col.selectbox(
                feat.replace("_", " "), options=options, key=f"{key_prefix}_{feat}"
            )
        else:
            inputs[feat] = col.text_input(
                feat.replace("_", " "), value="Unknown", key=f"{key_prefix}_{feat}"
            )

    inputs["Claim_Amount"] = st.number_input(
        "Claim Amount (Rs)", min_value=0.0, value=100000.0,
        step=1000.0, key=f"{key_prefix}_claim_amount"
    )
    sub_date = st.date_input(
        "Submission Date", value=datetime.date.today(), key=f"{key_prefix}_sub_date"
    )
    inputs["Sub_Month"]   = sub_date.month
    inputs["Sub_Weekday"] = sub_date.weekday()

    return inputs, sub_date


def section_prediction_form(pipeline_a, metadata_a, pipeline_b, metadata_b, pipeline_c, metadata_c, df):
    st.header("Predict New Claim")

    if pipeline_a is None or metadata_a is None:
        st.error("Model not loaded. Run `python train_model.py` first.")
        return

    st.info(
        "All three prediction models use only features available at or before "
        "submission time — no post-outcome amounts are accepted as inputs."
    )

    tab_a, tab_b, tab_c = st.tabs([
        "Model A: Approval Prediction",
        "Model B: Case Status (4-class)",
        "Model C: Delay Risk",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB A — Approved vs Not-Approved
    # ══════════════════════════════════════════════════════════════════════════
    with tab_a:
        st.write(
            "**Model A** predicts whether a claim will be Approved (1) or Not Approved (0). "
            f"Selected algorithm: **{metadata_a.get('model_name', 'N/A')}**. "
            f"Test accuracy: **{metadata_a.get('metrics', {}).get('accuracy', 0):.2%}**."
        )
        feats_a = metadata_a.get("all_features", [])
        cat_a   = metadata_a.get("cat_features", [])
        num_a   = metadata_a.get("num_features", [])

        with st.form("form_model_a"):
            inp_a, _ = _build_form_inputs(cat_a, num_a, df, "a")
            submitted_a = st.form_submit_button("Predict Approval")

        if submitted_a:
            inp_a = _derive_features(inp_a, df)
            try:
                row_a = {f: inp_a.get(f, 0) for f in feats_a}
                X_a = pd.DataFrame([row_a])
                pred_a = pipeline_a.predict(X_a)[0]
                proba_a = pipeline_a.predict_proba(X_a)[0]
                confidence_a = float(proba_a.max())
                classes_a = list(pipeline_a.classes_)
                label_a = "Approved" if pred_a == 1 else "Not Approved"
                risk_a  = "High" if proba_a[1] >= 0.70 else ("Medium" if proba_a[1] >= 0.40 else "Low")

                st.divider()
                st.subheader("Model A — Prediction Result")
                ra1, ra2, ra3 = st.columns(3)
                ra1.metric("Predicted Outcome", label_a)
                ra2.metric("Confidence", f"{confidence_a:.1%}")
                ra3.metric("Approval Risk", risk_a)

                prob_df_a = pd.DataFrame({
                    "Class":       ["Not Approved", "Approved"],
                    "Probability": [proba_a[0], proba_a[1]],
                })
                fig_a = px.bar(
                    prob_df_a, x="Class", y="Probability",
                    title="Model A — Class Probabilities",
                    labels={"Probability": "Probability", "Class": "Predicted Class"},
                    color="Class",
                    color_discrete_map={"Approved": "#22c55e", "Not Approved": "#ef4444"},
                )
                fig_a.update_yaxes(range=[0, 1])
                st.plotly_chart(fig_a, use_container_width=True)

                if pred_a == 1:
                    st.success("Claim predicted to be APPROVED.")
                else:
                    st.error("Claim predicted to be NOT APPROVED.")

                st.caption("Decision-support estimate only. Human review is mandatory.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB B — Case_Status multi-class
    # ══════════════════════════════════════════════════════════════════════════
    with tab_b:
        if pipeline_b is None or metadata_b is None:
            st.info("Model B not available. Re-run `python train_model.py`.")
        else:
            st.write(
                "**Model B** predicts the full Case_Status outcome across 4 classes: "
                "Approved / Rejected / Partially Approved / Pending. "
                f"Selected algorithm: **{metadata_b.get('model_name', 'N/A')}**. "
                f"Test accuracy: **{metadata_b.get('metrics', {}).get('accuracy', 0):.2%}** "
                "(honest, leakage-free)."
            )
            feats_b = metadata_b.get("all_features", [])
            cat_b   = metadata_b.get("cat_features", [])
            num_b   = metadata_b.get("num_features", [])

            with st.form("form_model_b"):
                inp_b, _ = _build_form_inputs(cat_b, num_b, df, "b")
                submitted_b = st.form_submit_button("Predict Case Status")

            if submitted_b:
                inp_b = _derive_features(inp_b, df)
                try:
                    row_b = {f: inp_b.get(f, 0) for f in feats_b}
                    X_b = pd.DataFrame([row_b])
                    pred_b = pipeline_b.predict(X_b)[0]
                    proba_b = pipeline_b.predict_proba(X_b)[0]
                    confidence_b = float(proba_b.max())
                    classes_b = list(pipeline_b.classes_)

                    st.divider()
                    st.subheader("Model B — Prediction Result")
                    rb1, rb2 = st.columns(2)
                    rb1.metric("Predicted Status", str(pred_b))
                    rb2.metric("Confidence", f"{confidence_b:.1%}")

                    prob_df_b = pd.DataFrame({
                        "Status":      classes_b,
                        "Probability": proba_b.tolist(),
                    }).sort_values("Probability", ascending=False)
                    fig_b = px.bar(
                        prob_df_b, x="Status", y="Probability",
                        title="Model B — Class Probabilities",
                        labels={"Probability": "Probability", "Status": "Case Status"},
                        color="Status",
                    )
                    fig_b.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig_b, use_container_width=True)

                    if pred_b == "Approved":
                        st.success("Claim predicted: APPROVED.")
                    elif pred_b == "Rejected":
                        st.error("Claim predicted: REJECTED.")
                    elif pred_b == "Partially Approved":
                        st.warning("Claim predicted: PARTIALLY APPROVED.")
                    else:
                        st.info(f"Claim predicted: {pred_b}")

                    st.caption("Decision-support estimate only. Human review is mandatory.")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB C — Delayed flag
    # ══════════════════════════════════════════════════════════════════════════
    with tab_c:
        if pipeline_c is None or metadata_c is None:
            st.info("Model C not available. Re-run `python train_model.py`.")
        else:
            threshold_c = metadata_c.get("delay_threshold_days", 23)
            st.write(
                f"**Model C** predicts whether a claim will take more than **{threshold_c} days** "
                "to settle (delayed = 1). Uses only submission-time features. "
                f"Selected algorithm: **{metadata_c.get('model_name', 'N/A')}**. "
                f"Test F1: **{metadata_c.get('metrics', {}).get('f1', 0):.4f}**."
            )
            st.warning(
                "Processing time is near-uniform in this dataset. "
                "This is a directional risk indicator — not a precise forecast."
            )
            feats_c = metadata_c.get("all_features", [])
            cat_c   = metadata_c.get("cat_features", [])
            num_c   = metadata_c.get("num_features", [])

            with st.form("form_model_c"):
                inp_c, _ = _build_form_inputs(cat_c, num_c, df, "c")
                submitted_c = st.form_submit_button("Predict Delay Risk")

            if submitted_c:
                inp_c = _derive_features(inp_c, df)
                try:
                    row_c = {f: inp_c.get(f, 0) for f in feats_c}
                    X_c = pd.DataFrame([row_c])
                    pred_c  = pipeline_c.predict(X_c)[0]
                    proba_c = pipeline_c.predict_proba(X_c)[0]
                    prob_delayed = float(proba_c[1])
                    risk_c = "High" if prob_delayed >= 0.70 else ("Medium" if prob_delayed >= 0.40 else "Low")

                    st.divider()
                    st.subheader("Model C — Delay Risk Result")
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Prediction", "Delayed" if pred_c == 1 else "Not Delayed")
                    rc2.metric("Delay Probability", f"{prob_delayed:.1%}")
                    rc3.metric("Risk Category", risk_c)

                    prob_df_c = pd.DataFrame({
                        "Class":       ["Not Delayed", "Delayed"],
                        "Probability": [proba_c[0], proba_c[1]],
                    })
                    fig_c = px.bar(
                        prob_df_c, x="Class", y="Probability",
                        title="Model C — Delay Risk Probabilities",
                        labels={"Probability": "Probability", "Class": ""},
                        color="Class",
                        color_discrete_map={"Delayed": "#ef4444", "Not Delayed": "#22c55e"},
                    )
                    fig_c.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig_c, use_container_width=True)

                    if risk_c == "High":
                        st.error("HIGH delay risk — prioritise this claim.")
                    elif risk_c == "Medium":
                        st.warning("MEDIUM delay risk — monitor closely.")
                    else:
                        st.success("LOW delay risk.")

                    st.caption(
                        "Directional estimate only. Processing time is near-uniformly "
                        "distributed in this dataset. Human review is mandatory."
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")


def section_data_quality(df, audit, pipeline, metadata):
    st.header("Data Quality and Workbook Audit")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Workbook Information")
        st.write(f"**Filename:** {os.path.basename(audit.get('workbook_path', 'N/A'))}")
        st.write(f"**Path:** {audit.get('workbook_path', 'N/A')}")
        st.write(f"**Selected Worksheet:** {audit.get('selected_sheet', 'N/A')}")
        st.write(f"**Number of Worksheets:** {len(audit.get('sheet_names', []))}")
        st.write(f"**Sheet Names:** {', '.join(audit.get('sheet_names', []))}")
        st.write(f"**Total Records:** {audit.get('total_rows', 0):,}")
        st.write(f"**Total Columns:** {audit.get('total_cols', 0)}")

    with col2:
        st.subheader("Column Mapping")
        orig = audit.get("original_columns", [])
        mapped = audit.get("mapped_columns", [])
        if orig:
            col_map_df = pd.DataFrame({
                "Original Column": orig,
                "Mapped Column": mapped[:len(orig)] if mapped else orig,
            })
            st.dataframe(col_map_df, use_container_width=True)

    st.subheader("Data Validation Summary")
    val_data = {
        "Check": [
            "Missing Required Columns",
            "Missing Value Cells",
            "Invalid Date Values",
            "Settlement Before Submission",
            "Negative Amounts",
            "Financial Inconsistencies",
            "Invalid Case Status Values",
            "Duplicate Case IDs",
        ],
        "Count / Status": [
            len(audit.get("missing_required", [])) or "None",
            sum(audit.get("missing_values", {}).values()) or "None",
            audit.get("invalid_dates", 0) or "None",
            audit.get("settlement_before_submission", 0) or "None",
            sum(audit.get("negative_amounts", {}).values()) or "None",
            audit.get("financial_inconsistencies", 0) or "None",
            len(audit.get("invalid_statuses", [])) or "None",
            audit.get("duplicate_case_ids", 0) or "None",
        ],
    }
    st.dataframe(pd.DataFrame(val_data), use_container_width=True)

    if audit.get("missing_required"):
        st.error(f"Missing required columns: {audit['missing_required']}")
    if audit.get("missing_values"):
        st.subheader("Missing Values by Column")
        mv_df = pd.DataFrame(
            list(audit["missing_values"].items()),
            columns=["Column", "Missing Count"]
        )
        st.dataframe(mv_df, use_container_width=True)
    if audit.get("invalid_statuses"):
        st.warning(
            f"Unexpected Case_Status values found (not renamed): "
            f"{audit['invalid_statuses']}"
        )

    st.subheader("Model Readiness")
    model_ready = pipeline is not None
    if model_ready:
        st.success(f"Model loaded successfully: {metadata.get('model_name', 'Unknown')}")
        st.write(f"- **Trained at:** {metadata.get('trained_at', 'N/A')}")
        st.write(f"- **Training records:** {metadata.get('train_size', 'N/A')}")
        st.write(f"- **Test records:** {metadata.get('test_size', 'N/A')}")
        st.write(f"- **F1 Score (test):** {metadata.get('metrics', {}).get('f1', 'N/A')}")
    else:
        st.error(
            "Model not found. Run `python train_model.py` to train and save the model."
        )

    model_file_exists = os.path.isfile(MODEL_PATH)
    if model_file_exists:
        model_size = os.path.getsize(MODEL_PATH)
        st.write(f"**Model file:** {MODEL_PATH} ({model_size:,} bytes)")
    else:
        st.write(f"**Model file:** Not found at {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    st.title("Healthcare Revenue Cycle & TPA Process Analytics Dashboard")
    st.caption(
        "Predictive Analytics | Machine Learning | Claims Management | "
        "Decision Support — Not a Clinical System"
    )

    # --- Load data ---
    with st.spinner("Loading TPA DATA.xlsx..."):
        wb_path = locate_workbook()
        if wb_path is None:
            st.error(
                "TPA DATA.xlsx was not found. "
                "Please place the actual Excel workbook in the project root "
                "directory or in the DATA/ folder.\n\n"
                "Expected locations:\n"
                "- TPA DATA.xlsx\n"
                "- DATA/TPA data.xlsx\n"
                "- data/TPA DATA.xlsx"
            )
            st.stop()

        df_raw, audit = load_data()

    if df_raw is None:
        st.error(
            "Failed to load the workbook. "
            "Please verify that TPA DATA.xlsx is a valid Excel file."
        )
        st.stop()

    # --- Load model ---
    with st.spinner("Loading trained model..."):
        pipeline_a, metadata_a, pipeline_b, metadata_b, pipeline_c, metadata_c, model_error = load_model()

    if model_error:
        st.warning(f"Model status: {model_error}")

    # --- Sidebar ---
    filters, threshold, delayed_filter = apply_sidebar_filters(df_raw)

    # --- Filter ---
    df = filter_dataframe(df_raw, filters, delayed_filter, threshold)

    # --- Predictions ---
    df = add_predictions(df, pipeline_a, metadata_a, pipeline_c, metadata_c)

    # --- Navigation ---
    sections = [
        "Overview Dashboard",
        "Claim Records",
        "TPA Analysis",
        "Department Analysis",
        "Process Performance",
        "Query and Rejection Analysis",
        "Predictive Analytics",
        "Predict New Claim",
        "Data Quality and Model Validation",
    ]
    selected_section = st.sidebar.radio("Navigation", sections)

    st.sidebar.divider()
    st.sidebar.caption(
        f"Dataset: {os.path.basename(audit.get('workbook_path', 'N/A'))} | "
        f"{len(df):,} records (filtered) / {len(df_raw):,} total"
    )

    # --- Render selected section ---
    if selected_section == "Overview Dashboard":
        section_overview(df, pipeline_a, metadata_a, threshold)

    elif selected_section == "Claim Records":
        section_claim_records(df)

    elif selected_section == "TPA Analysis":
        section_tpa_analysis(df)

    elif selected_section == "Department Analysis":
        section_department_analysis(df)

    elif selected_section == "Process Performance":
        section_process_performance(df)

    elif selected_section == "Query and Rejection Analysis":
        section_query_rejection(df)

    elif selected_section == "Predictive Analytics":
        section_predictive_analytics(df, pipeline_a, metadata_a, pipeline_b, metadata_b, pipeline_c, metadata_c)

    elif selected_section == "Predict New Claim":
        section_prediction_form(pipeline_a, metadata_a, pipeline_b, metadata_b, pipeline_c, metadata_c, df_raw)

    elif selected_section == "Data Quality and Model Validation":
        section_data_quality(df, audit, pipeline_a, metadata_a)


if __name__ == "__main__":
    main()
