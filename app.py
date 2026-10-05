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
DELAY_THRESHOLD_DAYS = 30
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

    df["Delayed"] = 0
    if "Processing_Time_Days" in df.columns:
        resolved = df["Processing_Time_Days"].notna()
        df.loc[resolved & (df["Processing_Time_Days"] > delay_threshold), "Delayed"] = 1
    if "Submission_Date" in df.columns:
        unresolved_mask = df.get("Processing_Time_Days", pd.Series(dtype=float)).isna()
        elapsed = (analysis_date - df["Submission_Date"]).dt.days
        df.loc[unresolved_mask & (elapsed > delay_threshold), "Delayed"] = 1

    if "Submission_Date" in df.columns:
        df["Submission_Month"] = df["Submission_Date"].dt.month.fillna(0).astype(int)
        df["Submission_Weekday"] = df["Submission_Date"].dt.dayofweek.fillna(0).astype(int)
        df["Submission_Year"] = df["Submission_Date"].dt.year
        df["Submission_YM"] = df["Submission_Date"].dt.to_period("M").astype(str)

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
        return None, None, "Model file not found at models/model.pkl. Run train_model.py first."
    try:
        payload = joblib.load(MODEL_PATH)
        pipeline = payload.get("pipeline")
        metadata = payload.get("metadata")
        if pipeline is None:
            return None, None, "Saved file does not contain a valid pipeline."
        return pipeline, metadata, None
    except Exception as e:
        return None, None, f"Failed to load model: {str(e)}"


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
        fdf["Delayed"] = 0
        resolved = fdf["Processing_Time_Days"].notna()
        fdf.loc[resolved & (fdf["Processing_Time_Days"] > threshold), "Delayed"] = 1
        if "Submission_Date" in fdf.columns:
            unresolved_mask = fdf["Processing_Time_Days"].isna()
            elapsed = (analysis_date - fdf["Submission_Date"]).dt.days
            fdf.loc[unresolved_mask & (elapsed > threshold), "Delayed"] = 1

    if delayed_filter == "Delayed Only":
        fdf = fdf[fdf["Delayed"] == 1]
    elif delayed_filter == "Not Delayed":
        fdf = fdf[fdf["Delayed"] == 0]

    return fdf


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def add_predictions(df, pipeline, metadata):
    """Add predicted probability and risk category to df. Returns modified df."""
    if pipeline is None or metadata is None:
        return df

    all_features = metadata.get("all_features", [])
    available = [f for f in all_features if f in df.columns]
    if len(available) < len(all_features):
        return df

    try:
        X = df[all_features].copy()
        df["Predicted_Delayed_Prob"] = pipeline.predict_proba(X)[:, 1]
        df["Predicted_Delayed"] = pipeline.predict(X)

        def risk_category(p):
            if p >= 0.70:
                return "High"
            elif p >= 0.40:
                return "Medium"
            else:
                return "Low"

        df["Predicted_Risk"] = df["Predicted_Delayed_Prob"].apply(risk_category)
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


def section_predictive_analytics(df, pipeline, metadata):
    st.header("Predictive Analytics")

    if pipeline is None or metadata is None:
        st.error(
            "No trained model found. "
            "Please run `python train_model.py` to train the model first."
        )
        return

    m = metadata
    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Name", m.get("model_name", "N/A"))
    col2.metric("Problem Type", m.get("problem_type", "N/A").title())
    col3.metric("Target Variable", m.get("target_col", "N/A"))

    st.info(
        f"**Target:** {m.get('target_col')} — A claim is classified as delayed if its "
        f"processing time exceeds {m.get('delay_threshold_days', 30)} calendar days. "
        f"For unresolved claims, elapsed time from submission is used. "
        f"This is a binary classification task."
    )

    # Feature list
    st.subheader("Features Used")
    feat_col, leak_col = st.columns(2)
    with feat_col:
        st.write("**Input Features (no leakage):**")
        for f in m.get("all_features", []):
            st.write(f"- {f}")
    with leak_col:
        st.write("**Excluded (data-leakage prevention):**")
        for f in m.get("leakage_excluded", []):
            st.write(f"- {f}")

    # Candidate model comparison
    st.subheader("Candidate Model Comparison")
    all_results = m.get("all_results", {})
    if all_results:
        rows = []
        for name, metrics in all_results.items():
            rows.append({
                "Model": name,
                "Accuracy": metrics.get("accuracy"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1": metrics.get("f1"),
                "ROC-AUC": metrics.get("roc_auc"),
                "CV F1 Mean": metrics.get("cv_f1_mean"),
                "Selected": "✅" if name == m.get("model_name") else "",
            })
        compare_df = pd.DataFrame(rows)
        st.dataframe(compare_df, use_container_width=True)

    st.caption(
        f"**Selection rule:** {m.get('selection_rule', 'Highest F1, Recall as tiebreaker.')}"
    )

    # Best model metrics
    st.subheader("Best Model Evaluation Metrics")
    best_metrics = m.get("metrics", {})
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.4f}")
    mc2.metric("Precision", f"{best_metrics.get('precision', 0):.4f}")
    mc3.metric("Recall", f"{best_metrics.get('recall', 0):.4f}")
    mc4.metric("F1 Score", f"{best_metrics.get('f1', 0):.4f}")
    if best_metrics.get("roc_auc") is not None:
        mc5, mc6 = st.columns(2)
        mc5.metric("ROC-AUC", f"{best_metrics['roc_auc']:.4f}")
        if best_metrics.get("cv_f1_mean") is not None:
            mc6.metric(
                "CV F1 (5-fold)",
                f"{best_metrics['cv_f1_mean']:.4f} ± {best_metrics.get('cv_f1_std', 0):.4f}"
            )

    # Confusion matrix
    if "confusion_matrix" in best_metrics:
        cm = best_metrics["confusion_matrix"]
        if cm:
            st.subheader("Confusion Matrix (Test Set)")
            cm_fig = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Not Delayed", "Delayed"],
                y=["Not Delayed", "Delayed"],
                title="Confusion Matrix",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(cm_fig, use_container_width=True)

    # Predicted delay distribution in dataset
    if "Predicted_Delayed_Prob" in df.columns:
        st.subheader("Predicted Delay Probability Distribution")
        fig = px.histogram(
            df, x="Predicted_Delayed_Prob", nbins=20,
            title="Distribution of Predicted Delay Probability",
            labels={"Predicted_Delayed_Prob": "Delay Probability"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Limitations
    st.subheader("Model Limitations and Warnings")
    st.warning(
        "**Important:** This model is trained on a limited dataset and is intended for "
        "decision-support only. Predictions are estimates and must be reviewed by a qualified "
        "healthcare revenue-cycle professional before any action is taken. "
        "This application is not a clinical decision-support system and is not production-ready "
        "without further validation."
    )
    st.info(
        "The model uses only information available at the time of claim submission. "
        "Post-outcome fields (Settlement_Date, Approved_Amount, Rejected_Amount, "
        "Pending_Amount after settlement) are excluded to prevent data leakage."
    )


def section_prediction_form(pipeline, metadata, df):
    st.header("Predict Delay Risk for a New Claim")

    if pipeline is None or metadata is None:
        st.error("Model not loaded. Run `python train_model.py` first.")
        return

    all_features = metadata.get("all_features", [])
    cat_features = metadata.get("cat_features", [])
    num_features = metadata.get("num_features", [])

    if not all_features:
        st.error("Model metadata does not contain feature information.")
        return

    st.info(
        "Enter claim details below to predict whether the claim is at risk of being delayed. "
        "Only information available at submission time is used."
    )

    with st.form("prediction_form"):
        st.subheader("Claim Details")
        user_inputs = {}

        # Categorical inputs from actual dataset values
        for feat in cat_features:
            if feat in df.columns:
                options = sorted(df[feat].dropna().unique().tolist())
                user_inputs[feat] = st.selectbox(f"{feat.replace('_', ' ')}", options=options)
            else:
                user_inputs[feat] = st.text_input(f"{feat.replace('_', ' ')}")

        # Numeric inputs
        if "Claim_Amount" in num_features:
            user_inputs["Claim_Amount"] = st.number_input(
                "Claim Amount (₹)", min_value=0.0, value=50000.0, step=1000.0
            )

        if "Submission_Month" in num_features or "Submission_Weekday" in num_features:
            sub_date = st.date_input(
                "Submission Date",
                value=datetime.date.today(),
            )
            if "Submission_Month" in num_features:
                user_inputs["Submission_Month"] = sub_date.month
            if "Submission_Weekday" in num_features:
                user_inputs["Submission_Weekday"] = sub_date.weekday()

        submitted = st.form_submit_button("Predict Delay Risk")

    if submitted:
        missing_inputs = [f for f in all_features if f not in user_inputs]
        if missing_inputs:
            st.error(f"Missing required inputs: {missing_inputs}")
            return

        try:
            input_df = pd.DataFrame([{f: user_inputs[f] for f in all_features}])
            prob = pipeline.predict_proba(input_df)[0][1]
            pred_class = pipeline.predict(input_df)[0]

            if prob >= 0.70:
                risk = "High"
            elif prob >= 0.40:
                risk = "Medium"
            else:
                risk = "Low"

            st.divider()
            st.subheader("Prediction Result")
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted Class", "Delayed" if pred_class == 1 else "Not Delayed")
            r2.metric("Delay Probability", f"{prob:.1%}")
            r3.metric("Risk Category", risk)

            if risk == "High":
                st.error(
                    "HIGH risk of delay. Human review recommended before processing."
                )
            elif risk == "Medium":
                st.warning(
                    "MEDIUM risk of delay. Monitor this claim closely."
                )
            else:
                st.success(
                    "LOW risk of delay. Claim is likely to be processed within the threshold."
                )

            st.caption(
                "This prediction is a decision-support estimate only. "
                "Human review is mandatory before taking any action."
            )

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


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
        pipeline, metadata, model_error = load_model()

    if model_error:
        st.warning(f"Model status: {model_error}")

    # --- Sidebar ---
    filters, threshold, delayed_filter = apply_sidebar_filters(df_raw)

    # --- Filter ---
    df = filter_dataframe(df_raw, filters, delayed_filter, threshold)

    # --- Predictions ---
    if pipeline is not None and metadata is not None:
        df = add_predictions(df, pipeline, metadata)

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
        section_overview(df, pipeline, metadata, threshold)

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
        section_predictive_analytics(df, pipeline, metadata)

    elif selected_section == "Predict New Claim":
        section_prediction_form(pipeline, metadata, df_raw)

    elif selected_section == "Data Quality and Model Validation":
        section_data_quality(df, audit, pipeline, metadata)


if __name__ == "__main__":
    main()
