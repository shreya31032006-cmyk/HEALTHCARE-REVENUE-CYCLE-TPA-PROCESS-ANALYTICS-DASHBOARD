# Healthcare Revenue Cycle and TPA Process Analytics Dashboard with Predictive Analysis

## Project Description

A complete machine-learning web application for healthcare revenue-cycle management and
Third-Party Administrator (TPA) process analytics. The application loads actual claim-level
data from an Excel workbook, validates the dataset, trains machine-learning models to
predict claim-processing delays, and presents an interactive Streamlit dashboard with
KPIs, trend charts, and a real-time prediction interface.

This project was developed using **IBM Bob** as a generative-AI-assisted
software-development tool. IBM Bob generated the Python code, project structure,
documentation, validation logic, machine-learning logic, model-training script,
testing instructions, and deployment instructions.

---

## Problem Statement

Healthcare organisations and hospitals face significant revenue-cycle challenges:
delayed claim settlements, high rejection rates, and opaque TPA processes. This
application aims to provide data-driven insights into claim performance and predict
which incoming claims are at risk of being delayed, enabling proactive intervention.

---

## Project Objectives

1. Provide a comprehensive analytics dashboard for healthcare revenue-cycle KPIs.
2. Analyse TPA performance including approval rates, rejection rates, and processing times.
3. Analyse departmental claim patterns and financial metrics.
4. Predict whether a newly submitted claim will be delayed using machine learning.
5. Identify common query and rejection reasons and their financial impact.
6. Ensure data quality through systematic validation of the input workbook.

---

## Scope

- Claim-level analytics and KPI visualisation.
- TPA and department performance benchmarking.
- Binary classification: predict delayed vs not delayed.
- Processing-time analysis and trend monitoring.
- Query and rejection reason analysis.
- Data-quality audit of the source workbook.

---

## Features

- Interactive Streamlit dashboard with eight analytics sections.
- Sidebar filters: date range, department, TPA, patient type, status, service type.
- Adjustable delay threshold (slider).
- Machine-learning predictions with risk categories (Low / Medium / High).
- Candidate model comparison table (Logistic Regression vs Decision Tree).
- Confusion matrix visualisation.
- Prediction form for new claim risk assessment.
- Full data-quality and model-audit section.
- No HTML, CSS, JavaScript, or external APIs.
- Local-only execution — no external services required.

---

## Dataset Description

The actual dataset is supplied as an Excel workbook:

```
TPA DATA.xlsx  (or  DATA/TPA data.xlsx)
```

The workbook contains **500 claim records** across **13 columns** in a single worksheet
named `Sheet1`.

### Required Fields

| Field | Description |
|---|---|
| Case_ID | Unique claim identifier |
| Patient_Type | Inpatient or Outpatient |
| Department | Clinical department |
| TPA_or_Payer | Third-party administrator or insurer |
| Service_Type | Type of medical service |
| Claim_Amount | Total amount claimed (₹) |
| Approved_Amount | Amount approved by TPA (₹) |
| Rejected_Amount | Amount rejected (₹) |
| Pending_Amount | Amount still pending (₹) |
| Submission_Date | Date claim was submitted |
| Settlement_Date | Date claim was settled (blank for unresolved) |
| Case_Status | Current status of the claim |
| Query_or_Rejection_Reason | Reason for query or rejection (if any) |

---

## Excel Workbook Instructions

1. Place `TPA DATA.xlsx` in the **project root directory** or the `DATA/` folder.
2. Do **not** rename the workbook unless absolutely necessary.
3. The application searches these locations in order:
   - `TPA DATA.xlsx`
   - `DATA/TPA data.xlsx`
   - `data/TPA DATA.xlsx`
   - `data/TPA data.xlsx`
4. If the workbook is not found, a clear error message is displayed and processing stops.
5. No dummy or sample data is generated automatically.

### Worksheet Selection Logic

If the workbook contains multiple worksheets, the application automatically selects the
worksheet that has the highest number of matching expected column names. If automatic
selection is ambiguous, a `st.selectbox` is provided for manual selection.

---

## Data Validation Rules

| Validation | Rule |
|---|---|
| File check | Workbook must exist at one of the candidate paths |
| Column check | All 13 expected columns must be present |
| Date check | Dates parsed with `pd.to_datetime(errors='coerce')` |
| Date ordering | Settlement_Date must not be earlier than Submission_Date |
| Financial check | Amounts must be numeric and non-negative |
| Amount consistency | Approved + Rejected + Pending must not exceed Claim_Amount |
| Status check | Case_Status validated against allowed values |
| Duplicate check | Duplicate Case_ID values are reported, not silently removed |
| Missing values | Reported by column; target missing values excluded from training |

---

## KPI Definitions

| KPI | Definition |
|---|---|
| Total Cases | Count of all claim records in the filtered dataset |
| Total Claim Amount | Sum of Claim_Amount |
| Approved Amount | Sum of Approved_Amount |
| Pending Amount | Sum of Pending_Amount |
| Rejected Amount | Sum of Rejected_Amount |
| Approval Rate | Approved_Amount / Claim_Amount × 100 |
| Rejection Rate | Rejected_Amount / Claim_Amount × 100 |
| Average Processing Time | Mean of Processing_Time_Days for resolved claims |
| Delayed Cases | Count where Processing_Time_Days > Delay Threshold |

---

## Delayed-Case Assumption

A claim is defined as **delayed** when:

- For **resolved** claims: `Processing_Time_Days = Settlement_Date − Submission_Date > DELAY_THRESHOLD_DAYS`
- For **unresolved** claims: elapsed days since `Submission_Date` as of today > `DELAY_THRESHOLD_DAYS`

**Default threshold:** 30 calendar days.  
Users can adjust the threshold using the sidebar slider (range: 7–90 days).

---

## Machine-Learning Methodology

### Automatic Target Detection

The training script inspects available columns and prioritises the documented healthcare
target: predicting whether a claim will be delayed. The target is derived as:

```
Delayed = 1  if Processing_Time_Days > DELAY_THRESHOLD_DAYS
Delayed = 0  otherwise
```

### Classification Methodology

- **Task:** Binary classification (Delayed vs Not Delayed).
- **Pipeline:** scikit-learn `Pipeline` with `ColumnTransformer`.
- **Preprocessing:** `SimpleImputer` + `OneHotEncoder` for categoricals; `SimpleImputer` + `StandardScaler` for numerics.
- **Split:** 80% train / 20% test, `random_state=42`, stratified.

### Candidate Models

| Model | Notes |
|---|---|
| LogisticRegression | Interpretable linear baseline; `max_iter=1000` |
| DecisionTreeClassifier | Interpretable tree; `max_depth=6`, `min_samples_leaf=5` |

### Evaluation Metrics

| Metric | Used for |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Proportion of predicted delays that are actually delayed |
| Recall | Proportion of actual delays that are correctly predicted |
| F1 Score | Harmonic mean of precision and recall (primary selection metric) |
| ROC-AUC | Overall discriminative ability |
| Cross-validation F1 (5-fold) | Stability estimate |
| Confusion matrix | Breakdown of TP, FP, FN, TN |

**Model selection rule:** Highest F1 score on the hold-out test set. Recall is used as
a tiebreaker because false negatives (missed delays) are more costly in revenue-cycle management.

---

## Data-Leakage Prevention

The following columns are **excluded** from model features to prevent data leakage:

| Excluded Column | Reason |
|---|---|
| Settlement_Date | Available only after processing completes |
| Processing_Time_Days | Derived from Settlement_Date; directly encodes the target |
| Delayed | The target variable itself |
| Approved_Amount | Finalised after processing |
| Rejected_Amount | Finalised after processing |
| Pending_Amount | Changes after processing |

Only features available **at or near submission** are used:
- `Patient_Type`, `Department`, `TPA_or_Payer`, `Service_Type`
- `Claim_Amount`, `Submission_Month`, `Submission_Weekday`

Preprocessing (imputer, encoder, scaler) is fitted **only on training data** through the
Pipeline to prevent information leakage from the test set.

---

## Model Saving

The complete trained pipeline (preprocessing + model) is saved using `joblib`:

```
models/model.pkl
```

The saved file contains:
- `pipeline`: the full scikit-learn Pipeline object.
- `metadata`: model name, problem type, target, features, metrics, delay threshold,
  evaluation results for all candidates, training timestamp.

**Note:** `joblib` serialisation is used for efficiency. Do not load `model.pkl` files
from untrusted sources, as pickle-based formats can execute arbitrary code.

---

## Installation Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <project-directory>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Place the workbook

Copy `TPA DATA.xlsx` into the project root or `DATA/` folder.

---

## Local Run Instructions

### Train the model

```bash
python train_model.py
```

This will:
1. Locate and load `TPA DATA.xlsx`.
2. Validate the dataset.
3. Create derived variables.
4. Train and compare Logistic Regression and Decision Tree classifiers.
5. Save the best pipeline to `models/model.pkl`.
6. Print a training summary.

### Launch the dashboard

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

---

## Testing Instructions

Run the application manually and verify:

1. The workbook is located and loaded.
2. The selected worksheet is shown in Data Quality section.
3. All 8 dashboard sections render without errors.
4. Sidebar filters work correctly.
5. The delay threshold slider updates the Delayed count.
6. Predictions appear in Claim Records when model is loaded.
7. The prediction form generates risk categories dynamically.
8. The Data Quality section shows the validation summary.

Run unit tests if included:

```bash
python -m unittest discover -s tests
```

---

## GitHub Instructions

```bash
git init
git add .
git commit -m "Initial commit: Healthcare RCM TPA Analytics Dashboard"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

**Important:** Do not commit `models/model.pkl` to public repositories if it was trained
on sensitive data. Add `models/model.pkl` to `.gitignore` if required.

---

## Streamlit Community Cloud Deployment

1. Push the project to a public GitHub repository.
2. Ensure `TPA DATA.xlsx` is included in the repository (or `DATA/TPA data.xlsx`).
3. Go to [share.streamlit.io](https://share.streamlit.io).
4. Connect your GitHub account.
5. Select the repository and set the main file to `app.py`.
6. Deploy.

**Note:** Run `python train_model.py` locally and commit `models/model.pkl` to the
repository so the deployed app can load the trained model without re-training.

---

## Render Deployment Notes

1. Create a new **Web Service** on [render.com](https://render.com).
2. Connect the GitHub repository.
3. Set **Build Command:** `pip install -r requirements.txt`
4. Set **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Ensure `TPA DATA.xlsx` and `models/model.pkl` are committed to the repository.

---

## Data Privacy Note

This application is designed for use with **anonymised or synthetic** healthcare data only.
Do **not** use real patient names, national ID numbers, contact details, or any other
personally identifiable information (PII) as input data. Ensure compliance with applicable
data-protection legislation (e.g. DPDP Act, GDPR, HIPAA) before deploying this application
with real data.

---

## Ethical Use Note

Predictions generated by this application are **decision-support estimates only**.
They must be reviewed by a qualified healthcare revenue-cycle professional before any
action is taken. The application is not a clinical decision-support system and is not
validated for clinical or financial decisions without further rigorous testing.

---

## Project Limitations

1. The model is trained on a single dataset of 500 records; generalisation to other
   hospitals or claim environments is not guaranteed.
2. The model does not account for seasonal trends beyond submission month.
3. The delay definition (> 30 days) is a configurable assumption, not a clinical standard.
4. Missing settlement dates for unresolved claims introduce uncertainty into the target.
5. The application does not support real-time data ingestion or live database connections.
6. No adversarial robustness testing has been performed.
7. The model is not production-ready without further validation, retraining, and monitoring.
