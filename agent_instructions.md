# agent_instructions.md
# IBM Bob — Agent Instructions
# Healthcare Revenue Cycle and TPA Process Analytics Dashboard

## Purpose

This document provides instructions for IBM Bob (or any AI coding assistant)
working on this project. It defines the context, constraints, dataset handling,
machine-learning logic, and ethical safeguards that must be respected at all times.

---

## Application Purpose

Build and maintain a **local machine-learning web application** for healthcare
revenue-cycle management (RCM) and Third-Party Administrator (TPA) process analytics.

The application must:
- Load the actual TPA claim dataset from `TPA DATA.xlsx`.
- Validate, clean, and analyse the data.
- Train machine-learning models to predict claim-processing delays.
- Present an interactive Streamlit analytics dashboard.
- Provide a real-time prediction interface for new claims.

---

## Healthcare Revenue-Cycle Context

In healthcare, a **revenue cycle** covers the entire financial process from patient
registration through claim submission to final payment. Key challenges include:

- TPA delays in processing and settling claims.
- High rejection or query rates leading to financial losses.
- Lack of visibility into departmental performance.
- Difficulty identifying at-risk claims before delays occur.

This application addresses these challenges using data-driven analytics and
machine-learning predictions.

---

## Actual Dataset File

```
TPA DATA.xlsx    (project root)
DATA/TPA data.xlsx   (DATA sub-folder — fallback)
```

- Do **not** rename the workbook unless absolutely necessary.
- Do **not** create dummy, sample, or synthetic data.
- Search these paths in order:
  1. `TPA DATA.xlsx`
  2. `DATA/TPA data.xlsx`
  3. `data/TPA DATA.xlsx`
  4. `data/TPA data.xlsx`
- If the workbook is missing, display a clear error and stop all analysis.

---

## Required Project Structure

```
project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── agent_instructions.md
├── .env.example
├── TPA DATA.xlsx          ← actual workbook
├── DATA/
│   └── TPA data.xlsx      ← alternative location
├── data/
│   └── .gitkeep
└── models/
    └── model.pkl
```

---

## Python-Only Requirement

- Use **Python** only.
- Use **Streamlit** for the web interface.
- Do **not** use HTML, CSS, or JavaScript.
- Do **not** use `st.markdown(..., unsafe_allow_html=True)`.
- Do **not** use Flask, Django, React, Node.js, or any other web framework.
- Do **not** use TensorFlow, PyTorch, XGBoost, LightGBM, or CatBoost.
- Do **not** use external AI APIs, ML APIs, chatbots, or generative-AI features.
- Do **not** use API keys.
- Do **not** use absolute file paths. Use relative paths only.

---

## Prohibited Technologies

| Prohibited | Use Instead |
|---|---|
| HTML / CSS / JS | Streamlit components |
| Flask / Django | Streamlit |
| TensorFlow / PyTorch | scikit-learn |
| XGBoost / LightGBM / CatBoost | LogisticRegression / DecisionTree |
| External AI APIs | Local scikit-learn model |
| Absolute paths | Relative paths |
| `unsafe_allow_html=True` | Standard Streamlit components |
| Jupyter Notebooks | Python scripts only |

---

## Excel Loading Process

```python
import pandas as pd

df = pd.read_excel("TPA DATA.xlsx", engine="openpyxl")
```

- Always use `engine="openpyxl"` for `.xlsx` files.
- Inspect sheet names before loading.
- Select the sheet with the highest count of matching expected columns.
- If multiple sheets are equally matched, provide a `st.selectbox` for manual selection.
- Do **not** combine worksheets unless they contain compatible claim-level records.

---

## Expected Dataset Fields

| Field | Type | Notes |
|---|---|---|
| Case_ID | String | Unique identifier; check for duplicates |
| Patient_Type | Categorical | e.g. Inpatient, Outpatient |
| Department | Categorical | Clinical department |
| TPA_or_Payer | Categorical | Third-party administrator or insurer |
| Service_Type | Categorical | Type of medical service |
| Claim_Amount | Numeric | Total amount claimed |
| Approved_Amount | Numeric | Amount approved |
| Rejected_Amount | Numeric | Amount rejected |
| Pending_Amount | Numeric | Amount pending |
| Submission_Date | Date | Date claim was submitted |
| Settlement_Date | Date | Date settled; blank for unresolved |
| Case_Status | Categorical | Approved / Pending / Rejected / Queried / In Process / Settled |
| Query_or_Rejection_Reason | String | Reason for query/rejection; may be blank |

---

## Column Mapping Rules

1. Strip leading and trailing whitespace.
2. Collapse repeated internal spaces to one space.
3. Match against canonical aliases (case-insensitive, with/without underscores).
4. If an unambiguous mapping exists, apply it automatically.
5. Preserve original column names in the audit log.
6. Display original → normalised → mapped column names in the Data Quality section.
7. Do **not** silently change the meaning of any field.
8. Do **not** fabricate missing columns.

---

## KPI Definitions

| KPI | Formula |
|---|---|
| Total Cases | COUNT(Case_ID) |
| Total Claim Amount | SUM(Claim_Amount) |
| Approved Amount | SUM(Approved_Amount) |
| Pending Amount | SUM(Pending_Amount) |
| Rejected Amount | SUM(Rejected_Amount) |
| Approval Rate | Approved / Claim × 100 |
| Rejection Rate | Rejected / Claim × 100 |
| Avg Processing Time | MEAN(Processing_Time_Days) for resolved claims |
| Delayed Cases | COUNT where Processing_Time_Days > threshold |

---

## Delayed-Case Assumption

```python
DELAY_THRESHOLD_DAYS = 30
```

- Resolved claim delayed: `Processing_Time_Days > DELAY_THRESHOLD_DAYS`
- Unresolved claim delayed: `(today - Submission_Date).days > DELAY_THRESHOLD_DAYS`
- Allow user to adjust via `st.sidebar.slider(min=7, max=90)`.
- Do **not** use an estimated processing time as the actual time for historical KPIs.

---

## Automatic Target Detection

1. Inspect available columns.
2. Check if `Submission_Date` and `Settlement_Date` are present.
3. Compute `Processing_Time_Days = Settlement_Date - Submission_Date`.
4. Derive `Delayed = 1 if Processing_Time_Days > threshold else 0`.
5. Use `Delayed` as the classification target.
6. Explain why the target was selected.
7. Display the target, problem type, class distribution, and threshold.

---

## Classification and Regression Logic

### Classification (primary task)

- **Target:** `Delayed` (binary: 0 / 1)
- **Candidate models:** `LogisticRegression`, `DecisionTreeClassifier`
- **Selection metric:** Highest F1 score; Recall as tiebreaker
- **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, CV F1

### Regression (secondary — extend if required)

- **Target:** `Processing_Time_Days` (continuous)
- **Candidate models:** `Ridge`, `DecisionTreeRegressor`
- **Selection metric:** Lowest MAE; RMSE as tiebreaker
- **Evaluation:** MAE, RMSE, R-squared

---

## Data Preprocessing Logic

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

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
```

- Fit the preprocessor **only on training data**.
- Apply it to test data via the Pipeline `.transform()` method.
- Never fit on the full dataset before splitting.

---

## Data-Leakage Prevention

**Excluded from features:**

| Column | Reason |
|---|---|
| Settlement_Date | Post-outcome information |
| Processing_Time_Days | Directly encodes the target |
| Delayed | The target variable itself |
| Approved_Amount | Finalised after settlement |
| Rejected_Amount | Finalised after settlement |
| Pending_Amount | Changes after settlement |

**Allowed features (available at submission):**

- `Patient_Type`, `Department`, `TPA_or_Payer`, `Service_Type`
- `Claim_Amount`, `Submission_Month`, `Submission_Weekday`

---

## Model Evaluation

- Compare all candidate models on the hold-out test set.
- Display a comparison table in the Predictive Analytics section.
- Select and document the best model.
- Display confusion matrix for the best model.
- Display all metrics in the dashboard.
- Do **not** claim clinical validation.
- Do **not** claim production-readiness.

---

## Model Saving

```python
import joblib

payload = {"pipeline": pipeline, "metadata": metadata}
joblib.dump(payload, "models/model.pkl")
```

Loading in `app.py`:

```python
payload = joblib.load("models/model.pkl")
pipeline = payload["pipeline"]
metadata = payload["metadata"]
```

- Save the **complete pipeline** (preprocessing + estimator), not just the estimator.
- Include metadata: model name, features, target, metrics, threshold, timestamp.
- Use `joblib`, not `pickle`, for efficiency with NumPy arrays.

---

## Prediction Interface Requirements

- The form must be generated from the actual model feature list in `metadata`.
- Use `st.selectbox` for categorical features with options from the actual dataset.
- Use `st.number_input` for numeric features.
- Use `st.date_input` for date-derived features.
- Use `st.form` and `st.form_submit_button`.
- Validate all inputs before predicting.
- Never hard-code predictions.
- Display: predicted class, delay probability, and risk category (Low / Medium / High).

---

## Relative Path Requirements

- Always use `os.path.join()` for multi-segment paths.
- Never use absolute paths (e.g. `C:\Users\...` or `/home/user/...`).
- Test path resolution from the project root directory.
- Use `os.path.isfile()` before loading any file.

---

## GitHub and Streamlit Deployment Requirements

- All source files must be committed to the repository.
- `TPA DATA.xlsx` or `DATA/TPA data.xlsx` must be committed.
- `models/model.pkl` should be committed if trained on anonymised data.
- `requirements.txt` must list all dependencies with no pinned versions (for Streamlit Cloud).
- The app must launch with `streamlit run app.py` with no additional configuration.
- No environment variables or API keys are required.

---

## Ethical Safeguards

1. **No PII:** Do not use real patient names, IDs, or contact information.
2. **Decision support only:** Predictions must be labelled as estimates requiring human review.
3. **Not a clinical system:** Display a clear disclaimer in the app.
4. **Transparency:** Display model features, excluded features, and selection reasoning.
5. **No fabrication:** Never generate replacement data for a missing workbook.
6. **No silent modifications:** Preserve original data values; document all transformations.

---

## Project Limitations

1. Model trained on 500 records; limited generalisation.
2. Delay threshold is an assumption, not a clinical or regulatory standard.
3. Missing settlement dates for unresolved claims introduce target uncertainty.
4. No real-time data ingestion.
5. Not validated for clinical or financial decision-making.
6. No adversarial testing performed.
