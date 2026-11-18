DEAN'S DILEMMA — STREAMLIT APP (NO CLUSTERING)

CONTENTS
- app.py: Streamlit dashboard application
- Dean Dilemma.xlsx: source dataset used by the dashboard
- requirements.txt: core dependencies
- requirements-sdv.txt: optional SDV dependency for Gaussian Copula synthetic data
- run_windows.bat: Windows one-click launcher

RUN ON WINDOWS
1. Install Python 3.11 or 3.12 and select "Add Python to PATH".
2. Extract this folder.
3. Double-click run_windows.bat.
4. Wait for dependencies to install and Streamlit to start.
5. Open http://localhost:8501 if the browser does not open automatically.

OPTIONAL SYNTHETIC DATA
The dashboard includes an SDV Gaussian Copula synthetic-data workflow and
original-vs-synthetic distribution comparisons. To enable it, open PowerShell
in this folder and run:
    python -m pip install -r requirements-sdv.txt
Then restart the app. If SDV is not installed, all other dashboard features
remain available.

RUN MANUALLY
    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

DASHBOARD CONTENTS
- Executive placement dashboard
- Missing values and Complete Case Analysis (CCA)
- Original vs CCA numerical and categorical distribution comparisons
- Descriptive statistics, skewness, correlation and categorical relationships
- IQR, Z-score, percentile and Isolation Forest outlier diagnostics
- Logistic Regression, LDA, Random Forest and AdaBoost
- Optional stratified 5-fold cross-validation and Random Forest tuning
- Confusion matrices, class reports, ROC and precision-recall curves
- Threshold analysis, balanced accuracy, MCC and feature importance
- SDV Gaussian Copula synthetic data generation (optional dependency)
- Original vs synthetic target, numerical and categorical comparisons
- Individual student placement probability scoring
- CSV exports

NO CLUSTERING
This project intentionally contains no K-means, clustering, cluster profiles,
cluster assignments or cluster charts.

NOTES
- The app computes analysis from the bundled Excel workbook when launched.
- It does not depend on Jupyter variables or a notebook pickle file.
- Synthetic rows are generated only when the user clicks the SDV button.
- Synthetic records are not additional real students. Distribution similarity
  does not establish privacy protection or model validity.
- This is an educational analytics prototype, not an automated decision system.
