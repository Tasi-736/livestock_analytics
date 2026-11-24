# LivestockLens — Streamlit Community Cloud deployment

## Exact repository layout

The repository root contains `.streamlit/config.toml` and the `livestock_analytics/` project folder.

## Streamlit Community Cloud settings

- Repository: `Tasi-736/livestock_analytics`
- Branch: `main`
- Main file path: `livestock_analytics/streamlit_app.py`
- Python: use the Python version selected for the existing working Community Cloud app (Python 3.13 is supported by the pinned-compatible package ranges in `requirements.txt`).
- Secrets: none required by the application by default.

The `.streamlit/config.toml` theme file must remain at the repository root. The dependency file may remain beside the entry point at `livestock_analytics/requirements.txt`.

## Run locally from the repository root

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r livestock_analytics/requirements.txt
streamlit run livestock_analytics/streamlit_app.py
```

## Included application features

- Overview KPIs and annual production trends
- Descriptive statistics and filters for animal type, province and year
- Production share, 2025 provincial distribution, top 10 hog producers, scatterplot and Pearson correlation heatmap
- Four classifiers: Logistic Regression, Decision Tree, Random Forest and Gradient Boosting
- Validation-based conditional SMOTENC augmentation (only attempted when validation accuracy is below 80%; augmented model retained only if validation accuracy improves)
- Untouched test-set Accuracy, Precision, Recall, F1-score and ROC-AUC comparison
- Confusion matrices and ROC curves
- Single and bulk 2026 High/Low category predictions
- Searchable/filterable records and CSV downloads
- Dataset upload support for ZIP, ARFF and CSV

The Streamlit interface imports the existing `app.py` module for its dataset cleaning, model specifications, training split, SMOTENC and evaluation logic. Do not remove `app.py`. The Flask templates and static assets are retained as a backup/alternate interface.

## Data and interpretation notes

The app attempts to load the public source dataset at startup. If automatic loading is unavailable, upload the original ZIP, ARFF or CSV using the Data Source section. The final Colab notebook is retained in `livestock_analytics/notebooks/`.

The model predicts High/Low production categories, not exact future production volumes or official PSA forecasts. SMOTENC samples are used only for training and are not official livestock observations. The training and model evaluation state is held in application memory and may need to be trained again after an app restart.
