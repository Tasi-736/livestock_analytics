# Deploy LivestockLens with Streamlit Community Cloud

## Streamlit entry point

Use this exact main file path in Streamlit Community Cloud:

`livestock_analytics/streamlit_app.py`

The repository currently keeps the application files inside the `livestock_analytics/` folder. Do not enter only `streamlit_app.py` unless you move the files to the repository root.

## Dependencies

The `requirements.txt` file is in the same `livestock_analytics/` folder as `streamlit_app.py`. Streamlit Community Cloud supports a requirements file beside an entry point in a subdirectory.

## Python version

Choose Python 3.12 in Advanced settings (or leave Streamlit Community Cloud's documented 3.12 default). Do not select the local WSL Python 3.14 for this deployment.

## Local run

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r livestock_analytics/requirements.txt
streamlit run livestock_analytics/streamlit_app.py
```

## Included features

- Overview KPIs and annual production trends
- Descriptive statistics, production share, 2025 distribution and top 10 hog producers
- Scatterplot and Pearson correlation heatmap
- Four classifiers: Logistic Regression, Decision Tree, Random Forest and Gradient Boosting
- Validation-based conditional SMOTENC augmentation (only when validation accuracy is below 80%; keep augmented model only when validation accuracy improves)
- Test-set Accuracy, Precision, Recall, F1-score and ROC-AUC comparison
- Confusion matrices and ROC curves
- Single and bulk 2026 High/Low category predictions
- Search/filter data records and downloadable CSV outputs
- Dataset upload support for ZIP, ARFF and CSV

The app reuses `app.py` for its data preparation and modeling logic. The original Flask interface and the final Colab notebook are retained in this repository.

Predictions are High/Low category estimates, not exact-volume forecasts or official PSA forecasts. Synthetic SMOTENC samples are training-only and are not official livestock observations.
