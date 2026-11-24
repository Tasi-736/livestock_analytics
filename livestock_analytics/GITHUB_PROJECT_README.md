# LivestockLens

Philippine livestock production analytics for 2010–2025, based on the final Google Colab analysis notebook.

## Deploy the Streamlit dashboard

Use Streamlit Community Cloud and set the main file path to:

`livestock_analytics/streamlit_app.py`

Choose Python 3.12 in Advanced settings. The app dependencies are in `livestock_analytics/requirements.txt` beside the entry point.

See `livestock_analytics/STREAMLIT_DEPLOYMENT.md` for deployment and local run instructions.

## Project contents

- `app.py`: shared data cleaning, Flask API and machine-learning pipeline
- `streamlit_app.py`: Streamlit interface for analytics, evaluation and prediction
- `templates/` and `static/`: original Flask interface retained as a backup
- `requirements.txt`: Streamlit, Plotly and backend dependencies
- `notebooks/LivestockLens_Final_Colab_Analysis.ipynb`: final Colab notebook

## Colab-aligned analytical workflow

1. Data cleaning and province-level filtering for 2010–2025
2. Descriptive analytics and data visualizations
3. Logistic Regression, Decision Tree, Random Forest and Gradient Boosting
4. Conditional SMOTENC training augmentation when validation accuracy is below 80%; retain it only if validation accuracy improves
5. Confusion matrices, ROC curves and model-comparison metrics
6. 2026 High/Low production category prediction

The target is a High/Low class based on the median provincial production. It is not an exact-volume forecast or an official PSA forecast. Synthetic SMOTENC samples are training-only and are not official livestock observations.
