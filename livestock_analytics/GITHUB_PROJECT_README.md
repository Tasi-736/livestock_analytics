# LivestockLens

LivestockLens is a Flask web application based on the final Google Colab analysis notebook.

## Project contents
- `app.py`: Flask application and data/model API
- `templates/` and `static/`: responsive website interface
- `requirements.txt`: Python dependencies, including Gunicorn
- `render.yaml` and `Procfile`: cloud deployment configuration
- `notebooks/LivestockLens_Final_Colab_Analysis.ipynb`: final Colab analysis notebook, including its executed outputs

## Analysis workflow
The notebook covers:
1. Data cleaning and provincial filtering for 2010–2025
2. Descriptive analytics and visualizations
3. Four classifiers: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting
4. Conditional SMOTENC training augmentation when validation accuracy is below 80%
5. Confusion matrices, ROC curves, and a model-comparison table
6. 2026 High/Low production category predictions

The target is a High/Low category based on the median production value. It is not an exact-volume forecast or an official PSA forecast. Synthetic SMOTENC rows are training-only and are not official livestock observations.

## Deploy
Connect this repository to Render. The included `render.yaml` defines the web service configuration. The app attempts to download the original dataset from the public dataset repository at startup. If that fails, use the Data Source upload panel to provide the original ARFF/CSV/ZIP dataset.

See `README.md` inside this folder for local setup and additional details.
