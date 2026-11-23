# LivestockLens: Livestock Production Analytics Website

A local web application for exploring the Philippine Statistics Authority livestock annual volume dataset (2010–2025). The website follows the data cleaning, descriptive analytics, classification models, validation-based SMOTENC augmentation, test-set evaluation, and 2026 High/Low category prediction process in the supplied Colab notebook.

## Features

- Dashboard with record, province, animal-category, and total-production indicators.
- Descriptive analytics: production totals and shares by animal, annual trends, top 10 provinces, 2025 distribution, and descriptive statistics.
- Filters for animal type, province, and year.
- Searchable and paginated provincial records table.
- Four classification models: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Model comparison using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- ROC curves and confusion matrices for all four models.
- Conditional SMOTENC augmentation: attempted only when validation accuracy is below 80%; an augmented model is kept only when validation accuracy improves.
- Prediction of a High or Low production category for a selected animal, province, and year.
- CSV downloads for cleaned records, model comparison results, and 2026 category predictions.
- Dataset upload support for the GitHub ZIP archive, ARFF, and CSV files.

## Important data note

The original dataset is **not included** in this generated project. The GitHub archive could not be downloaded in the build environment. On startup, the app attempts to retrieve the raw dataset URL used in the supplied Colab notebook. If that automatic download does not work, upload the repository ZIP or the original `.csv.arff` file in the website's **Data Source** panel. The app can read an ARFF/CSV file directly from inside the ZIP, so extraction is not required.

The project does not create or substitute fake livestock observations. It only analyses the uploaded/downloaded source data.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10 or newer
- Internet access during dependency installation (and for the optional automatic dataset download)
- A modern browser (Chrome, Edge, Firefox, or Safari)

## Run on Windows

1. Extract `livestock_analytics_website.zip` to a folder.
2. Open the extracted folder.
3. Double-click `run.bat`, or follow the manual steps below.
4. When the terminal shows `Open http://127.0.0.1:5000`, open that address in your browser.
5. When the dataset has not loaded automatically, open **Overview → Load the livestock dataset**, choose the GitHub ZIP archive or `.csv.arff` file, and upload it.
6. Open **Model evaluation** and click **Train four models**.
7. Open **Predictions** to generate a High/Low category result.

Keep the terminal window open while using the website. Press `Ctrl+C` in that window to stop the local server.

## Run manually

```bash
# 1. Open a terminal in the project folder

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# On Windows Command Prompt use:
# .venv\Scripts\activate.bat

# On macOS/Linux use:
# source .venv/bin/activate

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Start the website
python app.py
```

Open **http://127.0.0.1:5000**.

## Dataset source

- GitHub repository archive: <https://github.com/Tasi-736/Livestock-Annual-Volume-of-Production-by-Animal-Type-and-Province-for-2010-2025/archive/refs/heads/main.zip>
- Original raw-data URL used in the notebook: <https://raw.githubusercontent.com/Tasi-736/Livestock-Annual-Volume-of-Production-by-Animal-Type-and-Province-for-2010-2025/main/Livestock-%20Annual%20Volume%20of%20Production%20by%20Animal%20Type%20and%20Province%2C%202010-2025.csv.arff>
- PSA OpenSTAT livestock and poultry metadata: <https://openstat.psa.gov.ph/Metadata/Agriculture-Forestry-Fisheries/Livestock-and-Poultry>

The PSA metadata identifies the dataset as *Livestock: Annual Volume of Production by Animal Type and Province, 2010–2025*. The source reports the unit as metric tons. Review the PSA metadata and dataset notes when interpreting official statistical values.

## Data preparation rules used by this application

1. Load the ARFF file's comma-separated rows after the `@data` marker, or load a compatible CSV.
2. Convert the 2010–2025 year fields to numeric values; source missing-value markers such as `..` are treated as missing, not zero.
3. Retain the animal categories Carabao, Cattle, Goat, and Hog.
4. Retain province-level rows identified by exactly four leading dots in the original `Geolocation` field, as in the notebook. This removes national, regional, and six-dot city-level rows.
5. Convert the annual wide columns into one row per province, animal type, and year. Drop missing production observations.

## Classification and prediction method

- Target: `High Production` when the observation is greater than or equal to the median production in the cleaned provincial dataset; otherwise `Low Production`.
- Features: `Year`, encoded `Animal Type`, and encoded `Geolocation`.
- Split: 80% development and 20% final test, then 80% training and 20% validation inside the development set. This is equivalent to a 64% / 16% / 20% overall split.
- Models: Logistic Regression, Decision Tree (maximum depth 5), Random Forest (100 trees, maximum depth 6), and Gradient Boosting (100 estimators, learning rate 0.1, maximum depth 4).
- Conditional augmentation: when validation accuracy is below 80%, SMOTENC attempts to increase each training class count by 25%. Animal and province encoded columns are categorical inputs to SMOTENC. The augmented model is retained only if validation accuracy improves. The final test set is not used to create synthetic records or choose between baseline and augmented models.
- Evaluation: the selected model for each algorithm is evaluated against the untouched test set. The comparison is sorted by Accuracy, then ROC-AUC, then F1-score.
- Future category prediction: each algorithm is refit using all available real provincial observations. Synthetic training records are not used for the final future prediction refit.

**Interpretation caution:** The prediction interface predicts a High/Low category, not an exact future production volume. The output is a model-based category estimate, not an official PSA forecast. Because the supplied notebook uses a random row split, records from the same province can occur in both development and test sets; interpret the reported metrics in that context.

## Project structure

```text
livestock_analytics_website/
├── app.py
├── requirements.txt
├── README.md
├── run.bat
├── .gitignore
├── data/
│   └── README.txt
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

## Troubleshooting

**The dashboard says the dataset is not loaded**

Upload the repository ZIP or original `.csv.arff` file from the Data Source panel. The automatic download is optional and depends on your network allowing access to GitHub.

**`python` is not recognised**

Install Python 3.10+ and enable the option to add Python to PATH, then reopen the terminal.

**PowerShell does not allow virtual-environment activation**

Use Command Prompt and run `.venv\Scripts\activate.bat`, or temporarily allow activation in the current PowerShell session according to your school's device policy.

**Model training fails**

Check that both High and Low classes have enough non-missing province-level observations. Also verify that `pip install -r requirements.txt` completed successfully.

**The charts are blank**

The charts use Chart.js from jsDelivr, so the browser needs internet access to load that library. The Python data processing and API run locally.
