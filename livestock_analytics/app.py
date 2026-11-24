"""LivestockLens: Philippine livestock production analytics web application.

The data-processing and classification workflow follows the supplied Colab notebook.
Run locally with: python app.py
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTENC

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_URL = (
    "https://raw.githubusercontent.com/Tasi-736/"
    "Livestock-Annual-Volume-of-Production-by-Animal-Type-and-Province-for-2010-2025/"
    "main/Livestock-%20Annual%20Volume%20of%20Production%20by%20Animal%20Type%20and%20Province%2C%202010-2025.csv.arff"
)
REPOSITORY_ZIP_URL = (
    "https://github.com/Tasi-736/"
    "Livestock-Annual-Volume-of-Production-by-Animal-Type-and-Province-for-2010-2025/"
    "archive/refs/heads/main.zip"
)
YEARS = [str(year) for year in range(2010, 2026)]
ANIMALS = ["Carabao", "Cattle", "Goat", "Hog"]
ID_COLUMNS = ["Animal Type", "Geolocation"]
RANDOM_STATE = 42
ACCURACY_THRESHOLD = 0.80
AUGMENT_FACTOR = 1.25
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

STATE: dict[str, Any] = {"df": None, "source": None, "error": None}
MODEL_STATE: dict[str, Any] = {"ready": False, "error": None}
LOCK = threading.RLock()


def _json_number(value: Any) -> float | None:
    """Convert pandas/numpy values to JSON-safe floats."""
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _normalise_columns(columns: list[Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for column in columns:
        name = str(column).strip().strip("'\"\ufeff")
        key = re.sub(r"\s+", " ", name.lower().replace("_", " "))
        if key in {"animal type", "animal", "animal category"}:
            mapping[column] = "Animal Type"
        elif key in {"geolocation", "geography", "province", "location"}:
            mapping[column] = "Geolocation"
        elif key in {"year", "production year"}:
            mapping[column] = "Year"
        elif key in {"production", "value", "volume", "production volume"}:
            mapping[column] = "Production"
        elif name in YEARS:
            mapping[column] = name
    return mapping


def _read_dataset_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read ARFF-with-CSV-data, CSV, or a ZIP containing the original dataset."""
    lower_name = filename.lower()
    if lower_name.endswith(".zip") or zipfile.is_zipfile(io.BytesIO(file_bytes)):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            candidates = [
                name for name in archive.namelist()
                if not name.endswith("/") and name.lower().endswith((".arff", ".csv"))
            ]
            if not candidates:
                raise ValueError("The ZIP archive does not contain a .arff or .csv dataset file.")
            candidates.sort(key=lambda name: ("livestock" not in name.lower(), len(name)))
            inner_name = candidates[0]
            file_bytes = archive.read(inner_name)
            filename = inner_name

    text = file_bytes.decode("utf-8-sig", errors="replace")
    has_arff_data = any(line.strip().lower() == "@data" for line in text.splitlines())

    if has_arff_data:
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
        csv_text = "\n".join(lines[start:]).strip()
        if not csv_text:
            raise ValueError("The ARFF file has an @data section but contains no records.")
        raw = pd.read_csv(
            io.StringIO(csv_text),
            header=None,
            names=YEARS + ID_COLUMNS,
            quotechar="'",
            skipinitialspace=True,
            dtype=str,
            engine="python",
            on_bad_lines="skip",
        )
        return _wide_to_long(raw)

    # First try a regular CSV with a header.
    raw = pd.read_csv(io.StringIO(text), quotechar="'", dtype=str, engine="python")
    rename = _normalise_columns(list(raw.columns))
    raw = raw.rename(columns=rename)

    # Some ARFF exports are saved as CSV without a header row.
    if not set(ID_COLUMNS).issubset(raw.columns) and raw.shape[1] == len(YEARS) + 2:
        raw = pd.read_csv(
            io.StringIO(text), header=None, names=YEARS + ID_COLUMNS,
            quotechar="'", skipinitialspace=True, dtype=str, engine="python",
            on_bad_lines="skip",
        )

    if set(ID_COLUMNS).issubset(raw.columns) and set(YEARS).issubset(raw.columns):
        return _wide_to_long(raw[YEARS + ID_COLUMNS].copy())

    # Also accept a tidy CSV containing one row per year/province/animal.
    if {"Animal Type", "Geolocation", "Year", "Production"}.issubset(raw.columns):
        return _long_csv_to_long(raw)

    required = ", ".join(YEARS + ID_COLUMNS)
    raise ValueError(
        "Dataset columns were not recognised. Expected the 2010–2025 year columns plus "
        "Animal Type and Geolocation, or a tidy CSV with Year and Production. "
        f"Columns found: {', '.join(map(str, raw.columns))}."
    )


def _wide_to_long(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw.columns = [str(c).strip().strip("'\"") for c in raw.columns]
    missing = [column for column in YEARS + ID_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError("The source file is missing required columns: " + ", ".join(missing))

    for year in YEARS:
        raw[year] = pd.to_numeric(
            raw[year].astype(str).str.strip().replace({"..": np.nan, "": np.nan, "nan": np.nan}),
            errors="coerce",
        )

    raw["Animal Type"] = raw["Animal Type"].astype(str).str.strip().str.strip("'\"")
    raw["Geolocation_Raw"] = raw["Geolocation"].astype(str).str.strip().str.strip("'\"")
    raw["Geolocation"] = raw["Geolocation_Raw"].str.replace(r"^\.+", "", regex=True).str.strip()
    raw = raw[raw["Animal Type"].isin(ANIMALS)].copy()

    # Match the supplied notebook: retain exactly four-dot province rows only.
    province_mask = raw["Geolocation_Raw"].str.match(r"^\.{4}(?!\.)", na=False)
    province_wide = raw.loc[province_mask, YEARS + ID_COLUMNS[:-1] + ["Geolocation"]].copy()
    if province_wide.empty:
        raise ValueError(
            "No province-level records were found. The supplied notebook expects province "
            "labels marked with exactly four leading dots in the Geolocation field."
        )

    long = province_wide.melt(
        id_vars=["Animal Type", "Geolocation"],
        value_vars=YEARS,
        var_name="Year",
        value_name="Production",
    )
    long["Year"] = pd.to_numeric(long["Year"], errors="coerce")
    long["Production"] = pd.to_numeric(long["Production"], errors="coerce")
    return _finish_long(long)


def _long_csv_to_long(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw[["Animal Type", "Geolocation", "Year", "Production"]].copy()
    data["Animal Type"] = data["Animal Type"].astype(str).str.strip().str.strip("'\"")
    data["Geolocation_Raw"] = data["Geolocation"].astype(str).str.strip().str.strip("'\"")
    # If hierarchy prefixes are present, use only the four-dot province-level rows.
    has_province_prefix = data["Geolocation_Raw"].str.match(r"^\.{4}(?!\.)", na=False).any()
    if has_province_prefix:
        data = data[data["Geolocation_Raw"].str.match(r"^\.{4}(?!\.)", na=False)].copy()
    else:
        data = data[~data["Geolocation_Raw"].str.upper().isin({"PHILIPPINES", "NAN", "NONE"})].copy()
        data = data[~data["Geolocation_Raw"].str.startswith("..", na=False)].copy()
    data["Geolocation"] = data["Geolocation_Raw"].str.replace(r"^\.+", "", regex=True).str.strip()
    return _finish_long(data)


def _finish_long(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["Production"] = pd.to_numeric(
        data["Production"].astype(str).str.strip().replace({"..": np.nan, "": np.nan}),
        errors="coerce",
    )
    data = data[data["Animal Type"].isin(ANIMALS)]
    data = data.dropna(subset=["Year", "Production", "Geolocation"])
    data["Year"] = data["Year"].astype(int)
    data = data[data["Year"].between(2010, 2025) & (data["Production"] >= 0)]
    data = data.sort_values(["Geolocation", "Animal Type", "Year"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid provincial livestock observations remain after cleaning.")
    return data[["Year", "Animal Type", "Geolocation", "Production"]]


def _load_from_path(path: Path) -> pd.DataFrame:
    return _read_dataset_bytes(path.read_bytes(), path.name)


def initialise_dataset() -> None:
    """Load an existing local data file or try the notebook's direct GitHub raw URL."""
    preferred = [
        DATA_DIR / "active_dataset.zip",
        DATA_DIR / "active_dataset.arff",
        DATA_DIR / "active_dataset.csv",
        DATA_DIR / "livestock_source.arff",
    ]
    for path in preferred:
        if path.exists():
            try:
                STATE.update(df=_load_from_path(path), source=path.name, error=None)
                return
            except Exception as exc:
                STATE["error"] = str(exc)

    # The file is not bundled in this generated project. Attempt the direct link used by the notebook.
    try:
        request_obj = urllib.request.Request(RAW_DATA_URL, headers={"User-Agent": "LivestockLens/1.0"})
        with urllib.request.urlopen(request_obj, timeout=8) as response:
            payload = response.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise ValueError("The remote dataset exceeds the 100 MB download limit.")
        path = DATA_DIR / "livestock_source.arff"
        path.write_bytes(payload)
        STATE.update(df=_read_dataset_bytes(payload, path.name), source="GitHub raw dataset", error=None)
        return
    except Exception as exc:
        STATE.update(
            df=None,
            source=None,
            error=(
                "Dataset is not loaded. Upload the repository ZIP or the original .csv.arff file "
                f"using the Data Source panel. Automatic download was unsuccessful ({exc})."
            ),
        )


# Load the dataset when the module is imported by a production WSGI server (Gunicorn).
# This is also safe for local execution; the main block below only retries if needed.
initialise_dataset()


def _model_specs() -> dict[str, Any]:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE
        ),
    }


def train_models() -> dict[str, Any]:
    """Replicate the notebook's median-class split and conditional SMOTENC logic."""
    with LOCK:
        data = STATE.get("df")
        if data is None:
            raise ValueError(STATE.get("error") or "Load the dataset before training.")

        median_value = float(data["Production"].median())
        model_df = data.copy()
        model_df["Class"] = (model_df["Production"] >= median_value).astype(int)
        if model_df["Class"].nunique() != 2 or model_df["Class"].value_counts().min() < 3:
            raise ValueError("There are not enough records in both production classes to train and evaluate models.")

        animal_encoder = LabelEncoder()
        province_encoder = LabelEncoder()
        model_df["Animal_Encoded"] = animal_encoder.fit_transform(model_df["Animal Type"])
        model_df["Province_Encoded"] = province_encoder.fit_transform(model_df["Geolocation"])
        feature_columns = ["Year", "Animal_Encoded", "Province_Encoded"]
        X = model_df[feature_columns].copy()
        y = model_df["Class"].copy()

        X_development, X_test, y_development, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_development, y_development, test_size=0.20,
            random_state=RANDOM_STATE, stratify=y_development,
        )

        metrics: list[dict[str, Any]] = []
        augmentation_log: list[dict[str, Any]] = []
        selected_models: dict[str, Any] = {}
        roc_data: dict[str, Any] = {}
        confusion_data: dict[str, Any] = {}
        specs = _model_specs()

        for model_name, model_spec in specs.items():
            baseline = clone(model_spec)
            baseline.fit(X_train, y_train)
            baseline_validation_accuracy = float(accuracy_score(y_val, baseline.predict(X_val)))

            selected_model = baseline
            rows_after_attempt = len(X_train)
            synthetic_created = 0
            validation_after: float | None = None
            triggered = baseline_validation_accuracy < ACCURACY_THRESHOLD
            augmented_selected = False
            decision = "Not needed: validation accuracy is at least 80%."

            if triggered:
                class_counts = y_train.value_counts()
                if int(class_counts.min()) > 1:
                    sampling_strategy = {
                        int(label): int(math.ceil(int(count) * AUGMENT_FACTOR))
                        for label, count in class_counts.items()
                    }
                    try:
                        sampler = SMOTENC(
                            categorical_features=[1, 2],
                            sampling_strategy=sampling_strategy,
                            random_state=RANDOM_STATE,
                            k_neighbors=min(5, int(class_counts.min()) - 1),
                        )
                        X_augmented, y_augmented = sampler.fit_resample(X_train, y_train)
                        rows_after_attempt = len(X_augmented)
                        synthetic_created = len(X_augmented) - len(X_train)
                        augmented_model = clone(model_spec)
                        augmented_model.fit(X_augmented, y_augmented)
                        validation_after = float(accuracy_score(y_val, augmented_model.predict(X_val)))
                        if validation_after > baseline_validation_accuracy:
                            selected_model = augmented_model
                            augmented_selected = True
                            decision = "Augmented model selected: validation accuracy improved."
                        else:
                            decision = "Augmentation tested; baseline retained because validation accuracy did not improve."
                    except Exception as exc:
                        decision = f"Augmentation could not be completed; baseline retained ({type(exc).__name__})."
                else:
                    decision = "Accuracy below 80%, but too few samples to safely generate synthetic rows."

            predicted = selected_model.predict(X_test)
            probabilities = selected_model.predict_proba(X_test)[:, 1]
            auc_value = float(roc_auc_score(y_test, probabilities))
            cm = confusion_matrix(y_test, predicted, labels=[0, 1]).tolist()
            fpr, tpr, _ = roc_curve(y_test, probabilities)
            # Keep the page payload compact while retaining the curve shape.
            step = max(1, int(math.ceil(len(fpr) / 180)))
            roc_points = [
                {"x": round(float(fpr[i]), 6), "y": round(float(tpr[i]), 6)}
                for i in range(0, len(fpr), step)
            ]
            if roc_points[-1] != {"x": 1.0, "y": 1.0}:
                roc_points.append({"x": 1.0, "y": 1.0})

            selected_models[model_name] = selected_model
            confusion_data[model_name] = cm
            roc_data[model_name] = {"auc": auc_value, "points": roc_points}
            metrics.append({
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, predicted)),
                "precision": float(precision_score(y_test, predicted, zero_division=0)),
                "recall": float(recall_score(y_test, predicted, zero_division=0)),
                "f1": float(f1_score(y_test, predicted, zero_division=0)),
                "roc_auc": auc_value,
                "validation_before": baseline_validation_accuracy,
                "validation_after": validation_after,
                "augmentation_triggered": triggered,
                "synthetic_records_created": int(synthetic_created),
                "training_records_used": int(len(X_train) + synthetic_created if augmented_selected else len(X_train)),
                "augmentation_selected": augmented_selected,
                "augmentation_decision": decision,
            })
            augmentation_log.append({
                "model": model_name,
                "validation_before": baseline_validation_accuracy,
                "threshold": ACCURACY_THRESHOLD,
                "triggered": triggered,
                "training_rows_before": int(len(X_train)),
                "training_rows_after_attempt": int(rows_after_attempt),
                "synthetic_rows_created": int(synthetic_created),
                "validation_after": validation_after,
                "augmented_model_selected": augmented_selected,
                "final_training_rows_used": int(len(X_train) + synthetic_created if augmented_selected else len(X_train)),
                "decision": decision,
            })

        metrics.sort(key=lambda row: (row["accuracy"], row["roc_auc"], row["f1"]), reverse=True)
        best_model_name = metrics[0]["model"]

        # Match the notebook: refit each model on real observations only for future class predictions.
        final_models: dict[str, Any] = {}
        X_all = model_df[feature_columns].copy()
        y_all = model_df["Class"].copy()
        for model_name, model_spec in specs.items():
            final_model = clone(model_spec)
            final_model.fit(X_all, y_all)
            final_models[model_name] = final_model

        MODEL_STATE.clear()
        MODEL_STATE.update({
            "ready": True,
            "error": None,
            "median": median_value,
            "animal_encoder": animal_encoder,
            "province_encoder": province_encoder,
            "feature_columns": feature_columns,
            "selected_models": selected_models,
            "final_models": final_models,
            "model_specs": specs,
            "metrics": metrics,
            "augmentation_log": augmentation_log,
            "roc": roc_data,
            "confusion": confusion_data,
            "best_model": best_model_name,
            "train_count": int(len(X_train)),
            "validation_count": int(len(X_val)),
            "test_count": int(len(X_test)),
            "record_count": int(len(model_df)),
            "class_counts": {
                "Low Production": int((model_df["Class"] == 0).sum()),
                "High Production": int((model_df["Class"] == 1).sum()),
            },
        })
        return model_results_payload()


def model_results_payload() -> dict[str, Any]:
    if not MODEL_STATE.get("ready"):
        return {"ready": False, "error": MODEL_STATE.get("error")}
    return {
        "ready": True,
        "median_threshold": MODEL_STATE["median"],
        "best_model": MODEL_STATE["best_model"],
        "record_count": MODEL_STATE["record_count"],
        "train_count": MODEL_STATE["train_count"],
        "validation_count": MODEL_STATE["validation_count"],
        "test_count": MODEL_STATE["test_count"],
        "class_counts": MODEL_STATE["class_counts"],
        "metrics": MODEL_STATE["metrics"],
        "augmentation_log": MODEL_STATE["augmentation_log"],
        "roc": MODEL_STATE["roc"],
        "confusion": MODEL_STATE["confusion"],
    }


def _require_data() -> pd.DataFrame:
    data = STATE.get("df")
    if data is None:
        raise ValueError(STATE.get("error") or "Dataset not loaded.")
    return data


def _filtered_data(args: Any) -> pd.DataFrame:
    data = _require_data()
    animal = args.get("animal", "all")
    province = args.get("province", "all")
    year = args.get("year", "all")
    result = data.copy()
    if animal and animal != "all":
        result = result[result["Animal Type"] == animal]
    if province and province != "all":
        result = result[result["Geolocation"] == province]
    if year and year != "all":
        result = result[result["Year"] == int(year)]
    return result


def _api_error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    data = STATE.get("df")
    return jsonify({
        "loaded": data is not None,
        "source": STATE.get("source"),
        "error": STATE.get("error"),
        "records": int(len(data)) if data is not None else 0,
        "models_ready": bool(MODEL_STATE.get("ready")),
    })


@app.get("/api/options")
def api_options():
    try:
        data = _require_data()
        return jsonify({
            "animals": sorted(data["Animal Type"].dropna().unique().tolist()),
            "provinces": sorted(data["Geolocation"].dropna().unique().tolist()),
            "years": sorted(data["Year"].dropna().astype(int).unique().tolist()),
        })
    except Exception as exc:
        return _api_error(str(exc), 503)


@app.post("/api/upload")
def api_upload():
    upload = request.files.get("dataset")
    if upload is None or not upload.filename:
        return _api_error("Choose the repository ZIP or the original ARFF/CSV file first.")
    filename = Path(upload.filename).name
    ext = Path(filename).suffix.lower()
    if ext not in {".zip", ".arff", ".csv"}:
        return _api_error("Unsupported file type. Upload .zip, .arff, or .csv.")
    payload = upload.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        return _api_error("The selected file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        return _api_error("The file is larger than the 100 MB upload limit.", 413)
    try:
        cleaned = _read_dataset_bytes(payload, filename)
        target = DATA_DIR / f"active_dataset{ext}"
        # Remove older active upload formats so startup always loads the newest file.
        for old in DATA_DIR.glob("active_dataset.*"):
            if old != target:
                old.unlink(missing_ok=True)
        target.write_bytes(payload)
        STATE.update(df=cleaned, source=filename, error=None)
        MODEL_STATE.clear()
        MODEL_STATE.update(ready=False, error=None)
        return jsonify({
            "ok": True,
            "source": filename,
            "records": int(len(cleaned)),
            "provinces": int(cleaned["Geolocation"].nunique()),
            "message": "Dataset loaded and cleaned successfully. Train the models when ready.",
        })
    except Exception as exc:
        return _api_error(f"The file could not be loaded: {exc}")


@app.get("/api/overview")
def api_overview():
    try:
        data = _require_data()
        totals = data.groupby("Animal Type", as_index=False)["Production"].sum()
        totals = totals.sort_values("Production", ascending=False)
        total_production = float(data["Production"].sum())
        latest_year = int(data["Year"].max())
        latest = data[data["Year"] == latest_year]
        return jsonify({
            "kpis": {
                "observations": int(len(data)),
                "provinces": int(data["Geolocation"].nunique()),
                "animal_types": int(data["Animal Type"].nunique()),
                "total_production": total_production,
                "latest_year": latest_year,
                "latest_year_records": int(len(latest)),
            },
            "animals": [
                {"animal": row["Animal Type"], "production": float(row["Production"]),
                 "share": float(row["Production"] / total_production * 100) if total_production else 0}
                for _, row in totals.iterrows()
            ],
            "years": sorted(data["Year"].unique().astype(int).tolist()),
            "source": STATE.get("source"),
        })
    except Exception as exc:
        return _api_error(str(exc), 503)


@app.get("/api/analytics")
def api_analytics():
    try:
        data = _require_data()
        selected = _filtered_data(request.args)
        if selected.empty:
            return _api_error("No records match the selected filters.")

        animal_totals = selected.groupby("Animal Type", as_index=False)["Production"].sum()
        animal_totals = animal_totals.sort_values("Production", ascending=False)
        total = float(animal_totals["Production"].sum())
        animal_totals["Share"] = animal_totals["Production"] / total * 100 if total else 0

        annual_base = data.copy()
        animal = request.args.get("animal", "all")
        province = request.args.get("province", "all")
        if animal != "all":
            annual_base = annual_base[annual_base["Animal Type"] == animal]
        if province != "all":
            annual_base = annual_base[annual_base["Geolocation"] == province]
        annual = annual_base.groupby(["Year", "Animal Type"], as_index=False)["Production"].sum()
        trend = [
            {"year": int(r["Year"]), "animal": r["Animal Type"], "production": float(r["Production"])}
            for _, r in annual.sort_values("Year").iterrows()
        ]

        table_year = request.args.get("year", "all")
        province_base = data.copy()
        if animal != "all":
            province_base = province_base[province_base["Animal Type"] == animal]
        if table_year != "all":
            province_base = province_base[province_base["Year"] == int(table_year)]
        else:
            province_base = province_base[province_base["Year"] == int(data["Year"].max())]
        top_provinces = province_base.groupby("Geolocation", as_index=False)["Production"].sum()
        top_provinces = top_provinces.nlargest(10, "Production").sort_values("Production")

        desc = selected.groupby("Animal Type")["Production"].agg(
            Count="count", Mean="mean", Median="median", Std_Dev="std", Minimum="min", Maximum="max"
        ).reset_index()
        distribution_year = data[data["Year"] == int(data["Year"].max())]
        distribution = [
            {"animal": animal_name, "production": float(part["Production"].median()),
             "min": float(part["Production"].min()), "max": float(part["Production"].max())}
            for animal_name, part in distribution_year.groupby("Animal Type")
        ]

        recent = selected.sort_values(["Year", "Production"], ascending=[False, False]).head(12)
        return jsonify({
            "summary": {"records": int(len(selected)), "production": total,
                        "median": float(selected["Production"].median()),
                        "first_year": int(selected["Year"].min()),
                        "last_year": int(selected["Year"].max())},
            "animal_totals": [
                {"animal": r["Animal Type"], "production": float(r["Production"]), "share": float(r["Share"])}
                for _, r in animal_totals.iterrows()
            ],
            "annual_trend": trend,
            "top_provinces": [
                {"province": r["Geolocation"], "production": float(r["Production"])}
                for _, r in top_provinces.iterrows()
            ],
            "descriptive": [
                {key: (_json_number(value) if key != "Animal Type" and key != "Count" else (int(value) if key == "Count" else value))
                 for key, value in r.items()}
                for _, r in desc.iterrows()
            ],
            "distribution_2025": distribution,
            "recent_records": [
                {"year": int(r["Year"]), "animal": r["Animal Type"],
                 "province": r["Geolocation"], "production": float(r["Production"])}
                for _, r in recent.iterrows()
            ],
            "unit": "Metric tons",
        })
    except Exception as exc:
        return _api_error(str(exc), 400)


@app.get("/api/records")
def api_records():
    try:
        data = _filtered_data(request.args)
        query = request.args.get("q", "").strip().lower()
        if query:
            mask = (
                data["Animal Type"].str.lower().str.contains(query, na=False)
                | data["Geolocation"].str.lower().str.contains(query, na=False)
                | data["Year"].astype(str).str.contains(query, na=False)
            )
            data = data[mask]
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(10, int(request.args.get("page_size", 25))))
        total = len(data)
        data = data.sort_values(["Year", "Geolocation", "Animal Type"], ascending=[False, True, True])
        chunk = data.iloc[(page - 1) * page_size:page * page_size]
        return jsonify({
            "total": int(total), "page": page, "page_size": page_size,
            "pages": int(math.ceil(total / page_size)) if total else 0,
            "rows": [
                {"year": int(r["Year"]), "animal": r["Animal Type"],
                 "province": r["Geolocation"], "production": float(r["Production"])}
                for _, r in chunk.iterrows()
            ],
        })
    except Exception as exc:
        return _api_error(str(exc))


@app.post("/api/train")
def api_train():
    try:
        result = train_models()
        return jsonify({"ok": True, **result})
    except Exception as exc:
        MODEL_STATE["error"] = str(exc)
        return _api_error(f"Model training failed: {exc}", 500)


@app.get("/api/models")
def api_models():
    return jsonify(model_results_payload())


@app.post("/api/predict")
def api_predict():
    try:
        if not MODEL_STATE.get("ready"):
            raise ValueError("Train the classification models before requesting predictions.")
        body = request.get_json(silent=True) or request.form
        animal = str(body.get("animal_type", "")).strip()
        province = str(body.get("province", "")).strip()
        year = int(body.get("year", 2026))
        if animal not in ANIMALS:
            raise ValueError("Select one of the available animal types.")
        animal_encoder: LabelEncoder = MODEL_STATE["animal_encoder"]
        province_encoder: LabelEncoder = MODEL_STATE["province_encoder"]
        try:
            animal_encoded = int(animal_encoder.transform([animal])[0])
            province_encoded = int(province_encoder.transform([province])[0])
        except ValueError:
            raise ValueError("The selected animal or province is not included in the loaded dataset.")
        input_row = pd.DataFrame(
            [[year, animal_encoded, province_encoded]],
            columns=MODEL_STATE["feature_columns"],
        )
        results = []
        for model_name, model in MODEL_STATE["final_models"].items():
            predicted = int(model.predict(input_row)[0])
            probability_high = float(model.predict_proba(input_row)[0][1])
            results.append({
                "model": model_name,
                "class": "High Production" if predicted == 1 else "Low Production",
                "probability_high": probability_high,
                "confidence": probability_high if predicted == 1 else 1 - probability_high,
            })
        best_name = MODEL_STATE["best_model"]
        best_result = next(item for item in results if item["model"] == best_name)
        return jsonify({
            "ok": True, "animal_type": animal, "province": province, "year": year,
            "median_threshold": MODEL_STATE["median"], "best_model": best_name,
            "best_prediction": best_result["class"], "results": results,
            "notice": "This is a High/Low category prediction, not a numerical production-volume forecast. It is based on the model trained from the 2010–2025 records.",
        })
    except Exception as exc:
        return _api_error(str(exc))


@app.get("/api/export/<kind>.csv")
def api_export(kind: str):
    try:
        data = _require_data()
        if kind == "cleaned-data":
            output = data.copy()
            filename = "cleaned_provincial_livestock_2010_2025.csv"
        elif kind == "model-comparison":
            if not MODEL_STATE.get("ready"):
                raise ValueError("Train the models before exporting their comparison.")
            output = pd.DataFrame(MODEL_STATE["metrics"])
            filename = "livestock_model_comparison.csv"
        elif kind == "predictions-2026":
            if not MODEL_STATE.get("ready"):
                raise ValueError("Train the models before exporting predictions.")
            latest = data[data["Year"] == 2025].copy()
            if latest.empty:
                raise ValueError("There are no valid 2025 observations for generating 2026 predictions.")
            animal_encoder = MODEL_STATE["animal_encoder"]
            province_encoder = MODEL_STATE["province_encoder"]
            latest["Animal_Encoded"] = animal_encoder.transform(latest["Animal Type"])
            latest["Province_Encoded"] = province_encoder.transform(latest["Geolocation"])
            X_future = latest[["Year", "Animal_Encoded", "Province_Encoded"]].copy()
            X_future["Year"] = 2026
            output = latest[["Geolocation", "Animal Type", "Year", "Production"]].rename(
                columns={"Geolocation": "Province", "Year": "Current Year", "Production": "Production in 2025"}
            )
            for model_name, model in MODEL_STATE["final_models"].items():
                predicted = model.predict(X_future)
                probability = model.predict_proba(X_future)[:, 1]
                output[f"{model_name} - Predicted 2026 Class"] = np.where(predicted == 1, "High Production", "Low Production")
                output[f"{model_name} - Probability High"] = probability
            filename = "predicted_2026_high_low_by_province_animal.csv"
        else:
            return _api_error("Unknown export type.", 404)

        buffer = io.BytesIO(output.to_csv(index=False).encode("utf-8-sig"))
        buffer.seek(0)
        return send_file(buffer, mimetype="text/csv", as_attachment=True, download_name=filename)
    except Exception as exc:
        return _api_error(str(exc))


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"ok": False, "error": "Upload is larger than the 100 MB limit."}), 413


if __name__ == "__main__":
    if STATE.get("df") is None:
        initialise_dataset()
    print("\nLivestockLens is starting.")
    print("Open http://127.0.0.1:5000 in your browser.")
    if STATE.get("df") is None:
        print(STATE.get("error"))
    else:
        print(f"Loaded {len(STATE['df']):,} cleaned province-year observations from {STATE['source']}.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
