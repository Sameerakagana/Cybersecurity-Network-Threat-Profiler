
import os
import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse


app = FastAPI(
    title="Cybersecurity Network Threat & Intrusion Profiler",
    version="1.0.0"
)


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")


# --------------------------------------------------
# MODEL FILES
# --------------------------------------------------

REQUIRED_MODELS = [
    "classifier.pkl",
    "scaler.pkl",
    "anomaly_model.pkl",
    "encoders.pkl",
    "target_encoder.pkl",
    "feature_columns.pkl",
]


models_ready = all(
    os.path.exists(os.path.join(MODEL_DIR, f))
    for f in REQUIRED_MODELS
)


classifier = None
scaler = None
anomaly_model = None
encoders = None
target_encoder = None
feature_columns = None


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

if models_ready:

    classifier = joblib.load(
        os.path.join(MODEL_DIR, "classifier.pkl")
    )

    scaler = joblib.load(
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    anomaly_model = joblib.load(
        os.path.join(MODEL_DIR, "anomaly_model.pkl")
    )

    encoders = joblib.load(
        os.path.join(MODEL_DIR, "encoders.pkl")
    )

    target_encoder = joblib.load(
        os.path.join(MODEL_DIR, "target_encoder.pkl")
    )

    feature_columns = joblib.load(
        os.path.join(MODEL_DIR, "feature_columns.pkl")
    )


# --------------------------------------------------
# FRONTEND
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home():

    index_path = os.path.join(BASE_DIR, "index.html")

    if not os.path.exists(index_path):
        return HTMLResponse(
            "<h1>index.html not found</h1>",
            status_code=404
        )

    with open(index_path, "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())


@app.get("/style.css")
async def style():

    return FileResponse(
        os.path.join(BASE_DIR, "style.css"),
        media_type="text/css"
    )


@app.get("/script.js")
async def script():

    return FileResponse(
        os.path.join(BASE_DIR, "script.js"),
        media_type="application/javascript"
    )


# --------------------------------------------------
# API STATUS
# --------------------------------------------------

@app.get("/api/status")
async def status():

    return {
        "status": "online",
        "models_ready": models_ready,
        "classifier": "Random Forest",
        "anomaly_detector": "Isolation Forest"
    }


# --------------------------------------------------
# ANALYZE NETWORK TRAFFIC
# --------------------------------------------------

@app.post("/api/analyze")
async def analyze(data: dict):

    if not models_ready:

        return {
            "error": (
                "Models are not trained yet. "
                "Put data/NSL_KDD.csv in place and run: "
                "python train_model.py"
            )
        }

    try:

        # Create a row containing all model features
        # with default value 0.

        row = {
            col: 0
            for col in feature_columns
        }


        # Add values received from the frontend

        for key, value in data.items():

            if key in row:
                row[key] = value


        # Convert input into DataFrame

        frame = pd.DataFrame(
            [row],
            columns=feature_columns
        )


        # Encode categorical columns

        for col, encoder in encoders.items():

            value = str(
                frame[col].iloc[0]
            )

            try:

                frame[col] = encoder.transform(
                    [value]
                )

            except ValueError:

                frame[col] = 0


        # Scale input

        X = scaler.transform(frame)


        # Random Forest classification

        pred = classifier.predict(X)[0]

        label = target_encoder.inverse_transform(
            [pred]
        )[0]


        # Isolation Forest anomaly detection

        anomaly_pred = anomaly_model.predict(X)[0]

        anomaly_score = float(
            anomaly_model.decision_function(X)[0]
        )


        if anomaly_pred == -1:

            anomaly_status = "Anomalous"

        else:

            anomaly_status = "Normal"


        # Determine whether classification is normal

        normal_labels = {
            "normal",
            "normal."
        }

        is_normal = (
            str(label).strip().lower()
            in normal_labels
        )


        # Determine risk

        if anomaly_status == "Anomalous":

            risk = "HIGH"

        elif not is_normal:

            risk = "HIGH"

        else:

            risk = "LOW"


        # Return result to frontend

        return {

            "classification": str(label),

            "anomaly_status": anomaly_status,

            "anomaly_score": round(
                anomaly_score,
                4
            ),

            "risk": risk

        }


    except Exception as exc:

        return {
            "error": str(exc)
        }