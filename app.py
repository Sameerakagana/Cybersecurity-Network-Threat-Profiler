import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(
    title="Cybersecurity Network Threat & Intrusion Profiler",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL_DIR = "models"
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

classifier = scaler = anomaly_model = encoders = target_encoder = feature_columns = None

if models_ready:
    classifier = joblib.load(f"{MODEL_DIR}/classifier.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    anomaly_model = joblib.load(f"{MODEL_DIR}/anomaly_model.pkl")
    encoders = joblib.load(f"{MODEL_DIR}/encoders.pkl")
    target_encoder = joblib.load(f"{MODEL_DIR}/target_encoder.pkl")
    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"models_ready": models_ready}
    )

@app.get("/api/status")
async def status():
    return {
        "status": "online",
        "models_ready": models_ready,
        "classifier": "Random Forest",
        "anomaly_detector": "Isolation Forest"
    }

@app.post("/api/analyze")
async def analyze(data: dict):
    if not models_ready:
        return {
            "error": "Models are not trained yet. Put data/NSL_KDD.csv in place and run: python train_model.py"
        }

    try:
        # The UI sends a small set of common NSL-KDD fields.
        # Any remaining model features are filled with 0 so the demo
        # can still run without exposing every NSL-KDD field on screen.
        row = {col: 0 for col in feature_columns}

        for key, value in data.items():
            if key in row:
                row[key] = value

        frame = pd.DataFrame([row], columns=feature_columns)

        for col, encoder in encoders.items():
            value = str(frame[col].iloc[0])
            try:
                frame[col] = encoder.transform([value])
            except ValueError:
                frame[col] = 0

        X = scaler.transform(frame)

        pred = classifier.predict(X)[0]
        label = target_encoder.inverse_transform([pred])[0]

        anomaly_pred = anomaly_model.predict(X)[0]
        anomaly_score = float(anomaly_model.decision_function(X)[0])

        anomaly_status = "Anomalous" if anomaly_pred == -1 else "Normal"
        normal_labels = {"normal", "normal."}
        is_normal = str(label).strip().lower() in normal_labels

        if anomaly_status == "Anomalous":
            risk = "HIGH"
        elif not is_normal:
            risk = "HIGH"
        else:
            risk = "LOW"

        return {
            "classification": str(label),
            "anomaly_status": anomaly_status,
            "anomaly_score": round(anomaly_score, 4),
            "risk": risk
        }

    except Exception as exc:
        return {"error": str(exc)}
