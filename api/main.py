from fastapi import FastAPI, HTTPException
from api.schemas import PredictionRequest, PredictionResponse
from api.model_loader import model, feature_columns
import json
import numpy as np
import pandas as pd
import psutil
import time
from datetime import datetime, timezone
from pathlib import Path


app = FastAPI(
    title="Credit Scoring API",
    description="API de prédiction du risque de défaut client",
    version="1.0"
)

THRESHOLD = 0.5
LOG_PATH = Path("monitoring/logs/production_logs.jsonl")


def get_system_metrics():
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": psutil.virtual_memory().percent
    }


def log_prediction(
    features,
    score=None,
    prediction=None,
    inference_time_ms=None,
    success=True,
    status_code=200,
    error_message=None
):
    """
    Enregistre chaque appel à l'API dans un fichier JSONL.

    Les logs contiennent les inputs, outputs, temps d'exécution, statut de la requête et métriques système.
    """

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    system_metrics = get_system_metrics()

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": bool(success),
        "status_code": int(status_code),
        "error_message": error_message,
        "score": float(score) if score is not None else None,
        "prediction": int(prediction) if prediction is not None else None,
        "inference_time_ms": float(inference_time_ms) if inference_time_ms is not None else None,
        "cpu_percent": float(system_metrics["cpu_percent"]),
        "memory_percent": float(system_metrics["memory_percent"]),
        "features": features
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


@app.get("/")
def root():
    return {"message": "Credit Scoring API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = time.time()
    input_data = request.features

    try:
        df = pd.DataFrame([input_data])

        missing_cols = set(feature_columns) - set(df.columns)

        if missing_cols:
            inference_time_ms = round((time.time() - start_time) * 1000, 2)

            log_prediction(
                features=input_data,
                inference_time_ms=inference_time_ms,
                success=False,
                status_code=400,
                error_message=f"Colonnes manquantes : {list(missing_cols)}"
            )

            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes : {list(missing_cols)}"
            )

        df = df[feature_columns]
        # Conversion en NumPy pour limiter les conversions internes de LightGBM
        input_array = df.to_numpy(dtype=np.float32)

        score = model.predict_proba(input_array)[0][1]
        prediction = int(score >= THRESHOLD)

        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        log_prediction(
            features=input_data,
            score=score,
            prediction=prediction,
            inference_time_ms=inference_time_ms,
            success=True,
            status_code=200,
            error_message=None
        )

        return PredictionResponse(
            prediction=prediction,
            score=float(score),
            threshold=THRESHOLD,
            inference_time_ms=inference_time_ms
        )

    except HTTPException as e:
        raise e

    except Exception as e:
        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        log_prediction(
            features=input_data,
            inference_time_ms=inference_time_ms,
            success=False,
            status_code=500,
            error_message=str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )