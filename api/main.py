from fastapi import FastAPI, HTTPException
from api.schemas import PredictionRequest, PredictionResponse
from api.model_loader import model, feature_columns

import json
import numpy as np
import pandas as pd
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


def log_prediction(features, score, prediction, inference_time_ms):
    """
    Enregistre chaque prédiction dans un fichier JSONL.

    Chaque ligne du fichier correspond à un appel API.
    Ces logs serviront au monitoring et à l'analyse du data drift.
    """
    
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score": float(score),
        "prediction": int(prediction),
        "inference_time_ms": float(inference_time_ms),
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

    try:
        input_data = request.features

        df = pd.DataFrame([input_data])

        missing_cols = set(feature_columns) - set(df.columns)

        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes : {list(missing_cols)}"
            )

        df = df[feature_columns]

        input_array = df.to_numpy(dtype=np.float32)

        score = model.predict_proba(input_array)[0][1]
        prediction = int(score >= THRESHOLD)

        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        log_prediction(
            features=input_data,
            score=score,
            prediction=prediction,
            inference_time_ms=inference_time_ms
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
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )