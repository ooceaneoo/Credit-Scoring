from pydantic import BaseModel
from typing import Dict, Any


class PredictionRequest(BaseModel):
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: int
    score: float
    threshold: float
    inference_time_ms: float