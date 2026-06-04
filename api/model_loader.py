import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "optimized_lightgbm.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"

# Chargement au démarrage
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)