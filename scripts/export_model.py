import joblib
from pathlib import Path
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_train = joblib.load(PROJECT_ROOT / "data" / "X_train.pkl")
y_train = joblib.load(PROJECT_ROOT / "data" / "y_train.pkl")

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

model.fit(X_train, y_train)

models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(model, models_dir / "optimized_lightgbm.pkl")
joblib.dump(list(X_train.columns), models_dir / "feature_columns.pkl")

print("Modèle exporté dans models/optimized_lightgbm.pkl")
print("Colonnes exportées dans models/feature_columns.pkl")