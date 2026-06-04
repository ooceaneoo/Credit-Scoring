import json
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_test = joblib.load(PROJECT_ROOT / "data" / "X_test.pkl")

sample = {
    "features": X_test.iloc[0].to_dict()
}

output_path = PROJECT_ROOT / "sample_input.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=2)

print(f"Exemple généré : {output_path}")