import json
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Chargement du jeu de test
X_test = joblib.load(PROJECT_ROOT / "data" / "X_test.pkl")

# Sélection aléatoire d'un client
random_client = X_test.sample(1)

print(f"Client sélectionné : {random_client.index[0]}")

sample = {
    "features": random_client.iloc[0].to_dict()
}

output_path = PROJECT_ROOT / "sample_input.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=2)

print(f"Exemple généré : {output_path}")