from fastapi.testclient import TestClient

from api.main import app
from api.model_loader import feature_columns

# Client permettant de simuler des requêtes HTTP sur l'API
client = TestClient(app)


def generate_valid_payload():
    """
    Génère un payload valide contenant toutes les variables attendues par le modèle.
    """

    features = {}

    for col in feature_columns:
        features[col] = 0

    return {"features": features}


def test_root():
    """
    Vérifie que la route racine répond correctement.
    """

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Credit Scoring API is running"


def test_health():
    """
    Vérifie que l'endpoint de santé indique que le service est opérationnel.
    """

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_input():
    """
    Vérifie qu'une prédiction est produite lorsqu'un payload valide est envoyé.
    """

    payload = generate_valid_payload()

    response = client.post(
        "/predict",
        json=payload
    )

    data = response.json()

    assert response.status_code == 200

    assert "prediction" in data
    assert "score" in data
    assert "threshold" in data
    assert "inference_time_ms" in data


def test_predict_missing_features():
    """
    Vérifie que l'API refuse une requête lorsque des variables obligatoires sont absentes.
    """

    payload = {
        "features": {
            "AGE": 35
        }
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 500


def test_predict_wrong_type():
    """
    Vérifie que l'API gère correctement un type de données invalide.
    """

    payload = {
        "features": "ceci_n_est_pas_un_dictionnaire"
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code in [422, 500]


def test_predict_empty_payload():
    """
    Vérifie qu'une requête vide est rejetée.
    """

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code in [422, 500]