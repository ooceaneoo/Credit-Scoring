# Credit Scoring - Machine Learning & MLOps

## Présentation

Ce projet a pour objectif de développer une solution complète de **scoring crédit** capable d'estimer le risque de défaut d'un client lors d'une demande de prêt. Au-delà de la construction d'un modèle performant, le projet couvre l'ensemble du cycle de vie d'un modèle de Machine Learning : développement, déploiement, monitoring, détection de dérive et optimisation post-déploiement.

## Objectifs

-   Analyse exploratoire des données et feature engineering
-   Comparaison de plusieurs modèles de classification
-   Optimisation du modèle (Optuna, seuil de décision et métrique
    métier)
-   Suivi des expérimentations avec MLflow et Model Registry
-   Déploiement du modèle sous forme d'une API REST avec FastAPI
-   Conteneurisation avec Docker et pipeline CI/CD
-   Monitoring des prédictions et des performances de l'API
-   Détection du data drift avec le PSI et Evidently AI
-   Développement d'un dashboard Streamlit de supervision
-   Analyse et optimisation des performances post-déploiement

------------------------------------------------------------------------

## Problématique métier

Dans le contexte du crédit bancaire :

-   **Faux négatif (FN)** : un client risqué est accepté → perte
    financière importante.
-   **Faux positif (FP)** : un bon client est refusé → manque à gagner
    plus limité.

Une métrique métier personnalisée est utilisée :

`Coût = 10 × Faux négatifs + Faux positifs`

L'objectif principal est de **minimiser ce coût métier**.

------------------------------------------------------------------------

## Pipeline Machine Learning

Le pipeline comprend :

-   préparation des données
-   feature engineering
-   comparaison de plusieurs modèles de classification (DummyClassifier, Régression Logistique, Random Forest, LightGBM et XGBoost)
-   validation croisée stratifiée
-   optimisation des hyperparamètres avec Optuna
-   optimisation du seuil de décision selon la métrique métier
-   suivi des expérimentations avec MLflow
-   export du modèle final

Le modèle retenu est un **LightGBM optimisé**, sélectionné pour son bon compromis entre performance statistique et coût métier. Les hyperparamètres ont été optimisés avec **Optuna** et le seuil de décision a été ajusté afin de minimiser le coût métier.

## Modélisation et optimisation

Les modèles ont été évalués avec les métriques suivantes :

-   Accuracy
-   ROC-AUC
-   Recall
-   Precision
-   F1-score
-   Coût métier

Une validation croisée stratifiée à 5 folds a été utilisée afin
d'obtenir une estimation robuste des performances.

Le modèle finalement retenu est un **LightGBM optimisé**, sélectionné
pour son coût métier minimal, son recall élevé, son bon ROC-AUC et sa
stabilité.

## Suivi des expérimentations avec MLflow

MLflow a été utilisé pour suivre :

-   les paramètres
-   les métriques
-   les tags
-   les graphiques
-   la comparaison des runs
-   le Model Registry

Les principaux runs correspondent au benchmark des modèles, aux validations croisées, au modèle champion et au modèle optimisé final.

------------------------------------------------------------------------

## Déploiement

Une API REST a été développée avec **FastAPI** afin d'exposer le modèle via un endpoint `/predict`.

Fonctionnalités :

-   chargement automatique du modèle
-   validation des données avec Pydantic
-   conteneurisation Docker
-   pipeline GitHub Actions
-   tests automatisés avec Pytest

------------------------------------------------------------------------

## Monitoring de production

Chaque appel API est enregistré dans un fichier JSONL contenant :

-   horodatage
-   variables d'entrée
-   score
-   décision
-   temps d'inférence
-   statut HTTP
-   succès/échec
-   utilisation CPU
-   utilisation mémoire

Ces données servent au suivi du comportement du modèle, à la détection d'anomalies et à l'analyse des performances.

------------------------------------------------------------------------

## Détection du data drift

Deux approches complémentaires sont utilisées :

-   **Population Stability Index (PSI)** pour comparer les distributions entre données d'entraînement et de production.
-   **Evidently AI** pour générer un rapport HTML détaillé accessible depuis le dashboard.

------------------------------------------------------------------------

## Dashboard Streamlit

Le dashboard centralise les principaux indicateurs de monitoring :

-   prédictions réalisées
-   distribution des scores
-   répartition des décisions
-   performances de l'API
-   temps d'inférence
-   utilisation CPU et mémoire
-   erreurs HTTP
-   analyses PSI
-   rapport Evidently
-   recommandations de surveillance

------------------------------------------------------------------------

## Optimisation post-déploiement

Les performances ont été étudiées après le déploiement à l'aide :

-   d'un benchmark du modèle
-   d'un benchmark de l'API
-   d'une comparaison du temps modèle/API
-   d'un profiling avec cProfile
-   d'une comparaison DataFrame pandas/tableau NumPy

Le profiling a montré qu'une partie du temps d'exécution provenait des conversions effectuées avant `predict_proba`.

L'optimisation retenue consiste à convertir les données d'entrée au format **NumPy** avant la prédiction. Cette modification réduit le temps moyen d'inférence d'environ 25 % sans modifier les prédictions.

------------------------------------------------------------------------

## Technologies

Python, Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, Optuna, MLflow, FastAPI, Streamlit, Evidently AI, Docker, GitHub Actions, Pytest, psutil et Matplotlib.

------------------------------------------------------------------------

## Architecture du projet

Le projet est organisé autour de quatre composantes principales :

-   **Machine Learning** : préparation des données, entraînement et optimisation.
-   **API** : exposition du modèle via FastAPI et Docker.
-   **Monitoring** : collecte des logs, analyse du drift et suivi des performances.
-   **Visualisation** : dashboard Streamlit.

------------------------------------------------------------------------

## Structure du projet

``` text
Credit-Scoring/
│
├── api/                              → API REST FastAPI permettant de servir le modèle en production
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
│
├── data/                             → Jeux de données bruts et jeux de données prétraités
│   ├── application_train.csv
│   ├── application_test.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── POS_CASH_balance.csv
│   ├── installments_payments.csv
│   ├── credit_card_balance.csv
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   └── y_test.pkl
│
├── models/                           → Modèle final exporté et variables attendues par l'API
│   ├── optimized_lightgbm.pkl
│   └── feature_columns.pkl
│
├── monitoring/                       → Monitoring, analyse du drift et optimisation post-déploiement
│   ├── dashboard.py
│   ├── drift_analysis.ipynb
│   ├── performance_optimization.ipynb
│   └── logs/
│       ├── .gitkeep
│       └── production_logs.jsonl
│
├── notebooks/                        → Développement complet du pipeline Machine Learning
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_feature_engineering.ipynb
│   ├── 04_modeling_mlflow.ipynb
│   └── 05_model_optimization.ipynb
│
├── reports/                          → Rapports et figures générés au cours du projet
│   ├── evidently_data_drift_report.html
│   └── figures/
│       ├── mlflow/
│       ├── modeling/
│       └── optimization/
│
├── scripts/                          → Scripts utilitaires
│   ├── export_model.py
│   └── generate_sample_input.py
│
├── tests/                            → Tests automatisés de l'API
│   └── test_api.py
│
├── .github/                          → Pipeline d'intégration continue GitHub Actions
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile                        → Conteneurisation de l'API
├── requirements.txt                  → Dépendances du projet
├── requirements-api.txt              → Dépendances spécifiques à l'API
├── sample_input.json                 → Exemple de requête pour tester l'API
├── mlflow.db                         → Base de données locale MLflow
├── .dockerignore
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

## Installation

``` bash
pip install -r requirements.txt
```

## Lancer l'API

``` bash
uvicorn api.main:app --reload
```

## Lancer le dashboard

``` bash
streamlit run monitoring/dashboard.py
```

## Lancer MLflow

``` bash
mlflow ui
```

## Construire Docker

```bash
docker build -t credit-scoring-api .
```

```bash
docker run -p 8000:8000 credit-scoring-api
```

## Tests

``` bash
python -m pytest tests/test_api.py -v
```

------------------------------------------------------------------------

## Résultats

Le projet couvre l'ensemble du cycle de vie d'un modèle de Machine Learning :

-   développement
-   optimisation métier
-   suivi des expérimentations
-   déploiement
-   monitoring
-   détection du data drift
-   optimisation post-déploiement

Il constitue une **preuve de concept (PoC) MLOps** pour un système de scoring crédit.

------------------------------------------------------------------------

## Perspectives

-   stockage des logs dans une base de données
-   monitoring temps réel avec Grafana
-   détection automatique des dérives
-   réentraînement automatique
-   optimisation avec ONNX Runtime
-   déploiement sur une infrastructure cloud
