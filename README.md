# Credit Scoring - Projet Machine Learning & MLOps

## Présentation du projet

L’objectif de ce projet est de construire un modèle de **scoring crédit** capable de prédire le risque de défaut d’un client lors d’une demande de prêt.

Au-delà de la simple performance statistique, le projet intègre une **vision métier**, en cherchant à minimiser les pertes financières liées à de mauvaises décisions d’octroi de crédit.

Un pipeline complet de Machine Learning a été mis en place :

- Analyse exploratoire des données
- Préparation des données et feature engineering
- Comparaison de plusieurs modèles de classification
- Création d’une métrique métier personnalisée
- Validation croisée
- Optimisation des hyperparamètres avec Optuna
- Optimisation du seuil de décision
- Suivi des expérimentations avec MLflow
- Enregistrement des modèles dans le Model Registry

## Problématique métier

Dans un contexte bancaire :

- **Faux négatif (FN)** : client risqué accepté → perte financière importante
- **Faux positif (FP)** : bon client refusé → manque à gagner plus limité

Une métrique métier a donc été définie :

```python
Coût = 10 × Faux Négatifs + 1 × Faux Positifs
```

L’objectif principal est de **minimiser ce coût métier**, plutôt que de maximiser uniquement l’accuracy.

## Technologies utilisées

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- Optuna
- MLflow
- Matplotlib
- Seaborn
- Git / GitHub

## Structure du projet

```bash
Credit-Scoring/
│── data/
│── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_feature_engineering.ipynb
│   ├── 04_modeling_mlflow.ipynb
│   └── 05_model_optimization.ipynb
│
│── reports/
│   └── figures/
│       ├── modeling/
│       ├── optimization/
│       └── mlflow/
│
│── README.md
│── requirements.txt
│── .gitignore
```

## Phase de modélisation

Plusieurs modèles ont été comparés :

- DummyClassifier
- Régression Logistique
- Random Forest
- LightGBM
- XGBoost

Les métriques utilisées :

- Accuracy
- ROC-AUC
- Recall
- Precision
- F1-score
- Coût métier

Une **validation croisée stratifiée à 5 folds** a été utilisée pour comparer les modèles de manière robuste.

## Meilleur modèle initial

Le meilleur modèle retenu lors de la première phase est :

### LightGBM

Ce choix s’explique par :

- le coût métier le plus faible ;
- un recall élevé ;
- un bon ROC-AUC ;
- une bonne stabilité en validation croisée.

## Phase d’optimisation

Le modèle LightGBM a ensuite été amélioré grâce à :

### Optimisation des hyperparamètres

Utilisation d’**Optuna** pour rechercher automatiquement les meilleurs réglages.

### Optimisation du seuil de décision

Le seuil standard de classification, fixé à 0.50, a été remplacé par un seuil minimisant le coût métier.

## Suivi des expérimentations avec MLflow

Toutes les expérimentations ont été tracées avec MLflow :

- paramètres des modèles ;
- métriques ;
- tags ;
- graphiques ;
- comparaison des runs ;
- Model Registry.

Exemples de runs enregistrés :

- benchmark des modèles ;
- validation croisée ;
- modèle champion ;
- modèle optimisé final.

## Résultats clés

Ce projet montre qu’un bon modèle de Machine Learning ne doit pas uniquement être performant statistiquement, mais également **aligné avec les enjeux métier**.

L’accuracy seule peut être trompeuse sur un dataset déséquilibré.

L’utilisation :

- d’une métrique métier ;
- d’un ajustement du seuil ;
- d’un suivi des expériences ;
- d’une optimisation structurée ;

permet d’obtenir une solution plus réaliste et exploitable en entreprise.

## Lancer le projet

### Installer les dépendances

```bash
pip install -r requirements.txt
```

### Ouvrir les notebooks

```bash
jupyter notebook
```

### Lancer MLflow UI

```bash
mlflow ui
```

Puis ouvrir :

```text
http://127.0.0.1:5000
```
