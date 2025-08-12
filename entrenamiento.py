"""Entrenamiento del modelo para predecir el método de victoria.

Este script carga las estadísticas históricas de peleas, selecciona las
características numéricas definidas en ``columnas_X`` (métricas previas y
atributos físicos) y entrena un modelo de stacking para predecir el
``Method`` de una pelea. La validación se realiza de forma cronológica,
utilizando las peleas más antiguas para entrenamiento y las más recientes
para prueba.  Las métricas del modelo se guardan en ``models``.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42

DATA_DIR = os.path.abspath("data")
MODELS_DIR = os.path.abspath("models")
os.makedirs(MODELS_DIR, exist_ok=True)
fight_stats = pd.read_csv(os.path.join(DATA_DIR, "fight_stats.csv"))
columnas_X = (
    pd.read_csv(os.path.join(DATA_DIR, "columnas_X.csv"), header=None)
    .squeeze()
    .tolist()
)

# Características exclusivamente numéricas y de atributos físicos
X = fight_stats[columnas_X]
y_method = fight_stats["Method"]

# Separación cronológica: primeras peleas para entrenamiento,
# últimas peleas para prueba
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train_method, y_test_method = y_method.iloc[:split], y_method.iloc[split:]

models = {
    "XGBoost": (
        XGBClassifier(random_state=RANDOM_STATE),
        {
            "learning_rate": [0.01, 0.1],
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
        },
    ),
    "LightGBM": (
        LGBMClassifier(random_state=RANDOM_STATE),
        {
            "learning_rate": [0.01, 0.1],
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
        },
    ),
    "CatBoost": (
        CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
        {"learning_rate": [0.01, 0.1], "iterations": [100, 200]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 200],
            "max_depth": [10, 20],
            "min_samples_split": [2, 5],
        },
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "learning_rate": [0.01, 0.1],
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
        },
    ),
    "LogisticRegression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    ),
    "SVC": (
        SVC(probability=True, random_state=RANDOM_STATE),
        {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    ),
}


def entrenar(y_train, y_test):
    """Entrena modelos base y un stacking final para el método de victoria."""

    mejores = []
    print("\n📊 Iniciando entrenamiento para METHOD\n")
    for nombre, (modelo, params) in models.items():
        print(f"⚙️ GridSearch para {nombre}...")
        grid = GridSearchCV(
            modelo, params, cv=3, scoring="accuracy", verbose=1, n_jobs=-1
        )
        grid.fit(X_train, y_train)
        joblib.dump(
            grid.best_estimator_,
            os.path.join(MODELS_DIR, f"{nombre.lower()}_method.pkl"),
        )
        mejores.append((nombre, grid.best_estimator_))
        print(f"✅ {nombre} mejores parámetros: {grid.best_params_}")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    stacking = StackingClassifier(
        estimators=mejores,
        final_estimator=LogisticRegression(random_state=RANDOM_STATE),
        cv=cv,
        n_jobs=-1,
    )
    stacking.fit(X_train, y_train)
    joblib.dump(stacking, os.path.join(MODELS_DIR, "stacking_method.pkl"))

    y_pred = stacking.predict(X_test)
    y_proba = stacking.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    if len(set(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    print("\n✅ MÉTRICAS PARA METHOD:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Random state: {RANDOM_STATE}\n")

    with open(os.path.join(MODELS_DIR, "metrics_method.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"Random state: {RANDOM_STATE}\n")


if __name__ == "__main__":
    entrenar(y_train_method, y_test_method)

