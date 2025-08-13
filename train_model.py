"""Generic training utilities for UFC Predictor models.

This module exposes a single function :func:`train` that trains the
ensemble models used for the project. The function accepts the name of
the target column (e.g. ``"Method"`` or ``"Winner"``) and handles the
entire training pipeline: loading data, searching hyperparameters for
base estimators, fitting a stacking classifier and persisting both the
models and their evaluation metrics under ``models/{MODEL_VERSION}``.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42
MODEL_VERSION = "1.0"

DATA_DIR = os.path.abspath("data")
MODELS_DIR = os.path.abspath(os.path.join("models", MODEL_VERSION))
os.makedirs(MODELS_DIR, exist_ok=True)

fight_stats = pd.read_csv(os.path.join(DATA_DIR, "fight_stats.csv"))
columnas_X = (
    pd.read_csv(os.path.join(DATA_DIR, "columnas_X.csv"), header=None)
    .squeeze()
    .tolist()
)

X = fight_stats[columnas_X]

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


def train(target_column: str) -> None:
    """Entrena modelos base y un stacking final para ``target_column``.

    Parameters
    ----------
    target_column:
        Nombre de la columna objetivo, por ejemplo ``"Method"`` o ``"Winner"``.
    """

    y = fight_stats[target_column]
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    mejores = []
    print(f"\n📊 Iniciando entrenamiento para {target_column.upper()}\n")
    target_suffix = target_column.lower()
    for nombre, (modelo, params) in models.items():
        print(f"⚙️ GridSearch para {nombre}...")
        grid = GridSearchCV(
            modelo, params, cv=3, scoring="accuracy", verbose=1, n_jobs=-1
        )
        grid.fit(X_train, y_train)
        joblib.dump(
            grid.best_estimator_,
            os.path.join(MODELS_DIR, f"{nombre.lower()}_{target_suffix}.pkl"),
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
    joblib.dump(stacking, os.path.join(MODELS_DIR, f"stacking_{target_suffix}.pkl"))

    y_pred = stacking.predict(X_test)
    y_proba = stacking.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    if len(set(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    print(f"\n✅ MÉTRICAS PARA {target_column.upper()}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Random state: {RANDOM_STATE}\n")

    with open(os.path.join(MODELS_DIR, f"metrics_{target_suffix}.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"Random state: {RANDOM_STATE}\n")
