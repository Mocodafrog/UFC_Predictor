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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC

from ufc_predictor.config import MODEL_VERSION

RANDOM_STATE = 42

DATA_DIR = os.path.abspath("data")
MODELS_BASE_DIR = "models"
MODELS_DIR: str | None = None


def _build_models(XGBClassifier, LGBMClassifier, CatBoostClassifier):
    """Return dictionary of base models and parameter grids."""
    return {
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

    La función carga datos y dependencias de forma perezosa y mostrará un
    mensaje amigable si faltan archivos requeridos o librerías opcionales.

    Parameters
    ----------
    target_column:
        Nombre de la columna objetivo, por ejemplo ``"Method"`` o ``"Winner"``.
    """

    global MODELS_DIR

    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
    except ModuleNotFoundError:
        print(
            "❌ Faltan dependencias opcionales (xgboost, lightgbm o catboost). "
            "Instálalas antes de entrenar."
        )
        return

    MODELS_DIR = os.path.abspath(os.path.join(MODELS_BASE_DIR, MODEL_VERSION))
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        fight_stats = pd.read_csv(os.path.join(DATA_DIR, "fight_stats.csv"))
    except FileNotFoundError:
        print(
            "❌ No se encontró 'data/fight_stats.csv'. Genera este archivo antes de entrenar."
        )
        return

    try:
        columnas_X = (
            pd.read_csv(os.path.join(DATA_DIR, "columnas_X.csv"), header=None)
            .squeeze()
            .tolist()
        )
    except FileNotFoundError:
        print(
            "❌ No se encontró 'data/columnas_X.csv'. Asegúrate de crear este archivo."
        )
        return

    X = fight_stats[columnas_X]
    y = fight_stats[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = _build_models(XGBClassifier, LGBMClassifier, CatBoostClassifier)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    mejores = []
    print(f"\n📊 Iniciando entrenamiento para {target_column.upper()}\n")
    target_suffix = target_column.lower()
    for nombre, (modelo, params) in models.items():
        print(f"⚙️ GridSearch para {nombre}...")
        grid = GridSearchCV(
            modelo, params, cv=cv, scoring="accuracy", verbose=1, n_jobs=-1
        )
        grid.fit(X_train, y_train)
        joblib.dump(
            grid.best_estimator_,
            os.path.join(MODELS_DIR, f"{nombre.lower()}_{target_suffix}.pkl"),
        )
        mejores.append((nombre, grid.best_estimator_))
        print(f"✅ {nombre} mejores parámetros: {grid.best_params_}")

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
