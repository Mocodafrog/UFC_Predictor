"""Generic training utilities for UFC Predictor models.

This module exposes a single function :func:`train` that trains the
ensemble models used for the project. The function accepts the name of
the target column (e.g. ``"Method"`` or ``"Winner"``) and handles the
entire training pipeline: loading data, searching hyperparameters for
base estimators, fitting a stacking classifier and persisting both the
models and their evaluation metrics under ``models/{MODEL_VERSION}``.
"""

__all__ = ["train"]

import os

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from ufc_predictor.config import MODEL_VERSION

RANDOM_STATE = 42

DATA_DIR = os.path.abspath("data")


def train(
    target_column: str,
    data_dir: str = DATA_DIR,
    models_dir: str | None = None,
    models: dict | None = None,
) -> None:
    """Entrena modelos base y un stacking final para ``target_column``.

    Parameters
    ----------
    target_column:
        Nombre de la columna objetivo, por ejemplo ``"Method"`` o ``"Winner"``.
    data_dir:
        Directorio donde se encuentran ``fight_stats.csv`` y ``columnas_X.csv``.
    models_dir:
        Directorio donde se guardarán los modelos entrenados. Si ``None`` se
        utiliza ``models/{MODEL_VERSION}``.
    models:
        Diccionario opcional con los modelos a entrenar. Si no se provee se
        utilizan los modelos por defecto y se importan las dependencias
        pesadas dentro de esta función.

    Notes
    -----
    Se requiere al menos ``min(3, min_clase)`` muestras por clase para la
    validación estratificada donde ``min_clase`` es la cantidad mínima de
    ejemplos por clase en ``y``. Si alguna clase tiene menos ejemplos,
    aumenta los datos de entrenamiento.
    """

    if models_dir is None:
        models_dir = os.path.abspath(os.path.join("models", MODEL_VERSION))
    os.makedirs(models_dir, exist_ok=True)

    try:
        fight_stats = pd.read_csv(os.path.join(data_dir, "fight_stats.csv"))
    except FileNotFoundError:
        print(
            "❌ No se encontró 'data/fight_stats.csv'. Genera este archivo antes de entrenar."
        )
        raise SystemExit(1)

    try:
        columnas_X = pd.read_csv(
            os.path.join(data_dir, "columnas_X.csv"), header=None
        ).squeeze()
        if isinstance(columnas_X, str):
            columnas_X = [columnas_X]
        else:
            columnas_X = columnas_X.tolist()
    except FileNotFoundError:
        print(
            "❌ No se encontró 'data/columnas_X.csv'. Asegúrate de crear este archivo."
        )
        raise SystemExit(1)

    # Normalizar nombres de columnas para empatar con los de ``fight_stats``.
    def _sanitize(name: str) -> str:
        return name.lower().replace(" ", "_").replace(".", "").replace("/", "_")

    columnas_procesadas: list[str] = []
    faltantes: list[str] = []
    for col in columnas_X:
        if col in fight_stats.columns:
            columnas_procesadas.append(col)
            continue
        sanitized = _sanitize(col)
        # Intentar emparejar con versión normalizada
        if sanitized in fight_stats.columns:
            columnas_procesadas.append(sanitized)
        else:
            faltantes.append(col)

    if faltantes:
        print(
            "⚠️ Las siguientes columnas no se encontraron y serán ignoradas:",
            ", ".join(faltantes),
        )

    if not columnas_procesadas:
        raise SystemExit("No se pudieron alinear columnas para entrenamiento")

    X = fight_stats[columnas_procesadas]
    # Elimina columnas no numéricas que provocarían errores en los modelos
    no_numericas = X.select_dtypes(exclude="number").columns.tolist()
    if no_numericas:
        print(
            "⚠️ Las siguientes columnas son no numéricas y se descartarán:",
            ", ".join(no_numericas),
        )
        X = X.drop(columns=no_numericas)
        columnas_procesadas = [c for c in columnas_procesadas if c not in no_numericas]

    y = fight_stats[target_column]
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y), name=target_column)
    joblib.dump(
        encoder,
        os.path.join(models_dir, f"label_encoder_{target_column.lower()}.pkl"),
    )
    target_suffix = target_column.lower()
    min_clase = y.value_counts().min()

    if models is None:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    n_splits = min(3, min_clase)
    # Se requieren al menos ``n_splits`` muestras por clase o debes aumentar los datos.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    mejores = []
    print(f"\n📊 Iniciando entrenamiento para {target_column.upper()}\n")
    for nombre, (modelo, params) in models.items():
        print(f"⚙️ GridSearch para {nombre}...")
        grid = GridSearchCV(
            modelo, params, cv=cv, scoring="accuracy", verbose=1, n_jobs=-1
        )
        grid.fit(X_train, y_train)
        joblib.dump(
            grid.best_estimator_,
            os.path.join(models_dir, f"{nombre.lower()}_{target_suffix}.pkl"),
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
    joblib.dump(stacking, os.path.join(models_dir, f"stacking_{target_suffix}.pkl"))

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

    with open(os.path.join(models_dir, f"metrics_{target_suffix}.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"Random state: {RANDOM_STATE}\n")
