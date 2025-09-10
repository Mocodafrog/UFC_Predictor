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
import json
import math

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from ufc_predictor.config import MODEL_VERSION, STACKING_FINAL_ESTIMATOR

RANDOM_STATE = 42

DATA_DIR = os.path.abspath("data")


def _merge_params(base: dict, overrides: dict) -> dict:
    """Merge hyperparameter grids, extending or replacing existing values."""

    def _to_list(value):
        return list(value) if isinstance(value, (list, tuple)) else [value]

    merged = {k: _to_list(v) for k, v in base.items()}
    for param, values in overrides.items():
        vals = _to_list(values)
        if param in merged:
            merged[param].extend(v for v in vals if v not in merged[param])
        else:
            merged[param] = vals
    return merged


def train(
    target_column: str,
    data_dir: str = DATA_DIR,
    models_dir: str | None = None,
    models: dict | None = None,
    model_names: list[str] | None = None,
    fast_mode: bool = False,
    extended_search: bool = False,
    grid_overrides: dict | None = None,
    cv_splits: int = 5,

):
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
    model_names:
        Lista opcional con los nombres de modelos a ejecutar. Los nombres
        deben coincidir con las claves del diccionario ``models``. Si es
        ``None`` se entrenan todos los modelos disponibles.
    fast_mode:
        Si es ``True`` se ejecuta un entrenamiento reducido pensado para
        ejecuciones rápidas: solo se entrenan modelos livianos con un
        único conjunto de hiperparámetros, se usa una fracción pequeña de
        los datos y la validación cruzada utiliza dos particiones.
    extended_search:
        Activa grids de hiperparámetros más amplios para una búsqueda más
        exhaustiva a costa de mayor tiempo de entrenamiento.
    grid_overrides:
        Diccionario con hiperparámetros adicionales por modelo. Las claves
        deben coincidir con las del diccionario ``models`` y sus valores se
        combinan con los grids por defecto, sustituyendo o ampliando los
        existentes.
    cv_splits:
        Número de particiones para la validación cruzada (se reduce a 2 si
        ``fast_mode`` es ``True``).
    search_method:
        Estrategia de búsqueda de hiperparámetros. Puede ser ``"grid"`` para
        :class:`~sklearn.model_selection.GridSearchCV`, ``"random"`` para
        :class:`~sklearn.model_selection.RandomizedSearchCV` o ``"bayes"`` para
        una búsqueda bayesiana mediante Optuna o scikit-optimize.
    final_estimator:
        Estimador final del stacking. Por defecto se utiliza el definido en
        ``ufc_predictor.config.STACKING_FINAL_ESTIMATOR``.
    passthrough:
        Si es ``True`` el meta-modelo recibe también las características
        originales además de las predicciones de los modelos base.

    Notes
    -----
    Se requiere al menos ``min(cv_splits, min_clase)`` muestras por clase para
    la validación estratificada (``min(2, min_clase)`` si se activa
    ``fast_mode``) donde ``min_clase`` es la cantidad mínima de ejemplos por
    clase en ``y``. Si alguna clase tiene menos ejemplos, aumenta los datos de
    entrenamiento.
    """

    if models_dir is None:
        models_dir = os.path.abspath(os.path.join("models", MODEL_VERSION))
    os.makedirs(models_dir, exist_ok=True)

    if final_estimator is None:
        final_estimator = STACKING_FINAL_ESTIMATOR

    try:
        fight_stats = pd.read_csv(os.path.join(data_dir, "fight_stats.csv"))
    except FileNotFoundError:
        print(
            "❌ No se encontró 'data/fight_stats.csv'. Genera este archivo antes de entrenar."
        )
        raise SystemExit(1)

    # Filtrar resultados inválidos para el entrenamiento
    fight_stats = fight_stats[
        (fight_stats["Winner"].isin(["W", "L"]))  # Mantener solo victorias o derrotas
        & (fight_stats["Method"] != "DQ")  # Excluir peleas terminadas por descalificación
    ]

    # Conservar una copia con todas las columnas originales para analizar
    # posteriormente los casos mal clasificados.
    fight_stats_original = fight_stats.copy()

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

    # Ensure essential bout metadata columns are always included when available
    critical_cols = [
        c for c in ["Rounds", "Format", "Weight Class"] if c in fight_stats.columns
    ]
    for col in critical_cols:
        if col not in columnas_procesadas:
            columnas_procesadas.append(col)

    if not columnas_procesadas:
        raise SystemExit("No se pudieron alinear columnas para entrenamiento")

    if "Rounds" in fight_stats.columns:
        fight_stats["Rounds"] = pd.to_numeric(
            fight_stats["Rounds"], errors="coerce"
        )
    if "Format" in fight_stats.columns:
        fight_stats["Format"] = (
            fight_stats["Format"].str.extract(r"(\d+)").astype(float)
        )
    if "Weight Class" in fight_stats.columns:
        fight_stats["Weight Class"] = pd.Categorical(
            fight_stats["Weight Class"]
        ).codes

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

    # Elimina columnas completamente vacías e imputa el resto
    vacias = X.columns[X.isna().all()].tolist()
    if vacias:
        print(
            "⚠️ Las siguientes columnas no contienen datos y se descartarán:",
            ", ".join(vacias),
        )
        X = X.drop(columns=vacias)
        columnas_procesadas = [c for c in columnas_procesadas if c not in vacias]

    if X.isna().any().any():
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    # Persistir las columnas utilizadas durante el entrenamiento
    target_suffix = target_column.lower()
    with open(os.path.join(models_dir, f"features_{target_suffix}.json"), "w") as f:
        json.dump(columnas_procesadas, f)

    y = fight_stats[target_column]
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y), name=target_column)
    joblib.dump(
        encoder,
        os.path.join(models_dir, f"label_encoder_{target_suffix}.pkl"),
    )

    if fast_mode:
        X = X.sample(frac=0.1, random_state=RANDOM_STATE)
        y = y.loc[X.index]

    min_clase = y.value_counts().min()

    if models is None or fast_mode:
        if fast_mode:
            models = {
                "RandomForest": (
                    RandomForestClassifier(random_state=RANDOM_STATE),
                    {
                        "n_estimators": [50],
                        "max_depth": [10],
                        "min_samples_split": [2],
                    },
                ),
                "LogisticRegression": (
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                    {"C": [1], "solver": ["lbfgs"]},
                ),
            }
            model_names = ["LogisticRegression", "RandomForest"]
        else:
            from lightgbm import LGBMClassifier
            from xgboost import XGBClassifier
            from catboost import CatBoostClassifier

            if extended_search:
                models = {
                    "XGBoost": (
                        XGBClassifier(random_state=RANDOM_STATE),
                        {
                            "learning_rate": [0.05, 0.1, 0.2],
                            "n_estimators": [50, 100, 200],
                            "max_depth": [3, 5, 7],
                        },
                    ),
                    "LightGBM": (
                        LGBMClassifier(random_state=RANDOM_STATE),
                        {
                            "learning_rate": [0.05, 0.1, 0.2],
                            "n_estimators": [50, 100, 200],
                            "max_depth": [3, 5, 7],
                        },
                    ),
                    "CatBoost": (
                        CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
                        {
                            "learning_rate": [0.01, 0.1],
                            "iterations": [50, 100, 200],
                        },
                    ),
                    "RandomForest": (
                        RandomForestClassifier(random_state=RANDOM_STATE),
                        {
                            "n_estimators": [100, 200, 300],
                            "max_depth": [None, 10, 20, 30],
                            "min_samples_split": [2, 5, 10],
                        },
                    ),
                    "GradientBoosting": (
                        GradientBoostingClassifier(random_state=RANDOM_STATE),
                        {
                            "learning_rate": [0.05, 0.1],
                            "n_estimators": [100, 200],
                            "max_depth": [3, 5, 7],
                        },
                    ),
                    "LogisticRegression": (
                        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                        {
                            "C": [0.01, 0.1, 1, 10, 100],
                            "solver": ["lbfgs", "liblinear"],
                        },
                    ),
                    "SVC": (
                        SVC(probability=True, random_state=RANDOM_STATE),
                        {
                            "C": [0.1, 1, 10, 100],
                            "kernel": ["linear", "rbf", "poly"],
                        },
                    ),
                }
            else:
                # Hyperparameter grids are intentionally small to keep CI runtime low.
                # Con ``n_splits=3`` el grid más grande genera <50 ajustes.
                models = {
                    "XGBoost": (
                        XGBClassifier(random_state=RANDOM_STATE),
                        {
                            "learning_rate": [0.1],
                            "n_estimators": [50, 100],
                            "max_depth": [3, 5],
                        },
                    ),
                    "LightGBM": (
                        LGBMClassifier(random_state=RANDOM_STATE),
                        {
                            "learning_rate": [0.1],
                            "n_estimators": [50, 100],
                            "max_depth": [3, 5],
                        },
                    ),
                    "CatBoost": (
                        CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
                        {"learning_rate": [0.1], "iterations": [50, 100]},
                    ),
                    "RandomForest": (
                        RandomForestClassifier(random_state=RANDOM_STATE),
                        {
                            "n_estimators": [50, 100],
                            "max_depth": [10, 20],
                            "min_samples_split": [2, 5],
                        },
                    ),
                    "GradientBoosting": (
                        GradientBoostingClassifier(random_state=RANDOM_STATE),
                        {
                            "learning_rate": [0.1],
                            "n_estimators": [50, 100],
                            "max_depth": [3, 5],
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

    if model_names:
        nombres = {n.lower() for n in model_names}
        models = {k: v for k, v in models.items() if k.lower() in nombres}
        if not models:
            raise SystemExit("No se encontraron modelos válidos en model_names")

    if grid_overrides:
        for nombre, override in grid_overrides.items():
            if nombre in models:
                modelo, params = models[nombre]
                models[nombre] = (modelo, _merge_params(params, override))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    n_splits = min(cv_splits if not fast_mode else 2, min_clase)
    # Se requieren al menos ``n_splits`` muestras por clase o debes aumentar los datos.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    mejores = []
    resultados: dict[str, dict] = {}
    print(f"\n📊 Iniciando entrenamiento para {target_column.upper()}\n")
    for nombre, (modelo, params) in models.items():
        print(f"⚙️ {search_method.capitalize()}Search para {nombre}...")
        if search_method == "grid":
            search = GridSearchCV(
                modelo, params, cv=cv, scoring="accuracy", verbose=1, n_jobs=-1
            )
        elif search_method == "random":
            total = math.prod(len(v) for v in params.values())
            n_iter = min(10, total)
            search = RandomizedSearchCV(
                modelo,
                params,
                n_iter=n_iter,
                cv=cv,
                scoring="accuracy",
                verbose=1,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        elif search_method == "bayes":
            try:
                from optuna.integration import OptunaSearchCV

                search = OptunaSearchCV(
                    modelo,
                    params,
                    cv=cv,
                    scoring="accuracy",
                    n_trials=10,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            except Exception:
                try:
                    from skopt import BayesSearchCV
                    from skopt.space import Categorical

                    space = {k: Categorical(v) for k, v in params.items()}
                    search = BayesSearchCV(
                        modelo,
                        space,
                        n_iter=10,
                        cv=cv,
                        scoring="accuracy",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    )
                except Exception as exc:  # pragma: no cover - env dep
                    raise ImportError(
                        "Optuna o scikit-optimize son necesarios para search_method='bayes'"
                    ) from exc
        else:
            raise ValueError("search_method debe ser 'grid', 'random' o 'bayes'")

        search.fit(X_train, y_train)
        joblib.dump(
            search.best_estimator_,
            os.path.join(models_dir, f"{nombre.lower()}_{target_suffix}.pkl"),
        )
        mejores.append((nombre, search.best_estimator_))
        resultados[nombre] = {
            "best_params": search.best_params_,
            "best_score": getattr(search, "best_score_", None),
        }
        print(
            f"✅ {nombre} mejores parámetros ({search_method}): {search.best_params_}"
        )

    stacking = StackingClassifier(
        estimators=mejores,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=-1,
        passthrough=passthrough,
    )
    stacking.fit(X_train, y_train)
    joblib.dump(stacking, os.path.join(models_dir, f"stacking_{target_suffix}.pkl"))

    y_pred = stacking.predict(X_test)
    y_proba = stacking.predict_proba(X_test)

    # Guardar filas donde el modelo se equivocó para su análisis posterior
    mis_idx = y_test.index[y_pred != y_test]
    mispredictions = fight_stats_original.loc[mis_idx]
    mispredictions.to_csv(
        os.path.join(data_dir, f"mispredictions_{target_suffix}.csv"), index=False
    )

    accuracy = accuracy_score(y_test, y_pred)
    if len(set(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    print(f"\n✅ MÉTRICAS PARA {target_column.upper()}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Random state: {RANDOM_STATE}\n")
    metrics_path = os.path.join(models_dir, f"metrics_{target_suffix}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Search method: {search_method}\n")
        for nombre, info in resultados.items():
            f.write(f"{nombre} best params: {info['best_params']}\n")
            if info["best_score"] is not None:
                f.write(f"{nombre} best score: {info['best_score']:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"Random state: {RANDOM_STATE}\n")

    return stacking
