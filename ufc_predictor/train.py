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
    model_names: list[str] | None = None,
    fast_mode: bool = False,
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
    model_names:
        Lista opcional con los nombres de modelos a ejecutar. Los nombres
        deben coincidir con las claves del diccionario ``models``. Si es
        ``None`` se entrenan todos los modelos disponibles.
    fast_mode:
        Si es ``True`` se ejecuta un entrenamiento reducido pensado para
        ejecuciones rápidas: solo se entrenan modelos livianos con un
        único conjunto de hiperparámetros, se usa una fracción pequeña de
        los datos y la validación cruzada utiliza dos particiones.

    Notes
    -----
    Se requiere al menos ``min(3, min_clase)`` muestras por clase para la
    validación estratificada (``min(2, min_clase)`` si se activa
    ``fast_mode``) donde ``min_clase`` es la cantidad mínima de ejemplos por
    clase en ``y``. Si alguna clase tiene menos ejemplos, aumenta los datos de
    entrenamiento.
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

    # Filtrar resultados inválidos para el entrenamiento
    fight_stats = fight_stats[
        (fight_stats["Winner"].isin(["W", "L"]))  # Mantener solo victorias o derrotas
        & (fight_stats["Method"] != "DQ")  # Excluir peleas terminadas por descalificación
    ]

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

    y = fight_stats[target_column]
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y), name=target_column)
    joblib.dump(
        encoder,
        os.path.join(models_dir, f"label_encoder_{target_column.lower()}.pkl"),
    )
    target_suffix = target_column.lower()

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

            # Hyperparameter grids are intentionally small to keep CI runtime low.  With
            # ``n_splits=3`` the largest grid below yields <50 fits.  For thorough
            # experiments, expand these ranges offline (e.g. include more
            # ``learning_rate`` values or deeper ``max_depth``).
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    n_splits = min(2 if fast_mode else 3, min_clase)
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
