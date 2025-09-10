
import json
import os
import sys

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ufc_predictor.train import RANDOM_STATE, train



def _load_overrides(path: str) -> dict:
    if path.lower().endswith((".yml", ".yaml")):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit("PyYAML es requerido para archivos YAML") from exc
        with open(path) as f:
            return yaml.safe_load(f)
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":


    model_names = os.getenv("MODEL_NAMES")
    nombres = [m.strip() for m in model_names.split(",") if m.strip()] if model_names else None
    extended = os.getenv("EXTENDED_SEARCH", "").lower() in ("1", "true", "yes")
    # Parámetros adicionales para controlar el stacking y la búsqueda
    search_method = os.getenv("SEARCH_METHOD", "grid")
    passthrough = os.getenv("STACK_PASSTHROUGH", "").lower() in ("1", "true", "yes")

    final_name = os.getenv("FINAL_ESTIMATOR")
    final_estimator = None
    if final_name:
        final_name = final_name.lower()
        if final_name == "logistic":
            final_estimator = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "logreg",
                        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                    ),
                ]
            )
        elif final_name == "gb":
            final_estimator = GradientBoostingClassifier(random_state=RANDOM_STATE)

    train(
        "Winner",
        model_names=nombres,
        extended_search=extended,
        search_method=search_method,
        final_estimator=final_estimator,
        passthrough=passthrough,
    )
