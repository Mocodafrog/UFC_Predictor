
import json
import os
import sys

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

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
    fast = os.getenv("FAST_MODE", "").lower() in ("1", "true", "yes")
    extended = os.getenv("EXTENDED_SEARCH", "").lower() in ("1", "true", "yes")

    train(
        "Winner",
        model_names=nombres,
        fast_mode=fast,
        extended_search=extended,
    )
