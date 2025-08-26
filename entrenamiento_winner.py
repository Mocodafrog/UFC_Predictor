
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ufc_predictor.train import RANDOM_STATE, train

if __name__ == "__main__":
    model_names = os.getenv("MODEL_NAMES")
    nombres = [m.strip() for m in model_names.split(",") if m.strip()] if model_names else None
    fast = os.getenv("FAST_MODE", "").lower() in ("1", "true", "yes")
    extended = os.getenv("EXTENDED_SEARCH", "").lower() in ("1", "true", "yes")
    meta_name = os.getenv("FINAL_ESTIMATOR", "LogisticRegression")
    meta_models = {
        "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    final_estimator = meta_models.get(meta_name)
    if final_estimator is None:
        raise SystemExit(f"Final estimator '{meta_name}' no soportado")
    train(
        "Winner",
        model_names=nombres,
        fast_mode=fast,
        extended_search=extended,
        final_estimator=final_estimator,
    )
