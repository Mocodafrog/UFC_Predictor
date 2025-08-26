import argparse
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ufc_predictor.train import RANDOM_STATE, train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv-splits", type=int, default=None, help="Número de pliegues de validación cruzada"
    )
    args = parser.parse_args()

    model_names = os.getenv("MODEL_NAMES")
    nombres = [m.strip() for m in model_names.split(",") if m.strip()] if model_names else None
    fast = os.getenv("FAST_MODE", "").lower() in ("1", "true", "yes")
    extended = os.getenv("EXTENDED_SEARCH", "").lower() in ("1", "true", "yes")

    train(
        "Method",
        model_names=nombres,
        fast_mode=fast,
        extended_search=extended,

    )
