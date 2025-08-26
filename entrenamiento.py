import argparse
import os

from ufc_predictor.train import train


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
    cv_env = os.getenv("CV_SPLITS")
    cv_splits = int(cv_env) if cv_env else (args.cv_splits or 3)

    train(
        "Method",
        model_names=nombres,
        fast_mode=fast,
        extended_search=extended,
        cv_splits=cv_splits,
    )
