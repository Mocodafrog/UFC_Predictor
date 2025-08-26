
import os

from ufc_predictor.train import train

if __name__ == "__main__":
    model_names = os.getenv("MODEL_NAMES")
    nombres = [m.strip() for m in model_names.split(",") if m.strip()] if model_names else None
    fast = os.getenv("FAST_MODE", "").lower() in ("1", "true", "yes")
    extended = os.getenv("EXTENDED_SEARCH", "").lower() in ("1", "true", "yes")
    train("Winner", model_names=nombres, fast_mode=fast, extended_search=extended)
