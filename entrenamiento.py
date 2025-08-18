import os

from ufc_predictor.train import train

if __name__ == "__main__":
    model_names = os.getenv("MODEL_NAMES")
    nombres = [m.strip() for m in model_names.split(",") if m.strip()] if model_names else None
    train("Method", model_names=nombres)
