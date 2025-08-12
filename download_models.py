import os
from pathlib import Path
import urllib.request

MODEL_BASE_URL = os.getenv(
    "MODEL_BASE_URL",
    "https://github.com/Mocodafrog/UFC_Predictor/releases/download/latest",
)

MODEL_FILES = [
    "stacking_winner.pkl",
    "stacking_method.pkl",
]

def download_model(filename: str):
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    dest = models_dir / filename
    if dest.exists():
        print(f"{filename} already exists, skipping")
        return
    url = f"{MODEL_BASE_URL}/{filename}"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    for fname in MODEL_FILES:
        download_model(fname)
