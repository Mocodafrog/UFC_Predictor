"""Project configuration constants."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "1.0"

# Default meta-model used by the stacking classifier. Can be overridden when
# calling :func:`ufc_predictor.train`.
STACKING_FINAL_ESTIMATOR = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)
