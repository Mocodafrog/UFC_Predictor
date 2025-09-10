"""Project configuration constants."""

from sklearn.linear_model import LogisticRegression

MODEL_VERSION = "1.0"

# Default meta-model used by the stacking classifier. Can be overridden when
# calling :func:`ufc_predictor.train`.
STACKING_FINAL_ESTIMATOR = LogisticRegression()
