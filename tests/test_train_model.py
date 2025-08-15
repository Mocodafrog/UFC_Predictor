import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import types


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import train_model  # noqa: E402


class DummyGridSearchCV:
    """Lightweight replacement for GridSearchCV used in tests."""

    def __init__(self, estimator, *args, **kwargs):
        self.best_estimator_ = estimator
        self.best_params_ = {}

    def fit(self, X, y):  # pragma: no cover - trivial
        self.best_estimator_.fit(X, y)
        return self


class DummyStackingClassifier:
    """Simplified stacking classifier for fast deterministic tests."""

    def __init__(self, *args, **kwargs):
        self.classes_ = None

    def fit(self, X, y):  # pragma: no cover - trivial
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):  # pragma: no cover - trivial
        return np.repeat(self.classes_[0], len(X))

    def predict_proba(self, X):  # pragma: no cover - trivial
        proba = np.zeros((len(X), len(self.classes_)))
        proba[:, 0] = 1.0
        return proba


def test_train_split_deterministic(tmp_path, monkeypatch):
    X = pd.DataFrame({"feat": range(12)})
    y = [0, 1] * 6
    fight_stats = X.assign(Winner=y)
    fight_stats.to_csv(tmp_path / "fight_stats.csv", index=False)
    pd.DataFrame(["feat", "feat"]).to_csv(
        tmp_path / "columnas_X.csv", index=False, header=False
    )

    train_model.DATA_DIR = tmp_path
    train_model.MODELS_BASE_DIR = str(tmp_path)

    monkeypatch.setattr(train_model.joblib, "dump", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_model, "GridSearchCV", DummyGridSearchCV)
    monkeypatch.setattr(train_model, "StackingClassifier", DummyStackingClassifier)
    monkeypatch.setattr(
        train_model,
        "_build_models",
        lambda *args: {
            "LogReg": (LogisticRegression(random_state=train_model.RANDOM_STATE), {})
        },
    )

    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(XGBClassifier=LogisticRegression))
    monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace(LGBMClassifier=LogisticRegression))
    monkeypatch.setitem(sys.modules, "catboost", types.SimpleNamespace(CatBoostClassifier=LogisticRegression))

    splits = []
    real_tts = train_model.train_test_split

    def record_split(*args, **kwargs):
        result = real_tts(*args, **kwargs)
        splits.append(result)
        return result

    monkeypatch.setattr(train_model, "train_test_split", record_split)

    train_model.train("Winner")
    train_model.train("Winner")

    (X_train1, X_test1, y_train1, y_test1), (X_train2, X_test2, y_train2, y_test2) = splits

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)

