import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import ufc_predictor.train as train_module
from ufc_predictor.train import train


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
    y = ["W", "L"] * 6

    fight_stats = X.assign(Winner=y, Method="KO")
    fight_stats.to_csv(tmp_path / "fight_stats.csv", index=False)
    pd.Series(["feat"]).to_csv(tmp_path / "columnas_X.csv", index=False, header=False)

    models = {
        "LogReg": (LogisticRegression(random_state=train_module.RANDOM_STATE), {})
    }

    monkeypatch.setattr(train_module.joblib, "dump", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "GridSearchCV", DummyGridSearchCV)
    monkeypatch.setattr(train_module, "StackingClassifier", DummyStackingClassifier)

    splits = []
    real_tts = train_module.train_test_split

    def record_split(*args, **kwargs):
        result = real_tts(*args, **kwargs)
        splits.append(result)
        return result

    monkeypatch.setattr(train_module, "train_test_split", record_split)

    train("Winner", data_dir=str(tmp_path), models_dir=str(tmp_path), models=models)
    train("Winner", data_dir=str(tmp_path), models_dir=str(tmp_path), models=models)

    (X_train1, X_test1, y_train1, y_test1), (X_train2, X_test2, y_train2, y_test2) = splits

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)


def test_extended_search_expands_params(tmp_path, monkeypatch):
    X = pd.DataFrame({"feat": range(12)})
    y = ["W", "L"] * 6

    fight_stats = X.assign(Winner=y, Method="KO")
    fight_stats.to_csv(tmp_path / "fight_stats.csv", index=False)
    pd.Series(["feat"]).to_csv(tmp_path / "columnas_X.csv", index=False, header=False)

    captured: dict = {}

    def capture_grid(estimator, params, *args, **kwargs):
        captured["params"] = params
        return DummyGridSearchCV(estimator, *args, **kwargs)

    monkeypatch.setattr(train_module.joblib, "dump", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "GridSearchCV", capture_grid)
    monkeypatch.setattr(train_module, "StackingClassifier", DummyStackingClassifier)

    train(
        "Winner",
        data_dir=str(tmp_path),
        models_dir=str(tmp_path),
        model_names=["LogisticRegression"],
        extended_search=True,
    )

    assert len(captured["params"]["C"]) > 3
    assert 0.01 in captured["params"]["C"]
    assert 100 in captured["params"]["C"]
