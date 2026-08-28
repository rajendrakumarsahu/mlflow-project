import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from data_preprocessing import load_dataset, split_data
from feature_engineering import scale_features
from predict import CLASS_NAMES, MODEL_PATH, SCALER_PATH, predict


def test_load_dataset():
    df = load_dataset()
    assert not df.empty
    assert "target" in df.columns


def test_split_data():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) + len(X_test) == len(df)
    assert len(X_train) > len(X_test)


def test_scale_features():
    df = load_dataset()
    X_train, X_test, _, _ = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape


@pytest.mark.skipif(
    not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)),
    reason="Model not trained yet — run `python src/train.py` first",
)
def test_predict_returns_valid_class():
    sample = [5.1, 3.5, 1.4, 0.2]
    result = predict(sample)
    assert result in CLASS_NAMES
