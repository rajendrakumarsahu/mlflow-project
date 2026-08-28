import os

import joblib
import pandas as pd

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

CLASS_NAMES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "No trained model found. Run `python src/train.py` first."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict(features, model=None, scaler=None):
    if model is None or scaler is None:
        model, scaler = load_artifacts()
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)[0]
    return CLASS_NAMES[prediction]


def main():
    sample = [5.1, 3.5, 1.4, 0.2]  # sepal_length, sepal_width, petal_length, petal_width
    result = predict(sample)
    print(f"Input: {sample}")
    print(f"Predicted class: {result}")


if __name__ == "__main__":
    main()
