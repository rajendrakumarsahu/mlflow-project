import os

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_preprocessing import load_dataset, split_data

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "No trained model found. Run `python src/train.py` first."
        )

    df = load_dataset()
    _, X_test, _, y_test = split_data(df)

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
