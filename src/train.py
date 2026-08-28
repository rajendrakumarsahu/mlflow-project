import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 2. MLflow Experiment
# --------------------------------------------------

mlflow.set_experiment("Iris_Classification")


# --------------------------------------------------
# 3. Start MLflow Run
# --------------------------------------------------

with mlflow.start_run():

    # Model parameters
    n_estimators = 100
    max_depth = 5
    random_state = 42

    # --------------------------------------------------
    # 4. Create Model
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    # --------------------------------------------------
    # 5. Train Model
    # --------------------------------------------------

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # 6. Prediction
    # --------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # 7. Evaluation
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --------------------------------------------------
    # 8. Log Parameters
    # --------------------------------------------------

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # --------------------------------------------------
    # 9. Log Metrics
    # --------------------------------------------------

    mlflow.log_metric("accuracy", accuracy)

    # --------------------------------------------------
    # 10. Log Model
    # --------------------------------------------------

    mlflow.sklearn.log_model(
        model,
        "random_forest_model"
    )

    print("MLflow run completed successfully!")