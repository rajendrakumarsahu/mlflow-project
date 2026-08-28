import streamlit as st
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris


st.set_page_config(
    page_title="Iris ML Prediction",
    page_icon="🤖"
)

st.title("🤖 Iris Machine Learning Prediction")

st.write("Random Forest model with MLflow tracking")


# --------------------------------------------------
# Load model (latest run of the Iris_Classification experiment)
# --------------------------------------------------

EXPERIMENT_NAME = "Iris_Classification"
MODEL_ARTIFACT_NAME = "random_forest_model"


@st.cache_resource
def load_latest_model():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"No experiment named '{EXPERIMENT_NAME}' found. Run `python src/train.py` first."
        )

    runs = client.search_runs(
        [experiment.experiment_id], order_by=["start_time DESC"], max_results=1
    )
    if not runs:
        raise RuntimeError(
            f"No runs found for experiment '{EXPERIMENT_NAME}'. Run `python src/train.py` first."
        )

    latest_run_id = runs[0].info.run_id
    model_uri = f"runs:/{latest_run_id}/{MODEL_ARTIFACT_NAME}"
    return mlflow.sklearn.load_model(model_uri), latest_run_id


model, run_id = load_latest_model()
st.caption(f"Serving model from run `{run_id}`")


# --------------------------------------------------
# User Input
# --------------------------------------------------

sepal_length = st.number_input(
    "Sepal Length",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.0,
    value=0.2
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict"):

    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(input_data)

    iris = load_iris()

    predicted_class = iris.target_names[prediction[0]]

    st.success(
        f"Predicted Iris Class: {predicted_class}"
    )