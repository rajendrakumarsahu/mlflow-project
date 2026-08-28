# mlflow-project

An MLflow-tracked scikit-learn classifier (Iris dataset) with a modular training
pipeline, a Streamlit demo app, and Docker support.

## Project structure

```
mlflow-project/
│
├── data/
├── notebooks/
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   └── test_model.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## Setup

```
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Train the model (logs params/metrics/model to MLflow and saves artifacts to `models/`):

```
python src/train.py
```

Evaluate the trained model on the held-out test split:

```
python src/evaluate.py
```

Run a single prediction:

```
python src/predict.py
```

View MLflow tracking UI:

```
mlflow ui
```

Run the Streamlit demo app:

```
streamlit run app/streamlit_app.py
```

Run tests:

```
pytest tests/
```

## Docker

Build and run the Streamlit app in a container:

```
docker build -t mlflow-project .
docker run -p 8501:8501 mlflow-project
```
