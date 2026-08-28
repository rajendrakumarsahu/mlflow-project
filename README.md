# mlflow-project

## Project structure

```
mlflow-project/
│
├── data/
│   └── data.csv
│
├── models/
│
├── src/
│   └── train.py
│
├── mlruns/
│
├── requirements.txt
└── README.md
```

## Setup

```
pip install -r requirements.txt
```

## Usage

Place your training data in `data/data.csv`, then run:

```
python src/train.py
```

MLflow will track runs under `mlruns/`. View results with:

```
mlflow ui
```
