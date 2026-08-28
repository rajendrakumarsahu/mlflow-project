import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "target"


def load_dataset() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    return iris.frame


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
