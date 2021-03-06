import numpy as np
import pandas as pd
from typing import Union
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
import pickle

from ml_project.data.read_params import TrainingParams

SklearnClassifierModel = Union[GradientBoostingClassifier, LogisticRegression]


def create_pipline(
    model: SklearnClassifierModel, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("transform", transformer), ("model", model)])


def train_model(
    features: np.ndarray, target: np.ndarray, train_params: TrainingParams
) -> SklearnClassifierModel:
    if train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            class_weight=train_params.params, random_state=train_params.random_state
        )
    elif train_params.model_type == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(
            n_estimators=train_params.params["n_estimators"],
            max_depth=train_params.params["max_depth"],
            random_state=train_params.random_state,
        )
    else:
        raise NotImplementedError()

    model.fit(features, target)
    return model


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return model.predict(features)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = {
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "f1_score": f1_score(y_true, y_pred, pos_label=1),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    return report


def read_model(model_path: str) -> Pipeline:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def save_predict(prediction: np.ndarray, file_path: str):
    np.savetxt(file_path, prediction, delimiter=",", fmt="%d")


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
