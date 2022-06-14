import os
import pandas as pd
import numpy as np
import click
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import classification_report
import json


def read_model(model_path: str) -> LogisticRegression:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


@click.command(name="validator")
@click.option("--model_path")
@click.option("--features_dir")
@click.option("--targets_dir")
@click.option("--output_dir")
def main(model_path: str, features_dir: str, targets_dir: str, output_dir: str):
    model = read_model(model_path)

    X_test = np.loadtxt(f"{features_dir}/X_test.txt")
    y_test = pd.read_csv(f"{targets_dir}/y_test.csv")

    y_pred = model.predict(X_test)

    metrics = classification_report(y_test, y_pred, output_dict=True)

    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/metrics.json", "w") as fp:
        json.dump(metrics, fp)


if __name__ == "__main__":
    main()
