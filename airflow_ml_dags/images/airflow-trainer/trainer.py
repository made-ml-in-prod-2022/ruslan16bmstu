import pandas as pd
import numpy as np
import click
import os
from sklearn.linear_model import LogisticRegression
import pickle


@click.command(name="preprocessor")
@click.option("--features_dir")
@click.option("--targets_dir")
@click.option("--output_dir")
@click.option("--state", default=42)
def main(features_dir: str, targets_dir: str, output_dir: str, state: int):
    X_train = np.loadtxt(f"{features_dir}/X_train.txt")
    y_train = pd.read_csv(f"{targets_dir}/y_train.csv")

    model = LogisticRegression(random_state=state)
    model.fit(X_train, y_train)

    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
