import os
import pandas as pd
import numpy as np
import click
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


scale_list = ["age", "trestbps", "chol", "oldpeak"]
ohe_list = ["cp", "restecg", "slope", "ca", "thal"]

transformer = ColumnTransformer(
    [
        ("default", "passthrough", ["sex", "fbs", "exang"]),
        ("min_max", MinMaxScaler(), scale_list),
        ("ohe", OneHotEncoder(sparse=False), ohe_list),
    ]
)


def read_model(model_path: str) -> LogisticRegression:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


@click.command(name="validator")
@click.option("--model_path")
@click.option("--features_dir")
@click.option("--output_dir")
def main(model_path: str, features_dir: str, output_dir: str):
    model = read_model(model_path)

    df = pd.read_csv(f"{features_dir}/features.csv")

    features = transformer.fit_transform(df)

    y_pred = model.predict(features)

    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(f"{output_dir}/predictions.txt", y_pred)


if __name__ == "__main__":
    main()
