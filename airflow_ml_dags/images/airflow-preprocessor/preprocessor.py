import click
import os
import pandas as pd
import numpy as np
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


@click.command(name="preprocessor")
@click.option("--input_dir")
@click.option("--output_dir")
def main(input_dir: str, output_dir: str):
    X_train = pd.read_csv(f"{input_dir}/X_train.csv")
    X_test = pd.read_csv(f"{input_dir}/X_test.csv")

    X_train_processed = transformer.fit_transform(X_train)
    X_test_processed = transformer.transform(X_test)

    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(f"{output_dir}/X_train.txt", X_train_processed)
    np.savetxt(f"{output_dir}/X_test.txt", X_test_processed)


if __name__ == "__main__":
    main()
