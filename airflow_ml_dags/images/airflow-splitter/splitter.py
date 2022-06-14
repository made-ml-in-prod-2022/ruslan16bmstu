import pandas as pd
import click
import os
from sklearn.model_selection import train_test_split


@click.command(name="splitter")
@click.option("--input_dir")
@click.option("--output_dir")
@click.option("--val_size", default=0.2)
@click.option("--state", default=42)
def main(input_dir: str, output_dir: str, val_size: float, state: int):
    features = pd.read_csv(f"{input_dir}/features.csv")
    targets = pd.read_csv(f"{input_dir}/targets.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=val_size, random_state=state
    )

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)

    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)


if __name__ == "__main__":
    main()
