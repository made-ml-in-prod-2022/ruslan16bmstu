import pandas as pd
import requests
import click


@click.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000)
@click.option("--data_path", default="sample_data/features.csv")
def main(data_path, host, port):
    data = pd.read_csv(data_path, index_col=0)
    data = data.tail(10)
    request_features = list(data.columns)
    request_data = data.values.tolist()
    response = requests.get(
        f"http://{host}:{port}/predict",
        json={"data": request_data, "features": request_features},
    )
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()
