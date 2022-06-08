import logging
import sys
import click

from ml_project.data.make_dataset import read_data
from ml_project.models.model_tools import save_predict, read_model

log = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
log.addHandler(handler)


def predict(model_path: str, features_path: str, prediction_path: str):

    log.info(msg="reading model")
    model = read_model(model_path)
    log.info(msg=f"model: {model.named_steps['model']}")

    log.info(msg="reading features")
    df = read_data(features_path)

    if "condition" in df.columns:
        df = df.drop(labels="condition", axis=1)
    log.info(msg=f"features shape: {df.shape}")

    log.info(msg="start prediction")
    y_pred = model.predict(df)

    log.info(msg="saving result")
    save_predict(y_pred, prediction_path)

    log.info(msg="done!")


@click.command(name="predict_pipeline")
@click.argument("model_path")
@click.argument("features_path")
@click.argument("prediction_path")
def make_pipeline_command(model_path: str, features_path: str, prediction_path: str):
    predict(model_path, features_path, prediction_path)


if __name__ == "__main__":
    make_pipeline_command()
