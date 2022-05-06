import logging
import os
import json
import hydra
from omegaconf import OmegaConf

from data.read_params import read_training_pipeline_params, TrainingPipelineParams
from data.make_dataset import download_data, read_data, split_data
from features.build_features import build_transformer, extract_target
from models.model_tools import (
    create_pipline,
    train_model,
    predict_model,
    serialize_model,
    evaluate,
)

log = logging.getLogger(__name__)


def train(configs: TrainingPipelineParams):

    log.info(msg="reading dataset")
    if not os.path.exists(configs.downloading_params.file_path):
        download_data(configs.downloading_params)
    df = read_data(configs.downloading_params.file_path)

    log.info(msg="splitting dataset")
    df_train, df_test = split_data(
        df, configs.splitting_params, configs.feature_params.target_col
    )

    log.info(msg="target extraction")
    y_train = extract_target(df_train, configs.feature_params)
    y_test = extract_target(df_test, configs.feature_params)

    df_train = df_train.drop(labels=configs.feature_params.target_col, axis=1)
    df_test = df_test.drop(labels=configs.feature_params.target_col, axis=1)

    log.info(msg="building transformer")
    transformer = build_transformer(configs.feature_params)

    log.info(msg="transforming features")
    X_train = transformer.fit_transform(df_train)
    model = train_model(X_train, y_train, configs.model)

    log.info(msg="creating pipeline")
    pipe = create_pipline(model, transformer)
    y_pred = predict_model(pipe, df_test)

    log.info(msg="evalueting")
    report = evaluate(y_test, y_pred)
    log.info(msg=f"accuracy = {report['accuracy']}")

    log.info(msg="saving metrics")
    with open(configs.metrics_path, "w") as metric_file:
        json.dump(report, metric_file)

    log.info(msg="saving model")
    serialize_model(pipe, configs.output_model_path)

    log.info(msg="done!")


@hydra.main(config_path="../configs", config_name="config")
def make_pipeline_command(config: OmegaConf):
    cfg = OmegaConf.to_container(config, resolve=True)

    log.info(msg="reading config")
    configs = read_training_pipeline_params(cfg)

    train(configs)


if __name__ == "__main__":
    make_pipeline_command()
