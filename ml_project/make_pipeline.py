import logging
import click
import sys
import json

import hydra
from omegaconf import DictConfig, OmegaConf

from data.read_params import read_training_pipeline_params
from data.make_dataset import download_data, read_data, split_data
from features.build_features import build_transformer, extract_target, extract_features
from models.model_tools import create_pipline, train_model, predict_model, serialize_model, evaluate

log = logging.getLogger(__name__)
#handler = logging.StreamHandler(sys.stdout)
#log.setLevel(logging.INFO)
#log.addHandler(handler)

def make_pipeline(cfg: OmegaConf):
    log.info(msg="Start pipeline")

    configs = read_training_pipeline_params(cfg)
    download_data(configs.downloading_params)
    df = read_data(configs.downloading_params)

    df_train, df_test = split_data(df, configs.splitting_params, configs.feature_params)

    transformer = build_transformer(configs.feature_params)
    
    y_train = extract_target(df_train, configs.feature_params)
    y_test = extract_target(df_test, configs.feature_params)

    df_train = df_train.drop(labels=configs.feature_params.target_col, axis=1)
    df_test = df_test.drop(labels=configs.feature_params.target_col, axis=1)

    X_train = transformer.fit_transform(df_train)
    model = train_model(X_train, y_train, configs.train_params)

    pipe = create_pipline(model, transformer)    
    y_pred = predict_model(pipe, df_test)

    report = evaluate(y_test, y_pred)
    log.info(msg=f"accuracy = {report['accuracy']}")

    with open(configs.metrics_path, "w") as metric_file:
        json.dump(report, metric_file)

    serialize_model(pipe, configs.output_model_path)

    log.info(msg="Done!")


#@click.command(name="make_pipeline")
#@click.argument("config_path")
#def make_pipeline_command(config_path: str):
#    make_pipeline(config_path)

@hydra.main(config_path="../configs", config_name="config")
def make_pipeline_command(config: OmegaConf):
    cfg = OmegaConf.to_container(config, resolve=True)
    make_pipeline(cfg)


if __name__ == "__main__":
    make_pipeline_command()

