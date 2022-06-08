import pytest
import os
import pandas as pd
from faker_generator import gen_fake_dataset


from ml_project.train import train
from ml_project.features import extract_target, build_transformer
from ml_project.data.read_params import (
    FeatureParams,
    TrainingParams,
    DownloadParams,
    TrainingPipelineParams,
    SplittingParams,
)


@pytest.fixture()
def dataset() -> pd.DataFrame:
    return gen_fake_dataset(100, True)


@pytest.fixture()
def dataset_without_target() -> pd.DataFrame:
    return gen_fake_dataset(100, False)


@pytest.fixture()
def feature_params():
    return FeatureParams(
        default_features=["sex", "fbs", "exang"],
        integer_features=["cp", "restecg", "slope", "ca", "thal"],
        float_features=["age", "trestbps", "chol", "oldpeak"],
        target_col="condition",
    )


@pytest.fixture()
def training_params():
    return TrainingParams(
        model_type="LogisticRegression", params={0: 0.3, 1: 0.7}, random_state=42
    )


@pytest.fixture()
def download_params(tmp_path_factory):
    return DownloadParams(
        url="_", file_path=tmp_path_factory.getbasetemp() / "dataset.csv"
    )


@pytest.fixture()
def splitting_params():
    return SplittingParams(0.5, 42)


@pytest.fixture()
def labels(dataset, feature_params) -> pd.DataFrame:
    return extract_target(dataset, feature_params)


@pytest.fixture()
def transformed_dataset(dataset_without_target, feature_params) -> pd.DataFrame:
    transformer = build_transformer(feature_params)
    return transformer.fit_transform(dataset_without_target)


@pytest.fixture()
def training_pipeline_params(
    tmp_path_factory, splitting_params, feature_params, training_params, download_params
):
    return TrainingPipelineParams(
        metrics_path=tmp_path_factory.getbasetemp() / "report.json",
        output_model_path=tmp_path_factory.getbasetemp() / "model.pkl",
        splitting_params=splitting_params,
        feature_params=feature_params,
        model=training_params,
        downloading_params=download_params,
    )


@pytest.fixture()
def dummy_pipeline(dataset, training_pipeline_params):
    if os.path.isfile(training_pipeline_params.downloading_params.file_path):
        os.remove(training_pipeline_params.downloading_params.file_path)
    if os.path.isfile(training_pipeline_params.metrics_path):
        os.remove(training_pipeline_params.metrics_path)
    if os.path.isfile(training_pipeline_params.output_model_path):
        os.remove(training_pipeline_params.output_model_path)

    dataset.to_csv(training_pipeline_params.downloading_params.file_path)
    train(training_pipeline_params)
