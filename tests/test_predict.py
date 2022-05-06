import os
import sys

from predict import predict

sys.path.append("./ml_project")


def test_predict(training_pipeline_params, dataset_without_target):
    features_path = "tests/artifacts/features.csv"
    prediction_path = "tests/artifacts/prediction.txt"

    if os.path.isfile(features_path):
        os.remove(features_path)
    if os.path.isfile(prediction_path):
        os.remove(prediction_path)

    dataset_without_target.to_csv(features_path)

    predict(training_pipeline_params.output_model_path, features_path, prediction_path)

    assert os.path.isfile(prediction_path)
