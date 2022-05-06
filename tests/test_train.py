import os
import sys
sys.path.append("./ml_project")


from train import train


def test_train(dataset, training_pipeline_params):
    if os.path.isfile(training_pipeline_params.downloading_params.file_path):
        os.remove(training_pipeline_params.downloading_params.file_path)
    if os.path.isfile(training_pipeline_params.metrics_path):
        os.remove(training_pipeline_params.metrics_path)
    if os.path.isfile(training_pipeline_params.output_model_path):
        os.remove(training_pipeline_params.output_model_path)

    dataset.to_csv(training_pipeline_params.downloading_params.file_path)
    train(training_pipeline_params)

    assert os.path.isfile(training_pipeline_params.downloading_params.file_path)
    assert os.path.isfile(training_pipeline_params.metrics_path)
    assert os.path.isfile(training_pipeline_params.output_model_path)
