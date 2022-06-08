import os


from ml_project.predict import predict


def test_predict(
    tmp_path_factory, dummy_pipeline, training_pipeline_params, dataset_without_target
):
    dummy_pipeline
    features_path = tmp_path_factory.getbasetemp() / "features.csv"
    prediction_path = tmp_path_factory.getbasetemp() / "prediction.txt"

    if os.path.isfile(features_path):
        os.remove(features_path)
    if os.path.isfile(prediction_path):
        os.remove(prediction_path)

    dataset_without_target.to_csv(features_path)

    predict(training_pipeline_params.output_model_path, features_path, prediction_path)

    assert os.path.isfile(prediction_path)
