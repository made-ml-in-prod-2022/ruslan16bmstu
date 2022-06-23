import numpy as np


from ml_project.features.build_features import (
    build_transformer,
    extract_target,
    extract_features,
)


def test_extract_target(dataset, feature_params):
    target = extract_target(dataset, feature_params)
    assert isinstance(target, np.ndarray)
    assert len(target) == 100


def test_extract_features(dataset_without_target, feature_params):
    transformer = build_transformer(feature_params)
    transformer.fit(dataset_without_target)
    df = extract_features(dataset_without_target, transformer)
    assert df.shape == (100, 24)
