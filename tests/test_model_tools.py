import pytest
from sklearn.linear_model import LogisticRegression
import os

from ml_project.models.model_tools import train_model, serialize_model, read_model


@pytest.fixture()
def dummy_model(transformed_dataset, labels, training_params):
    return train_model(transformed_dataset, labels, training_params)


def test_train_model(transformed_dataset, labels, training_params):
    model = train_model(transformed_dataset, labels, training_params)
    assert isinstance(model, LogisticRegression)


def test_serialize_model(tmp_path_factory):
    file_path = tmp_path_factory.getbasetemp() / "array.pkl"
    array = list(range(0, 100))
    serialize_model(array, file_path)
    assert os.path.exists(file_path)


def test_read_model(tmp_path_factory):
    file_path = tmp_path_factory.getbasetemp() / "array.pkl"
    array = read_model(file_path)
    assert array == list(range(0, 100))
    assert array != list(range(0, 50))
