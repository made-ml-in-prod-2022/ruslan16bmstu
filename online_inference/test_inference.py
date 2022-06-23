from fastapi.testclient import TestClient
from online_inference.model_api import app, start


client = TestClient(app)
start("online_inference/model_object/model.pkl")


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Heart disease classificator"}


def test_health():
    response = client.get("/healz")
    assert response.status_code == 200
    assert response.content.decode() == "true"


def test_predict():
    request_data = [
        [66, 0, 0, 150, 226, 0, 0, 114, 0, 2.6, 2, 0, 0],
        [65, 1, 0, 138, 282, 1, 2, 174, 0, 1.4, 1, 1, 0],
    ]
    request_features = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]
    response = client.get(
        "/predict", json={"data": request_data, "features": request_features}
    )

    assert response.status_code == 200
    assert response.json() == {"predicts": [0, 1]}
