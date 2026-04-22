import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression, LogisticRegression

from sk_serve import SimpleAPI, serve


@pytest.fixture
def regressor():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 1.0, 0.0, 0.0]})
    y = [0.0, 1.0, 2.0, 3.0]
    return LinearRegression().fit(X, y)


@pytest.fixture
def classifier():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 1.0, 0.0, 0.0]})
    y = [0, 0, 1, 1]
    return LogisticRegression().fit(X, y)


def _make_client(api: SimpleAPI, model) -> TestClient:
    return TestClient(serve(api, pipeline=model))


def test_home_message(regressor):
    with _make_client(SimpleAPI(), regressor) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Available endpoints" in response.json()["message"]


def test_health_ok(regressor):
    with _make_client(SimpleAPI(), regressor) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_inference_single_record(regressor):
    with _make_client(SimpleAPI(), regressor) as client:
        response = client.post("/inference", json={"a": 1.0, "b": 1.0})
        assert response.status_code == 200
        body = response.json()
        assert "prediction" in body
        assert isinstance(body["prediction"], float)


def test_inference_validation_error(regressor):
    class Schema(BaseModel):
        a: float
        b: float

    with _make_client(SimpleAPI(validation_model=Schema), regressor) as client:
        response = client.post("/inference", json={"a": "not-a-float", "b": 1.0})
        assert response.status_code == 422


def test_batch_inference(regressor):
    with _make_client(SimpleAPI(), regressor) as client:
        response = client.post(
            "/batch-inference",
            json=[
                {"a": 0.0, "b": 1.0},
                {"a": 1.0, "b": 1.0},
                {"a": 2.0, "b": 0.0},
            ],
        )
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert len(predictions) == 3


def test_predict_proba_on_classifier(classifier):
    with _make_client(SimpleAPI(), classifier) as client:
        response = client.post(
            "/predict-proba",
            json=[{"a": 0.0, "b": 1.0}, {"a": 3.0, "b": 0.0}],
        )
        assert response.status_code == 200
        probabilities = response.json()["probabilities"]
        assert len(probabilities) == 2
        assert all(len(row) == 2 for row in probabilities)


def test_predict_proba_on_regressor_returns_501(regressor):
    with _make_client(SimpleAPI(), regressor) as client:
        response = client.post("/predict-proba", json=[{"a": 0.0, "b": 1.0}])
        assert response.status_code == 501


def test_decision_function_on_regressor_returns_501(regressor):
    with _make_client(SimpleAPI(), regressor) as client:
        response = client.post("/decision-function", json=[{"a": 0.0, "b": 1.0}])
        assert response.status_code == 501


def test_metadata(regressor):
    with _make_client(SimpleAPI(), regressor) as client:
        response = client.get("/metadata")
        assert response.status_code == 200
        body = response.json()
        assert body["model_type"] == "LinearRegression"
        assert "predict" in body["supports"]
        assert body["feature_names"] == ["a", "b"]
        assert body["loaded_at"] is not None


def test_backward_compat_model_path_joblib(tmp_path, regressor, monkeypatch):
    model_file = tmp_path / "model.joblib"
    joblib.dump(regressor, model_file)
    monkeypatch.setenv("MODEL_PATH", str(model_file))

    app = serve(SimpleAPI())
    with TestClient(app) as client:
        response = client.post("/inference", json={"a": 1.0, "b": 1.0})
        assert response.status_code == 200
        assert "prediction" in response.json()
