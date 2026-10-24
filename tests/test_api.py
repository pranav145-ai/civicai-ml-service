from fastapi.testclient import TestClient
from app.main import app


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "civicai-ml-service"


def test_model_info_endpoint():
    response = TestClient(app).get("/model/info")
    assert response.status_code == 200
    assert response.json()["category_model"] == "TF-IDF + LinearSVC"
