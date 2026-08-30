from fastapi.testclient import TestClient
import api.app as app_module


def test_health_without_model():
    original = app_module._model
    app_module._model = None
    try:
        client = TestClient(app_module.app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is False
    finally:
        app_module._model = original
