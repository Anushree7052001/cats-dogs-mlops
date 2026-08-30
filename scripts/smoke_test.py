import sys, requests

BASE = "http://localhost:8000"
health = requests.get(BASE + "/health", timeout=10)
health.raise_for_status()
health_data = health.json()
assert health_data.get("model_loaded") is True, health_data
print("HEALTH CHECK PASSED:", health_data)

if len(sys.argv) > 1:
    image = sys.argv[1]
    with open(image, "rb") as f:
        r = requests.post(BASE + "/predict", files={"file": (image, f, "image/jpeg")}, timeout=30)
    r.raise_for_status()
    data = r.json()
    assert data["label"] in {"cat", "dog"}
    assert 0 <= data["cat_probability"] <= 1
    assert 0 <= data["dog_probability"] <= 1
    print("PREDICTION CHECK PASSED:", data)
