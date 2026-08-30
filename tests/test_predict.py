import numpy as np
from PIL import Image
from src.predict import predict_image

class FakeModel:
    def predict(self, batch, verbose=0):
        assert batch.shape == (1,224,224,3)
        return np.array([[0.8]], dtype=np.float32)

def test_prediction_utility(tmp_path):
    path = tmp_path/"dog.jpg"
    Image.new("RGB",(64,64),(255,255,255)).save(path)
    result = predict_image(FakeModel(), path)
    assert result["label"] == "dog"
    assert result["dog_probability"] == 0.8
