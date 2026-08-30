import numpy as np
from PIL import Image
from src.preprocess import load_and_preprocess_image

def test_preprocess_returns_224_rgb(tmp_path):
    path = tmp_path/"sample.png"
    Image.new("RGB",(80,120),(100,150,200)).save(path)
    result = load_and_preprocess_image(path)
    assert result.shape == (224,224,3)
    assert result.dtype == np.float32
    assert 0.0 <= result.min() <= result.max() <= 1.0
