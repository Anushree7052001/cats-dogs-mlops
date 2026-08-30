from pathlib import Path
from PIL import Image
import numpy as np

IMAGE_SIZE = (224, 224)

def load_and_preprocess_image(path: str | Path) -> np.ndarray:
    """Load an image, convert it to RGB, resize to 224x224, and normalize to [0,1]."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        return np.asarray(image, dtype=np.float32) / 255.0

def preprocess_image_for_model(path: str | Path) -> np.ndarray:
    return np.expand_dims(load_and_preprocess_image(path), axis=0)
