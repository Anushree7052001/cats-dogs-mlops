from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocess import preprocess_image_for_model

MODEL_PATH = Path("models/cats_dogs_model.keras")

def load_trained_model(model_path=MODEL_PATH):
    return load_model(model_path)

def predict_image(model, image_path) -> dict:
    batch = preprocess_image_for_model(image_path)
    dog_probability = float(model.predict(batch, verbose=0)[0][0])
    cat_probability = 1.0 - dog_probability
    label = "dog" if dog_probability >= 0.5 else "cat"
    probability = dog_probability if label == "dog" else cat_probability
    return {
        "label": label,
        "probability": round(probability, 6),
        "cat_probability": round(cat_probability, 6),
        "dog_probability": round(dog_probability, 6)
    }
