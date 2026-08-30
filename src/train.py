from pathlib import Path
import json, random
import numpy as np
import tensorflow as tf
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.model import build_baseline_cnn

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"
PARAMS_FILE = ROOT / "params.yaml"
for d in (PROCESSED, MODEL_DIR, ARTIFACT_DIR):
    d.mkdir(exist_ok=True)


def load_params():
    # Keep the project dependency-light: params.yaml uses simple key/value pairs.
    params = {}
    for line in PARAMS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = [x.strip() for x in line.split(":", 1)]
        if value.startswith(('"', "'")) and value.endswith(('"', "'")):
            value = value[1:-1]
        else:
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass
        params[key] = value
    required = {"image_size", "batch_size", "epochs", "learning_rate", "seed"}
    missing = required - params.keys()
    if missing:
        raise RuntimeError(f"Missing parameters in params.yaml: {sorted(missing)}")
    return params


def discover_dataset():
    folders = sorted([p for p in RAW.iterdir() if p.is_dir()]) if RAW.exists() else []
    if len(folders) < 2:
        raise RuntimeError("Put the Kaggle Cats/Dogs dataset in data/raw with two class folders.")
    return folders[:2]


def main():
    params = load_params()
    image_size = int(params["image_size"])
    batch_size = int(params["batch_size"])
    epochs = int(params["epochs"])
    learning_rate = float(params["learning_rate"])
    seed = int(params["seed"])
    img_size = (image_size, image_size)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    classes = discover_dataset()
    class_names = [p.name.lower() for p in classes]
    images, labels = [], []
    valid = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for label, folder in enumerate(classes):
        for p in folder.rglob("*"):
            if p.suffix.lower() in valid:
                images.append(str(p))
                labels.append(label)

    if len(images) < 20:
        raise RuntimeError("Not enough images found.")

    idx = np.arange(len(images))
    np.random.default_rng(seed).shuffle(idx)
    images, labels = np.array(images)[idx], np.array(labels)[idx]

    n = len(images)
    nt = int(0.8 * n)
    nv = int(0.1 * n)
    train_x, train_y = images[:nt], labels[:nt]
    val_x, val_y = images[nt:nt + nv], labels[nt:nt + nv]
    test_x, test_y = images[nt + nv:], labels[nt + nv:]

    # Persist split manifests so the data split is reproducible and versionable.
    for name, xs, ys in (("train", train_x, train_y), ("validation", val_x, val_y), ("test", test_x, test_y)):
        lines = [f"path,label\n"] + [f"{x},{int(y)}\n" for x, y in zip(xs, ys)]
        (PROCESSED / f"{name}.csv").write_text("".join(lines))

    def decode(path, label):
        x = tf.io.read_file(path)
        x = tf.image.decode_image(x, channels=3, expand_animations=False)
        x.set_shape([None, None, 3])
        x = tf.image.resize(x, img_size)
        return tf.cast(x, tf.float32) / 255.0, tf.cast(label, tf.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).map(decode, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y)).map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_baseline_cnn(input_shape=(image_size, image_size, 3), learning_rate=learning_rate)
    mlflow.set_experiment("cats-dogs-classification")

    with mlflow.start_run():
        mlflow.log_params({
            "image_size": f"{image_size}x{image_size} RGB",
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "split": "80/10/10",
            "model": "baseline CNN",
            "augmentation": "RandomFlip + RandomRotation",
        })
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        loss, accuracy = model.evaluate(test_ds, verbose=0)
        mlflow.log_metrics({"test_loss": float(loss), "test_accuracy": float(accuracy)})

        plt.figure()
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="validation")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        loss_path = ARTIFACT_DIR / "loss_curve.png"
        plt.savefig(loss_path, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(history.history["accuracy"], label="train")
        plt.plot(history.history["val_accuracy"], label="validation")
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        acc_path = ARTIFACT_DIR / "accuracy_curve.png"
        plt.savefig(acc_path, bbox_inches="tight")
        plt.close()

        probs = model.predict(test_ds, verbose=0).ravel()
        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(test_y, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot()
        cm_path = ARTIFACT_DIR / "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        model_path = MODEL_DIR / "cats_dogs_model.keras"
        model.save(model_path)
        metrics = {
            "test_loss": float(loss),
            "test_accuracy": float(accuracy),
            "classes": class_names,
            "train_images": int(len(train_x)),
            "validation_images": int(len(val_x)),
            "test_images": int(len(test_x)),
            "split": "80/10/10",
            "image_size": image_size,
        }
        metric_path = ARTIFACT_DIR / "metrics.json"
        metric_path.write_text(json.dumps(metrics, indent=2))

        for artifact in [loss_path, acc_path, cm_path, metric_path, model_path]:
            mlflow.log_artifact(str(artifact))
        print(json.dumps(metrics, indent=2))
        print("MODEL:", model_path)


if __name__ == "__main__":
    main()
