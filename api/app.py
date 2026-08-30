import logging, time, tempfile, shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from src.predict import load_trained_model, predict_image, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cats-dogs-api")
app = FastAPI(title="Cats vs Dogs MLOps API", version="1.0.0")
_model = None
request_count = 0
total_latency = 0.0

@app.on_event("startup")
def startup():
    global _model
    if MODEL_PATH.exists():
        _model = load_trained_model(MODEL_PATH)
        logger.info("Model loaded")
    else:
        logger.warning("Model artifact not found")

@app.get("/health")
def health():
    return {"status":"healthy","model_loaded":_model is not None}

@app.get("/metrics")
def metrics():
    return {"request_count":request_count,
            "average_latency_seconds":round(total_latency/request_count,6) if request_count else 0.0}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global request_count, total_latency
    if _model is None:
        raise HTTPException(503, "Model is not loaded. Train the model first.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    suffix = Path(file.filename or ".jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp); temp_path = tmp.name
    start = time.perf_counter()
    try:
        result = predict_image(_model, temp_path)
        elapsed = time.perf_counter()-start
        request_count += 1; total_latency += elapsed
        logger.info("Prediction label=%s probability=%.4f latency=%.4fs", result["label"], result["probability"], elapsed)
        result["latency_seconds"] = round(elapsed,6)
        return result
    finally:
        Path(temp_path).unlink(missing_ok=True)
