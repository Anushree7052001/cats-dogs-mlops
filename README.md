# MLOps Assignment 2 — Cats vs Dogs

End-to-end MLOps pipeline for the Assignment 2 requirements: Git/DVC → preprocessing → baseline CNN → MLflow → FastAPI → Docker → pytest → GitHub Actions → Docker Hub → Docker Compose/host deployment → smoke test → monitoring.

## 1. Dataset
Download the Kaggle Cats and Dogs classification dataset and place the two class folders under `data/raw/` (for example `cats/` and `dogs/`). Do not commit the dataset to Git; use DVC.

## 2. Local setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. DVC
```bash
dvc init
dvc add data/raw
git add .dvc .gitignore data/raw.dvc
git commit -m "Add dataset with DVC"
```

The training stage creates reproducible `train.csv`, `validation.csv`, and `test.csv` split manifests under `data/processed/`, plus the trained model and metrics.

## 4. Train and track with MLflow/DVC
```bash
dvc repro
mlflow ui --host 127.0.0.1 --port 5000
```

Training uses the values in `params.yaml`, 224×224 RGB images, an 80/10/10 split, and augmentation in the CNN. It produces the model, metrics, loss curve, accuracy curve, and confusion matrix.

## 5. Test and serve
```bash
pytest -q
uvicorn api.app:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
python scripts/smoke_test.py path/to/cat_or_dog.jpg
curl http://localhost:8000/metrics
```

## 6. Docker
```bash
docker build -t cats-dogs-mlops:latest .
docker run --rm -p 8000:8000 cats-dogs-mlops:latest
```

For the image to serve predictions, `models/cats_dogs_model.keras` must exist before the Docker build (or be downloaded into the image through your chosen artifact workflow).

## 7. Docker Compose deployment
```bash
IMAGE_NAME=<dockerhub-user>/cats-dogs-mlops:latest docker compose -f deployment/docker-compose.yml up -d
python scripts/smoke_test.py path/to/cat_or_dog.jpg
```

## 8. CI/CD
Configure these GitHub repository secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `DEPLOY_HOST`
- `DEPLOY_USER`
- `DEPLOY_SSH_KEY`

A push to `main` runs tests, builds the Docker image, pushes it to Docker Hub, pulls the new image on the deployment host, replaces the running container, and fails if `/health` does not pass.

## 9. Post-deployment performance
Create a small CSV with columns `path,true_label`, then run:
```bash
python scripts/performance_test.py path/to/labels.csv
```
This writes `artifacts/post_deployment_performance.json` with sample count and accuracy.

## Submission evidence
Show Git/DVC, training, MLflow, pytest, FastAPI health/prediction, Docker build/run, GitHub Actions, Docker Hub, deployment, smoke test, request/latency metrics, and post-deployment performance. The final recording must be under 5 minutes.
