"""Post-deployment performance check using labelled image paths."""
import csv, json, sys
from pathlib import Path
from src.predict import load_trained_model, predict_image

if len(sys.argv) != 2:
    raise SystemExit("Usage: python scripts/performance_test.py path/to/labels.csv")

rows = list(csv.DictReader(Path(sys.argv[1]).open()))
model = load_trained_model()
correct = 0
for row in rows:
    result = predict_image(model, row["path"])
    correct += int(result["label"] == row["true_label"].lower())
accuracy = correct / len(rows) if rows else 0.0
out = {"samples": len(rows), "correct": correct, "accuracy": accuracy}
Path("artifacts/post_deployment_performance.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
