import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from gesture_core import LABELS, analyze, create_recognizer, features, load_image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def paths(root):
    for label in LABELS:
        folder = root / label
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                yield label, f


def evaluate(root, mode, output):
    recognizer = create_recognizer()
    custom = (
        joblib.load(Path(__file__).parent / "models" / "landmark_classifier.joblib")
        if mode == "custom"
        else None
    )
    y, pred, rows = [], [], []
    for label, path in paths(root):
        try:
            t = time.perf_counter()
            _, lm, scores, guess, _ = analyze(recognizer, load_image(path))
            if custom is not None and lm:
                f = features(lm)
                if f is not None:
                    probs = custom.predict_proba([f])[0]
                    scores = dict(zip(custom.classes_, map(float, probs)))
                    guess = max(scores, key=scores.get)
            if custom is not None and (not lm or features(lm) is None):
                guess = None
                scores = {k: 0.0 for k in LABELS}
            ms = 1000 * (time.perf_counter() - t)
            guess = guess or "no_prediction"
            y.append(label)
            pred.append(guess)
            rows.append(
                {
                    "path": str(path),
                    "true": label,
                    "predicted": guess,
                    "score": max(scores.values(), default=0),
                    "latency_ms": round(ms, 2),
                    "hand_detected": bool(lm),
                }
            )
        except Exception as exc:
            y.append(label)
            pred.append("error")
            rows.append(
                {
                    "path": str(path),
                    "true": label,
                    "predicted": "error",
                    "error": str(exc),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="") as f:
        fields = [
            "path",
            "true",
            "predicted",
            "score",
            "latency_ms",
            "hand_detected",
            "error",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary = {
        "mode": mode,
        "n": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)
        ),
        "confusion_labels": LABELS + ["no_prediction", "error"],
        "confusion_matrix": confusion_matrix(
            y, pred, labels=LABELS + ["no_prediction", "error"]
        ).tolist(),
        "classification_report": classification_report(
            y, pred, labels=LABELS, zero_division=0, output_dict=True
        ),
        "no_hand_or_unsupported": sum(z in ("no_prediction", "error") for z in pred),
        "avg_latency_ms": float(
            np.mean([r["latency_ms"] for r in rows if "latency_ms" in r])
        )
        if rows
        else None,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "n",
                    "accuracy",
                    "macro_f1",
                    "no_hand_or_unsupported",
                    "avg_latency_ms",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--mode", choices=["pretrained", "custom"], default="pretrained")
    a.add_argument("--out", type=Path, default=Path("results/new_session"))
    x = a.parse_args()
    evaluate(x.data, x.mode, x.out)
