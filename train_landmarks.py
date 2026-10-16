import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from evaluate import paths
from gesture_core import LABELS, analyze, create_recognizer, features, load_image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def run(root):
    recognizer = create_recognizer()
    X = []
    y = []
    failures = []
    for label, path in paths(root):
        try:
            _, lm, _, _, _ = analyze(recognizer, load_image(path))
            f = features(lm)
            if f is None:
                failures.append(str(path))
                continue
            X.append(f)
            y.append(label)
        except Exception:
            failures.append(str(path))
    if len(X) < 30 or len(set(y)) != 3:
        raise RuntimeError("Too few valid samples; check detection and dataset.")
    X = np.asarray(X)
    y = np.asarray(y)
    xtr, xval, ytr, yval = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    candidates = {
        "SVM_RBF_C1": make_pipeline(
            StandardScaler(), SVC(C=1, probability=True, random_state=42)
        ),
        "SVM_RBF_C10": make_pipeline(
            StandardScaler(), SVC(C=10, probability=True, random_state=42)
        ),
        "MLP_64": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,),
                max_iter=500,
                early_stopping=True,
                random_state=42,
            ),
        ),
    }
    scores = {}
    best = None
    bestscore = -1
    for name, model in candidates.items():
        model.fit(xtr, ytr)
        yp = model.predict(xval)
        score = float(f1_score(yval, yp, average="macro"))
        scores[name] = {
            "validation_macro_f1": score,
            "validation_accuracy": float(accuracy_score(yval, yp)),
        }
        if score > bestscore:
            bestscore = score
            best = name
    # Retrain selected model on all original-session samples; do NOT use new-session images.
    final = candidates[best]
    final.fit(X, y)
    folder = Path(__file__).parent / "models"
    folder.mkdir(exist_ok=True)
    joblib.dump(final, folder / "landmark_classifier.joblib")
    result = {
        "n_valid": len(y),
        "n_failed": len(failures),
        "failures": failures,
        "selection": best,
        "validation": scores,
        "note": "New-session photos must not be used in training or model selection.",
    }
    (folder / "training_summary.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: result[k] for k in ("n_valid", "n_failed", "selection", "validation")},
            indent=2,
        )
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    run(a.parse_args().data)