"""Train an original landmark MLP with reproducible train/validation/test curves.

Original-session images only. New-session photos are NEVER used in fitting or tuning.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evaluate import paths
from gesture_core import LABELS, analyze, create_recognizer, features, load_image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def collect(root, cache):
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        return data["X"], data["y"], data["paths"].tolist(), data["failed"].tolist()
    engine = create_recognizer()
    X = []
    y = []
    names = []
    failed = []
    for label, path in paths(root):
        try:
            _, lm, _, _, _ = analyze(engine, load_image(path))
            f = features(lm)
            if f is None:
                raise ValueError("No usable hand landmarks")
            X.append(f)
            y.append(label)
            names.append(str(path))
        except Exception as e:
            failed.append(f"{path}: {e}")
    if not X:
        raise RuntimeError("No hand landmarks extracted")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, X=X, y=y, paths=np.asarray(names), failed=np.asarray(failed, dtype=str)
    )
    return X, y, names, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/custom_mlp"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--cache", type=Path, default=None)
    a = ap.parse_args()
    if a.epochs < 2 or a.patience < 1:
        ap.error("epochs >=2 and patience >=1 required")
    random.seed(a.seed)
    np.random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.cache or a.out / "landmarks_original_session.npz"
    X, y, names, failed = collect(a.data, cache)
    counts = {k: int((y == k).sum()) for k in LABELS}
    if min(counts.values()) < 10:
        raise RuntimeError(f"Not enough detected examples per class: {counts}")
    # 70% train, 15% validation, 15% held-out test (stratified).
    indices = np.arange(len(y))
    tr, tmp = train_test_split(indices, test_size=0.30, random_state=a.seed, stratify=y)
    val, te = train_test_split(tmp, test_size=0.5, random_state=a.seed, stratify=y[tmp])
    scaler = StandardScaler().fit(X[tr])
    xtr = scaler.transform(X[tr])
    xval = scaler.transform(X[val])
    xte = scaler.transform(X[te])
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1,
        warm_start=False,
        random_state=a.seed,
    )
    history = []
    best = (-1.0, float("-inf"))
    best_epoch = 0
    best_state = None
    for epoch in range(1, a.epochs + 1):
        clf.partial_fit(xtr, y[tr], classes=np.asarray(LABELS))
        train_probs = clf.predict_proba(xtr)
        val_probs = clf.predict_proba(xval)
        tr_pred = clf.predict(xtr)
        val_pred = clf.predict(xval)
        row = {
            "epoch": epoch,
            "train_loss": float(log_loss(y[tr], train_probs, labels=clf.classes_)),
            "val_loss": float(log_loss(y[val], val_probs, labels=clf.classes_)),
            "train_accuracy": float(accuracy_score(y[tr], tr_pred)),
            "val_accuracy": float(accuracy_score(y[val], val_pred)),
            "train_macro_f1": float(
                f1_score(
                    y[tr], tr_pred, labels=LABELS, average="macro", zero_division=0
                )
            ),
            "val_macro_f1": float(
                f1_score(
                    y[val], val_pred, labels=LABELS, average="macro", zero_division=0
                )
            ),
        }
        history.append(row)
        # Best validation F1, with validation loss as tie-breaker.
        criterion = (row["val_macro_f1"], -row["val_loss"])
        if best_state is None or criterion > best:
            best = criterion
            best_epoch = epoch
            import copy

            best_state = copy.deepcopy(clf)
        if epoch - best_epoch >= a.patience:
            break
    model = Pipeline([("scaler", scaler), ("mlp", best_state)])
    pred = model.predict(X[te])
    p = model.predict_proba(X[te])
    test_true = y[te]
    metrics = {
        "training_samples": len(tr),
        "validation_samples": len(val),
        "heldout_test_samples": len(te),
        "detected_total": len(y),
        "failed_detection_count": len(failed),
        "class_counts": counts,
        "seed": a.seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "validation_accuracy_at_best": history[best_epoch - 1]["val_accuracy"],
        "validation_macro_f1_at_best": history[best_epoch - 1]["val_macro_f1"],
        "heldout_test_accuracy": float(accuracy_score(test_true, pred)),
        "heldout_test_macro_f1": float(
            f1_score(test_true, pred, labels=LABELS, average="macro", zero_division=0)
        ),
        "heldout_test_loss": float(log_loss(test_true, p, labels=model.classes_)),
        "confusion_labels": LABELS,
        "confusion_matrix": confusion_matrix(test_true, pred, labels=LABELS).tolist(),
        "classification_report": classification_report(
            test_true, pred, labels=LABELS, zero_division=0, output_dict=True
        ),
        "method": "MediaPipe pretrained landmarks + own supervised MLP; no SAM used for fitting",
        "split_note": "Image-level stratified split; correlated same-session photos can inflate performance. Report independent new-session evaluation separately.",
    }
    with (a.out / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (a.out / "failed_images.txt").write_text("\n".join(failed))
    with (a.out / "split_manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "path", "label"])
        for label, ids in [("train", tr), ("validation", val), ("test", te)]:
            for i in ids:
                w.writerow([label, names[i], y[i]])
    for title, keys, filename, ylabel in [
        (
            "Accuracy",
            ["train_accuracy", "val_accuracy"],
            "accuracy_curves.png",
            "Accuracy",
        ),
        ("Loss", ["train_loss", "val_loss"], "loss_curves.png", "Cross-entropy loss"),
        ("Macro F1", ["train_macro_f1", "val_macro_f1"], "f1_curves.png", "Macro F1"),
    ]:
        plt.figure(figsize=(8, 5))
        for key in keys:
            plt.plot(
                [r["epoch"] for r in history],
                [r[key] for r in history],
                label=key.replace("_", " ").title(),
            )
        plt.axvline(
            best_epoch,
            linestyle="--",
            color="gray",
            label=f"Best validation epoch {best_epoch}",
        )
        plt.title("Custom MLP — " + title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(a.out / filename, dpi=160)
        plt.close()
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array(metrics["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), [x.replace("_", " ") for x in LABELS])
    ax.set_yticks(range(3), [x.replace("_", " ") for x in LABELS])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Held-out original-session test")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(a.out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "landmark_classifier.joblib")
    print(
        json.dumps(
            {
                "saved_model": str(model_dir / "landmark_classifier.joblib"),
                "results": str(a.out),
                "best_epoch": best_epoch,
                "heldout_test_accuracy": metrics["heldout_test_accuracy"],
                "heldout_test_macro_f1": metrics["heldout_test_macro_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
