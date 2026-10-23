"""Paired ablation: same test images, MediaPipe original vs SAM-masked input."""
import argparse
import csv
import json
import time
from pathlib import Path

import joblib
from gesture_core import LABELS, analyze, create_recognizer, features, load_image
from PIL import Image
from sam2_core import load_predictor, segment_hand
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", required=True, help="Folder with peace/, stop/, thumbs_up/ subfolders"
    )
    ap.add_argument("--out", default="results/sam2_ablation")
    ap.add_argument(
        "--classifier", choices=["pretrained", "custom"], default="pretrained"
    )
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="0 means all; use small value for smoke test",
    )
    args = ap.parse_args()
    if not 0 <= args.threshold <= 1:
        ap.error("Threshold must be in [0,1]")
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for label in LABELS:
        folder = data / label
        if not folder.is_dir():
            ap.error(f"Missing folder: {folder}")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                items.append((label, p))
    if args.max_images:
        items = items[: args.max_images]
    if not items:
        ap.error("No images found")
    engine = create_recognizer()
    custom = (
        joblib.load(Path(__file__).parent / "models" / "landmark_classifier.joblib")
        if args.classifier == "custom"
        else None
    )

    def classify(lm, scores, label):
        if custom is None:
            return scores, label
        f = features(lm)
        if f is None:
            return {k: 0.0 for k in LABELS}, None
        probs = custom.predict_proba([f])[0]
        scores = dict(zip(custom.classes_, map(float, probs)))
        return scores, max(scores, key=scores.get)

    predictor = load_predictor(args.device)
    rows = []
    for i, (truth, path) in enumerate(items, 1):
        try:
            image = load_image(path)
            t = time.perf_counter()
            image, landmarks, scores, label, _ = analyze(engine, image)
            scores, label = classify(landmarks, scores, label)
            original_ms = (time.perf_counter() - t) * 1000
            original = (
                label if label and scores.get(label, 0) >= args.threshold else "unknown"
            )
            masked = "unknown"
            masked_score = 0.0
            mask_ms = None
            masked_ms = None
            err = ""
            if landmarks:
                t = time.perf_counter()
                seg = segment_hand(predictor, image, landmarks)
                mask_ms = (time.perf_counter() - t) * 1000
                t = time.perf_counter()
                _, _landmarks_masked, mscores, mlabel, _ = analyze(
                    engine, seg["cutout"]
                )
                mscores, mlabel = classify(_landmarks_masked, mscores, mlabel)
                masked_ms = (time.perf_counter() - t) * 1000
                masked_score = mscores.get(mlabel, 0) if mlabel else 0.0
                if mlabel and masked_score >= args.threshold:
                    masked = mlabel
            else:
                err = "no_hand_detected"
            rows.append(
                dict(
                    file=str(path),
                    truth=truth,
                    original=original,
                    masked=masked,
                    original_score=round(scores.get(label, 0) if label else 0, 4),
                    masked_score=round(masked_score, 4),
                    original_ms=round(original_ms, 1),
                    sam_ms=round(mask_ms, 1) if mask_ms is not None else "",
                    masked_ms=round(masked_ms, 1) if masked_ms is not None else "",
                    error=err,
                )
            )
        except Exception as exc:
            rows.append(
                dict(
                    file=str(path),
                    truth=truth,
                    original="unknown",
                    masked="unknown",
                    original_score=0,
                    masked_score=0,
                    original_ms="",
                    sam_ms="",
                    masked_ms="",
                    error=str(exc),
                )
            )
        print(
            f'{i}/{len(items)} {truth} {path.name}: {rows[-1]["original"]} -> {rows[-1]["masked"]}',
            flush=True,
        )
    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    truth = [r["truth"] for r in rows]
    report = {
        "n": len(rows),
        "classifier": args.classifier,
        "threshold": args.threshold,
        "unknown_counts": {},
        "latency_ms_mean": {},
    }
    for key in ("original", "masked"):
        pred = [r[key] for r in rows]
        report[key] = {
            "accuracy": accuracy_score(truth, pred),
            "macro_f1": f1_score(
                truth, pred, labels=LABELS, average="macro", zero_division=0
            ),
            "confusion_matrix_labels": LABELS + ["unknown"],
            "confusion_matrix": confusion_matrix(
                truth, pred, labels=LABELS + ["unknown"]
            ).tolist(),
            "classification_report": classification_report(
                truth, pred, labels=LABELS, output_dict=True, zero_division=0
            ),
        }
        report["unknown_counts"][key] = pred.count("unknown")
    for key in ("original_ms", "sam_ms", "masked_ms"):
        vals = [float(r[key]) for r in rows if r[key] != ""]
        report["latency_ms_mean"][key] = (
            round(sum(vals) / len(vals), 2) if vals else None
        )
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: {
                    "accuracy": report[k]["accuracy"],
                    "macro_f1": report[k]["macro_f1"],
                }
                for k in ("original", "masked")
            },
            indent=2,
        )
    )
    print("Results saved to", out)


if __name__ == "__main__":
    main()
