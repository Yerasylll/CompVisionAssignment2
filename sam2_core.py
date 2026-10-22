"""Optional Meta SAM 2.1 segmentation, guided by MediaPipe landmarks."""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "sam2.1_hiera_tiny.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def hand_box(landmarks, image_size, padding=0.16):
    """Return clipped XYXY pixel box from normalized MediaPipe landmarks."""
    if not landmarks:
        raise ValueError("A detected hand is needed to prompt SAM 2.1")
    w, h = image_size
    xy = np.asarray([(p.x * w, p.y * h) for p in landmarks], dtype=float)
    x1, y1 = xy.min(axis=0)
    x2, y2 = xy.max(axis=0)
    pad = max(x2 - x1, y2 - y1) * padding
    return np.array(
        [
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(w - 1, x2 + pad),
            min(h - 1, y2 + pad),
        ],
        dtype=np.float32,
    )


def load_predictor(device="cpu"):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "SAM 2.1 checkpoint missing. Run: python download_sam2.py"
        )
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError(
            'Install SAM 2 first: pip install -e "git+https://github.com/facebookresearch/sam2.git#egg=SAM-2"'
        ) from exc
    model = build_sam2(
        CONFIG, str(CHECKPOINT), device=device, apply_postprocessing=False
    )
    return SAM2ImagePredictor(model)


def segment_hand(predictor, image, landmarks):
    """Return binary mask, cutout, overlay, prompt box, and SAM confidence."""
    import torch

    image = image.convert("RGB")
    rgb = np.asarray(image)
    box = hand_box(landmarks, image.size)
    with torch.inference_mode():
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    idx = int(np.argmax(scores))
    mask = np.asarray(masks[idx], dtype=bool)
    cutout = rgb.copy()
    cutout[~mask] = 255
    blended = rgb.astype(np.float32).copy()
    blended[mask] = 0.65 * blended[mask] + 0.35 * np.array(
        [20, 210, 145], dtype=np.float32
    )
    return {
        "mask": Image.fromarray((mask * 255).astype(np.uint8), mode="L"),
        "cutout": Image.fromarray(cutout.astype(np.uint8)),
        "overlay": Image.fromarray(np.uint8(np.clip(blended, 0, 255))),
        "box": box,
        "sam_score": float(scores[idx]),
        "mask_fraction": float(mask.mean()),
    }