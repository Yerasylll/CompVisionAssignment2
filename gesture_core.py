from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).parent
LABELS = ["peace", "stop", "thumbs_up"]
MAPPING = {"Victory": "peace", "Open_Palm": "stop", "Thumb_Up": "thumbs_up"}
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def create_recognizer():
    path = ROOT / "models" / "gesture_recognizer.task"
    if not path.exists():
        raise FileNotFoundError("Run: python download_model.py")
    opts = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
    )
    return vision.GestureRecognizer.create_from_options(opts)


def load_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def analyze(recognizer, image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    # limit memory for large HEIC/JPEG files, retain aspect ratio
    image.thumbnail((1280, 1280))
    arr = np.asarray(image)
    result = recognizer.recognize(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr))
    )
    landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
    probs = {k: 0.0 for k in LABELS}
    if result.gestures:
        for cat in result.gestures[0]:
            key = MAPPING.get(cat.category_name)
            if key:
                probs[key] = float(cat.score)
    label = max(probs, key=probs.get) if probs and max(probs.values()) > 0 else None
    return image, landmarks, probs, label, result


def features(landmarks):
    if not landmarks:
        return None
    a = np.array([[x.x, x.y, x.z] for x in landmarks], dtype=np.float32)
    a -= a[0]
    scale = np.linalg.norm(a[9, :2])
    if scale < 1e-6:
        return None
    a /= scale
    return a.ravel()


def overlay(image, landmarks):
    out = image.copy()
    if landmarks is None:
        return out
    draw = ImageDraw.Draw(out)
    w, h = out.size
    xy = [(int(x.x * w), int(x.y * h)) for x in landmarks]
    for i, j in EDGES:
        draw.line((xy[i], xy[j]), fill=(0, 255, 200), width=max(2, w // 200))
    for x, y in xy:
        r = max(3, w // 180)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 90, 40))
    return out