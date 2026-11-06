# Hand Gesture Lab — Assignment 2 (MediaPipe + Meta SAM 2.1)

Three labels: **peace**, **stop**, **thumbs_up**. This version extends the original MediaPipe app with optional SAM 2.1 hand segmentation, a consent gate, privacy disclosure, side-by-side visual explanation, and a paired ablation evaluation script. The pretrained MediaPipe gesture recognizer is the default because it already worked well in manual tests. **Do not claim measured accuracy for the new model until you run the evaluation scripts.**

## 1. Run the app on macOS

Use **Python 3.11 or 3.12**, not Python 3.14.

```bash
cd repository_name
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_model.py
streamlit run app.py
```

This runs the **MediaPipe-only mode** immediately. The SAM 2.1 checkbox is OFF by default. The user must accept the consent checkbox before upload/camera input is shown.

## 2. Enable Meta SAM 2.1 (optional, large installation)

SAM 2.1 is an image segmentation foundation model, **not** a gesture classifier. This integration uses a bounding box computed from the 21 MediaPipe hand landmarks as the SAM prompt. SAM 2.1 predicts a hand-shaped foreground mask. This makes the explanation visual and allows background-removal experiments.

SAM 2.1 requires PyTorch >=2.5.1 and torchvision >=0.20.1; installation can be demanding, especially on macOS. Start with CPU; Apple Silicon users can try `mps` if their PyTorch/SAM build supports it. Official setup: https://github.com/facebookresearch/sam2

```bash
# In the activated project environment:
pip install -r requirements-sam2.txt
pip install -e "git+https://github.com/facebookresearch/sam2.git#egg=SAM-2"
python download_sam2.py
streamlit run app.py
```

The checkpoint is the official `sam2.1_hiera_tiny.pt` (roughly 150 MB). The model is loaded only when the SAM checkbox is enabled. Initial loading can take time. If `mps` errors, select `cpu` in the sidebar. **Internet is required** for installation and checkpoint downloads. The SAM model weights and MediaPipe task bundle are not included in this source archive.

## 3. Evaluate fairly

Extract the **untouched 120 new-session images** into this structure (40 images in each class):

```
data/new_session/
  peace/*.jpg
  stop/*.jpg
  thumbs_up/*.jpg
```

Run the pretrained recognizer first:

```bash
python evaluate.py --data data/new_session --mode pretrained --out results/pretrained_new_session
```

Then run the paired segmentation ablation (requires SAM 2.1):

```bash
python evaluate_sam_ablation.py --data data/new_session --out results/sam2_ablation --device cpu
```

This writes `predictions.csv` and `metrics.json` with accuracy, macro-F1, confusion matrices, unknown predictions and measured latency. **Every image is evaluated in both conditions.** The SAM-masked images are processed by the same pretrained recognizer without retraining. Improvement is a hypothesis, not a promise. SAM masks can be imperfect; their segmentation IoU/Dice cannot be measured without ground-truth masks.

To train your own optional landmark classifier on the **original 360 images**, follow `train_landmarks.py --help`. Keep the new-session 120 images held out for evaluation. Training curves for the old neural-network model belong to the old training experiment, not to this pretrained SAM/MediaPipe inference pipeline.

## 4. Consent, privacy, and ethical limitations

The app requires the user to attest they own the image or have permission before upload/camera capture. Images are processed in the Streamlit Python server process; the app does **not** explicitly save uploaded images, store them in a dataset, or send them to a separate cloud inference API. **If hosted online, images necessarily transit to the hosting server.** No guarantee is made about infrastructure logs, browser history, or third-party hosting retention. For the clearest local-only setup, run Streamlit on your own computer and avoid exposing it publicly. Dataset participants should provide informed permission for collection and reuse; the UI checkbox does not retroactively establish dataset consent. Delete datasets if you lack the needed permissions. Do not use this prototype for high-stakes decisions.

## 5. Academic framing

- **MediaPipe Gesture Recognizer**: Google's pretrained gesture recognition model, which also returns 21 hand landmarks. It is a modern computer vision framework; **do not claim it is a proven SOTA benchmark winner** without evidence.
- **Meta SAM 2.1**: promptable segmentation foundation model, used here for hand masks and background-isolation ablation. The model itself is not trained on your photos in this repository.
- **Original A1 baseline**: previously reported frozen ResNet-18 feature extraction + logistic regression; its original test-set score is **not directly comparable** with a new-session test score. Re-evaluate on identical held-out images for a fair comparison.
- **Your contribution**: system integration, gesture mapping, consent UI, landmark-to-box prompting, visualization, paired ablation and evaluation. Distinguish this work from pretrained model development.

## Sources

- Google MediaPipe Python: https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python
- Meta SAM 2.1 official: https://github.com/facebookresearch/sam2


## NEW: Train our own MLP with real learning curves

**Do this on your Mac in the SAME activated virtual environment as Streamlit.**

1. Put the original 360 images into `data/original/peace/`, `data/original/stop/`, and `data/original/thumbs_up/` (the ZIP has a `dataset/` folder; rename that folder to `original`). **Do not put new-session photos in this training folder.**
2. Run:

```bash
python -m pip install -r requirements.txt
python download_model.py
python train_custom_mlp.py --data data/original --epochs 100 --patience 20
```

The script extracts 21 hand landmarks, makes stratified train/validation/held-out-test splits (70/15/15 of detected samples), trains our **own** MLP with per-epoch measurements, selects the best epoch using validation F1/loss, and evaluates the original-session test **once**. It writes:

- `models/landmark_classifier.joblib` (used by the **Custom landmark model** radio button in the app)
- `results/custom_mlp/history.csv`
- `results/custom_mlp/accuracy_curves.png`, `loss_curves.png`, `f1_curves.png`
- `results/custom_mlp/confusion_matrix.png`, `metrics.json`, `split_manifest.csv`, `failed_images.txt`

The 120 new-session photos are a separate **untouched** generalization test. Put them in `data/new_session/{peace,stop,thumbs_up}` and run:

```bash
python evaluate.py --data data/new_session --mode pretrained --out results/new_pretrained
python evaluate.py --data data/new_session --mode custom --out results/new_custom
python evaluate_sam_ablation.py --data data/new_session --classifier pretrained --out results/ablation_pretrained
python evaluate_sam_ablation.py --data data/new_session --classifier custom --out results/ablation_custom
```

SAM 2.1 must be installed and checkpoint downloaded for the ablation scripts. Each SAM ablation uses **the same image** before/after segmentation. It is a background-removal ablation, not a segmentation-IoU test. Results are saved locally; no accuracy values are pre-filled or fabricated.

**Scientific limitations:** Photo-level random splits from one capture session are correlated, so the independent new-session results are more credible. MediaPipe and SAM 2.1 remain pretrained; only the landmark MLP is trained by us. Neither model's pretraining curves are available. SAM's mask score is not IoU or Dice. Do not publish identifiable face photos without explicit permission.
