# Train and evaluate — quick start

Activate `.venv` in the project folder. Install: `python -m pip install -r requirements.txt`.

Extract `gesture_dataset.zip` and rename its `dataset` directory to `data/original`.

Run `python train_custom_mlp.py --data data/original`.

Open `results/custom_mlp/accuracy_curves.png`, `loss_curves.png`, `f1_curves.png`, `metrics.json`.

Restart `streamlit run app.py` and choose **Custom landmark model** in sidebar.

Extract `new_gestures_converted_jpeg.zip` and rename `new_session_jpeg` to `data/new_session`.

Run `python evaluate.py --data data/new_session --mode custom --out results/new_custom`.

Run `python evaluate.py --data data/new_session --mode pretrained --out results/new_pretrained`.

For SAM ablation: `python evaluate_sam_ablation.py --data data/new_session --classifier custom --out results/ablation_custom`.

Do not train on the new-session photos. Keep all raw personal photos outside GitHub.
