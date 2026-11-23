# app/

Gradio web app: upload a surf video and get the detected maneuvers.

## Run

From the project root, with the environment from the main README installed:

```bash
python app/app.py
```

Then open http://127.0.0.1:7860.

The first start downloads the V-JEPA 2 ViT-L weights from Hugging Face (about 1.3 GB, cached afterwards) and, if they are not already in `notebooks/`, the YOLO26 detection and pose weights. Loading takes a few seconds after that.

## What you get

- **Annotated video:** pose skeleton, a box colored by maneuver and a timeline bar.
- **Probability plot:** the probability of each maneuver over time.
- **Segments table:** start, end, maneuver and confidence.

Controls: the minimum confidence below which a window becomes "no maneuver", and the unfamiliar-footage check (on by default). The examples under the upload box are held-out test clips; they are read from `data/external/demo/`, which is not versioned.

Uses `models/linear_probe_surfer.joblib` and `models/knn_novelty.joblib`. A GPU is recommended: on an RTX 3050 Laptop a 30 s video takes about one minute.
