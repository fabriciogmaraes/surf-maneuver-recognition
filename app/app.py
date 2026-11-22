"""Gradio demo: upload a surf video, get an annotated video and the maneuver timeline.

Run from the project root:  python app/app.py
"""
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: F401  (imported before ultralytics so it is fully initialized first)
import gradio as gr
import joblib
import matplotlib

matplotlib.use("Agg")   # draw figures without a window
import matplotlib.pyplot as plt
from ultralytics import YOLO

from src.inference.annotate import COLORS, NAMES, annotate_video
from src.inference.novelty import add_knn_distance
from src.inference.timeline import label_windows, read_video, to_segments, window_labels
from src.models.encoder import VJEPA2Encoder

WINDOW_S, STRIDE_S, SAMPLE_FPS = 2.8, 0.7, 12   # same settings as notebooks 10 and 11
MIN_SURFER_FRAMES = 8



def yolo_weights(name):
    """Reuse the weights the notebooks already downloaded; otherwise Ultralytics downloads them here."""
    local = PROJECT_ROOT / "notebooks" / name
    return str(local) if local.exists() else name


# Everything is loaded once at start-up (the V-JEPA 2 encoder takes a few seconds)
detector = YOLO(yolo_weights("yolo26s.pt"))
pose_model = YOLO(yolo_weights("yolo26s-pose.pt"))
encoder = VJEPA2Encoder()
probe = joblib.load(PROJECT_ROOT / "models" / "linear_probe_surfer.joblib")
novelty = joblib.load(PROJECT_ROOT / "models" / "knn_novelty.joblib")
CLASSES = list(probe.classes_)


def timeline_figure(windows, labels):
    """Class probabilities over time, with the window labels as a colored band at the bottom."""
    centers = (windows["start_s"] + windows["end_s"]) / 2
    probs = windows.reindex(columns=CLASSES)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for c in CLASSES:
        ax.plot(centers, probs[c], color=COLORS[c], lw=1.8, label=NAMES[c].lower())
    # Same split as the segments table: each window covers the time closest to its center
    edges = [0.0, *((centers.to_numpy()[1:] + centers.to_numpy()[:-1]) / 2), windows["end_s"].iloc[-1]]
    for start, end, label in zip(edges[:-1], edges[1:], labels):
        ax.axvspan(start, end, ymin=0, ymax=0.06, color=COLORS[label], lw=0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("probability")
    ax.spines[["top", "right"]].set_visible(False)
    # Legend above the axes so it never covers the curves
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0))
    fig.tight_layout()
    return fig


def analyze(video_path, min_confidence, flag_unfamiliar, progress=gr.Progress()):
    """Full pipeline: windows -> surfer crop -> V-JEPA 2 -> probe -> labels -> annotated video."""
    if video_path is None:
        raise gr.Error("Upload a video first.")

    progress(0.05, desc="Reading the video")
    frames, fps = read_video(video_path, SAMPLE_FPS)

    progress(0.15, desc="Finding the surfer and classifying each window")
    windows = label_windows(frames, fps, WINDOW_S, STRIDE_S, detector, encoder, probe)
    threshold = None
    if flag_unfamiliar:
        windows = add_knn_distance(windows, novelty)
        threshold = novelty.threshold_
    labels, confidence = window_labels(windows, CLASSES, min_confidence, MIN_SURFER_FRAMES, threshold)
    segments = to_segments(windows, CLASSES, min_confidence, MIN_SURFER_FRAMES, threshold)

    progress(0.6, desc="Drawing skeleton, boxes and timeline")
    out_path = Path(tempfile.mkdtemp()) / "annotated.mp4"
    annotate_video(video_path, out_path, windows, labels, confidence, pose_model)

    table = segments.assign(
        start_s=segments["start_s"].round(1),
        end_s=segments["end_s"].round(1),
        maneuver=segments["label"].map(NAMES),
    )[["start_s", "end_s", "maneuver", "confidence"]]
    return str(out_path), timeline_figure(windows, labels), table


legend = " · ".join(f"<span style='color:{COLORS[k]}'>■</span> {NAMES[k]}" for k in COLORS)
# Held-out test clips (unseen sessions): three per maneuver, as one montage and one by one
DEMO_DIR = PROJECT_ROOT / "data" / "external" / "demo"
examples = [p for p in [DEMO_DIR / name for name in (
    "montage_4_maneuvers.mp4",
    "takeoff_a.mp4", "cutback_a.mp4", "roller_a.mp4", "360_a.mp4",
    "takeoff_b.mp4", "cutback_b.mp4", "roller_b.mp4", "360_b.mp4",
    "takeoff_c.mp4", "cutback_c.mp4", "roller_c.mp4", "360_c.mp4")] if p.exists()]

with gr.Blocks(title="Surf Maneuver Recognition") as demo:
    gr.Markdown(
        "# 🏄 Surf Maneuver Recognition\n"
        "Upload a surf video. A window slides over it; in each window YOLO26 finds the surfer, "
        "the crop is encoded by V-JEPA 2 and a linear probe names the maneuver "
        "(take-off, cutback, roller or 360). Trained on side-view contest footage.\n\n" + legend
    )
    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Surf video", sources=["upload"], height=300)
            min_confidence = gr.Slider(0.3, 0.95, value=0.6, step=0.05, label="Minimum confidence")
            flag_unfamiliar = gr.Checkbox(value=True, label="Flag footage unlike the training data")
            run = gr.Button("Analyze video", variant="primary")
            if examples:
                gr.Examples([[str(p)] for p in examples], inputs=video_in, label="Examples")
        with gr.Column(scale=2):
            video_out = gr.Video(label="Annotated video", height=420)
            plot_out = gr.Plot(label="Timeline")
            table_out = gr.Dataframe(label="Segments", headers=["start_s", "end_s", "maneuver", "confidence"])

    run.click(analyze, [video_in, min_confidence, flag_unfamiliar], [video_out, plot_out, table_out])

if __name__ == "__main__":
    demo.launch()
