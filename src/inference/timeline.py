"""Slide a window over an unseen video and label each window with a maneuver."""
import cv2
import numpy as np
import pandas as pd

from src.data.surfer import crop_surfer, detect_surfer, surfer_region


def read_video(path, sample_fps, max_side=640):
    """Frames of a video sampled at about `sample_fps`, RGB, resized so the longer side is at most `max_side`.

    Returns (frames, effective_fps). Sampling while reading keeps long videos within memory.
    """
    cap = cv2.VideoCapture(str(path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native_fps / sample_fps))
    frames, index = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            h, w = frame.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        index += 1
    cap.release()
    return np.stack(frames), native_fps / step


def label_windows(frames, fps, window_s, stride_s, detector, encoder, probe):
    """Class probabilities (and encoder features) for every window; NaN where no surfer is found."""
    n_window = min(len(frames), max(16, round(window_s * fps)))
    n_stride = max(1, round(stride_s * fps))
    rows = []
    for start in range(0, len(frames) - n_window + 1, n_stride):
        # 16 evenly spaced frames, the input size of the encoder
        clip = frames[np.linspace(start, start + n_window - 1, 16).round().astype(int)]
        boxes = detect_surfer(clip, detector)
        region = surfer_region(boxes, clip.shape[1:])
        row = {"start_s": start / fps, "end_s": (start + n_window) / fps,
               "surfer_frames": int((~np.isnan(boxes).any(axis=1)).sum())}
        if region is not None:
            features = encoder(crop_surfer(clip, region)).mean(axis=0, keepdims=True)   # (1, 1024)
            row.update(zip(probe.classes_, probe.predict_proba(features)[0]))
            row["features"] = features[0]   # kept for out-of-distribution checks
        rows.append(row)
    return pd.DataFrame(rows)


def to_segments(windows, classes, min_confidence=0.6, min_surfer_frames=8):
    """Merge consecutive windows with the same label into segments.

    A window is labeled "none" when the surfer is found in fewer than `min_surfer_frames` of its
    16 frames or when the top class probability is below `min_confidence`: the training data has
    no "no maneuver" class, so low confidence is the only signal that nothing known is happening.
    """
    probs = windows.reindex(columns=classes).to_numpy(dtype=float)
    confidence = np.nan_to_num(probs, nan=0.0).max(axis=1)
    labels = np.array(classes)[np.nan_to_num(probs, nan=-1.0).argmax(axis=1)].astype(object)
    labels[(confidence < min_confidence) | (windows["surfer_frames"] < min_surfer_frames)] = "none"

    segments = []
    for i, label in enumerate(labels):
        if segments and segments[-1]["label"] == label:
            segments[-1]["end_s"] = windows["end_s"].iloc[i]
            segments[-1]["confidences"].append(confidence[i])
        else:
            segments.append({"label": label, "start_s": windows["start_s"].iloc[i],
                             "end_s": windows["end_s"].iloc[i], "confidences": [confidence[i]]})
    segments = pd.DataFrame(segments)
    segments["confidence"] = segments.pop("confidences").map(np.mean).round(2)
    return segments