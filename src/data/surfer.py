"""Locate the surfer in a clip and split the clip into a surfer-only and a scene-only version."""
import numpy as np

PERSON = 0   # COCO class id of "person"


def detect_surfer(frames, detector, conf=0.25):
    """Box (x1, y1, x2, y2) of the most confident person in each frame; NaN rows where nobody is found."""
    # Ultralytics reads numpy images as BGR, our frames are RGB
    bgr = [np.ascontiguousarray(frame[..., ::-1]) for frame in frames]
    results = detector.predict(bgr, classes=[PERSON], conf=conf, verbose=False)
    boxes = np.full((len(frames), 4), np.nan, dtype=np.float32)
    for i, result in enumerate(results):
        if len(result.boxes):
            boxes[i] = result.boxes.xyxy[result.boxes.conf.argmax()].cpu().numpy()
    return boxes


def surfer_region(boxes, frame_shape, margin=0.10, max_jump=3.0):
    """One square region (x1, y1, x2, y2) covering the surfer in every frame, or None if never detected.

    Boxes whose center is more than `max_jump` surfer sizes away from the clip's medoid center are
    treated as other people or false detections and ignored. The same region is used for all frames,
    so the clip keeps the surfer's movement across it.
    """
    found = boxes[~np.isnan(boxes).any(axis=1)]
    if len(found) == 0:
        return None
    centers = (found[:, :2] + found[:, 2:]) / 2
    sizes = np.maximum(found[:, 2] - found[:, 0], found[:, 3] - found[:, 1])
    # Reference point: the medoid, the detected center closest to all the others (always kept)
    pairwise = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    medoid = centers[pairwise.sum(axis=1).argmin()]
    distance = np.linalg.norm(centers - medoid, axis=1)
    found = found[distance <= max_jump * np.median(sizes)]

    height, width = frame_shape[:2]
    x1, y1 = found[:, 0].min(), found[:, 1].min()
    x2, y2 = found[:, 2].max(), found[:, 3].max()
    # Square side: the longer side of the union box plus a margin, at most the frame's shorter side
    side = int(min(max(x2 - x1, y2 - y1) * (1 + 2 * margin), height, width))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # Shift the square to stay inside the frame instead of cutting it
    left = int(np.clip(cx - side / 2, 0, width - side))
    top = int(np.clip(cy - side / 2, 0, height - side))
    return left, top, left + side, top + side


def crop_surfer(frames, region):
    """Surfer-only clip: every frame cut to the region."""
    x1, y1, x2, y2 = region
    return frames[:, y1:y2, x1:x2]


def mask_surfer(frames, region):
    """Scene-only clip: the region painted mid-gray in every frame."""
    x1, y1, x2, y2 = region
    masked = frames.copy()
    masked[:, y1:y2, x1:x2] = 127
    return masked