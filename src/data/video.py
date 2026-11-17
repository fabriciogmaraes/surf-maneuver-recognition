"""Read frames from video clips with OpenCV."""
from pathlib import Path

import cv2
import numpy as np


def read_frame(path: Path, position: float = 0.5) -> np.ndarray | None:
    """Return the frame at a relative position of the clip as an RGB array.

    position is 0 for the first frame, 0.5 for the middle and 1 for the last.
    Returns None if the frame cannot be read.
    """
    cap = cv2.VideoCapture(str(path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(position * (n_frames - 1)))
    ok, frame = cap.read()
    cap.release()
    # OpenCV reads BGR; models and matplotlib expect RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None

def read_clip(path: Path, num_frames: int = 16) -> np.ndarray:
    """Return `num_frames` evenly spaced RGB frames of a clip, shape (T, H, W, 3), dtype uint8.

    Frames are read sequentially and sampled afterwards: seeking frame by frame can land on
    the wrong frame in compressed videos, and the declared frame count is not always exact.
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < num_frames:
        raise ValueError(f"{path} has only {len(frames)} frames, {num_frames} needed")

    indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    return np.stack([cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in indices])