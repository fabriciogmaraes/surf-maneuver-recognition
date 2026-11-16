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