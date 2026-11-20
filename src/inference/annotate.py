"""Draw the surfer's skeleton, a box colored by the current maneuver and a timeline bar on a video."""
import cv2
import imageio_ffmpeg
import numpy as np

COLORS = {"360": "#2A6FDB", "cutback-frontside": "#E07B39", "roller": "#3BA55C", "take-off": "#9B59B6",
          "none": "#9AA0A6", "unfamiliar": "#4B5563"}
NAMES = {"360": "360", "cutback-frontside": "CUTBACK", "roller": "ROLLER", "take-off": "TAKE-OFF",
         "none": "no maneuver", "unfamiliar": "unfamiliar footage"}

# COCO keypoints: 0 nose, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
BAR_HEIGHT = 40


def bgr(hex_color):
    """'#RRGGBB' -> (B, G, R), the channel order OpenCV draws with."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return b, g, r


def draw_tag(frame, text, x, y, color):
    """Filled label with white text just above (x, y)."""
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    top = max(y - th - 14, 0)
    cv2.rectangle(frame, (x, top), (x + tw + 14, top + th + 14), color, -1)
    cv2.putText(frame, text, (x + 7, top + th + 6), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_skeleton(frame, points, visibility, color, min_visibility=0.5):
    """White limbs between confident keypoints, keypoints as dots in the label color."""
    visible = visibility >= min_visibility
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            cv2.line(frame, tuple(map(int, points[a])), tuple(map(int, points[b])), (255, 255, 255), 2, cv2.LINE_AA)
    for (x, y), ok in zip(points, visible):
        if ok:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)


def timeline_bar(width, centers, labels, t, duration):
    """Strip with one colored span per window (cut halfway between window centers) and a cursor at time t."""
    bar = np.full((BAR_HEIGHT, width, 3), 30, dtype=np.uint8)
    edges = np.concatenate([[0], (centers[1:] + centers[:-1]) / 2, [duration]])
    for k, label in enumerate(labels):
        x0, x1 = (int(edges[k] / duration * width), int(edges[k + 1] / duration * width))
        cv2.rectangle(bar, (x0, 8), (max(x0, x1 - 1), BAR_HEIGHT - 8), bgr(COLORS[label]), -1)
    cursor = int(min(t / duration, 1) * (width - 1))
    cv2.line(bar, (cursor, 0), (cursor, BAR_HEIGHT), (255, 255, 255), 3)
    return bar


def annotate_video(path, out_path, windows, labels, confidence, pose_model, max_width=960):
    """Write an H.264 copy of the video with skeleton, maneuver-colored box and timeline bar.

    `windows`, `labels` and `confidence` come from `label_windows` and `window_labels`; each frame
    takes the label of the window whose center is closest in time.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    centers = ((windows["start_s"] + windows["end_s"]) / 2).to_numpy()

    writer, index = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
        frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]   # H.264 needs even sizes

        t = index / fps
        k = int(np.abs(centers - t).argmin())
        label, color = labels[k], bgr(COLORS[labels[k]])

        # Pose model: person boxes plus 17 keypoints; keep the most confident person, as in detect_surfer
        result = pose_model.predict(frame, classes=[0], conf=0.25, verbose=False)[0]
        if len(result.boxes):
            best = int(result.boxes.conf.argmax())
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[best].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            text = NAMES[label] if label in ("none", "unfamiliar") else f"{NAMES[label]} {confidence[k]:.2f}"
            draw_tag(frame, text, x1, y1, color)
            if result.keypoints is not None and result.keypoints.conf is not None:
                draw_skeleton(frame, result.keypoints.xy[best].cpu().numpy(),
                              result.keypoints.conf[best].cpu().numpy(), color)

        frame = np.vstack([frame, timeline_bar(frame.shape[1], centers, labels, t, duration)])
        if writer is None:
            writer = imageio_ffmpeg.write_frames(str(out_path), (frame.shape[1], frame.shape[0]), fps=fps,
                                                 codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2)
            writer.send(None)   # start the encoder
        writer.send(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        index += 1

    cap.release()
    if writer is not None:
        writer.close()
    return out_path
