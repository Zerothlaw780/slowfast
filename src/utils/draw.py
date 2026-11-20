import cv2
import numpy as np


def draw_actions(frame_bgr: np.ndarray, lines) -> np.ndarray:
    """Writes the given text lines in a dark box at the top-left corner. Font scales with the frame size."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(w, h) / 600)
    thickness = max(1, round(scale * 2))
    pad = round(8 * scale)
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0] for t in lines]
    line_h = max(s[1] for s in sizes) + pad
    box_w = max(s[0] for s in sizes) + 2 * pad
    box_h = line_h * len(lines) + pad

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)
    for i, text in enumerate(lines):
        y = pad + line_h * (i + 1) - pad // 2
        cv2.putText(out, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def result_lines(rows, buffered: int, clip_frames: int):
    if rows[0]["rank"] == 0:
        return [f"SlowFast: warming up {buffered}/{clip_frames}"]
    return [f"{r['rank']}. {r['action']}  {r['score']:.2f}" for r in rows]


def draw_persons(frame_bgr: np.ndarray, detections, header=None) -> np.ndarray:
    """Draws each person's box and its actions (best first) above the box; optional header line at the top-left."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    scale = max(0.4, min(w, h) / 900)
    thickness = max(1, round(scale * 2))
    for det in detections:
        b = det["boundingBox"]
        b = b if isinstance(b, dict) else b.__dict__
        x1, y1 = int(b["left"]), int(b["top"])
        x2, y2 = int(b["left"] + b["width"]), int(b["top"] + b["height"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), thickness)
        lines = [f"{a['action']} {a['score']:.2f}" for a in det.get("actions", [])[:2]]
        y = y1
        for text in reversed(lines):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            y = max(th + 4, y - 4)
            cv2.rectangle(out, (x1, y - th - 4), (x1 + tw + 4, y + 2), (0, 0, 0), -1)
            cv2.putText(out, text, (x1 + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y -= th + 6
    if header:
        out = draw_actions(out, [header])
    return out
