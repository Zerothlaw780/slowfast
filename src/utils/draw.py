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
