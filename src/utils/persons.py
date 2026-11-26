"""Helpers for the Person Actions (AVA) executor: person boxes in, actions per person out."""

import numpy as np

from .labels import AVA


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def to_dict(det):
    """Detection (dict or model) -> plain dict, so it can be passed on with extra fields."""
    if isinstance(det, dict):
        return dict(det)
    if hasattr(det, "model_dump"):
        return det.model_dump()
    return dict(det.__dict__)


def person_detections(detections, person_label: str):
    """Detections with a box whose classLabel equals person_label (case-insensitive); all of them if the label is empty."""
    label = (person_label or "").strip().lower()
    out = []
    for det in detections or []:
        d = to_dict(det)
        box = d.get("boundingBox")
        if not box:
            continue
        if label and str(d.get("classLabel", "")).lower() != label:
            continue
        out.append(d)
    return out


def box_xyxy(det):
    b = det["boundingBox"]
    b = b if isinstance(b, dict) else to_dict(b)
    return [float(b["left"]), float(b["top"]), float(b["left"]) + float(b["width"]), float(b["top"]) + float(b["height"])]


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def actions_from_probs(probs: np.ndarray, k: int, threshold: float):
    """Best k AVA actions with probability >= threshold, as [{action, score}]. Can be empty (multi-label output)."""
    scored = [(float(probs[i - 1]), name) for i, name in AVA.items() if i - 1 < len(probs)]
    scored = [x for x in scored if x[0] >= threshold]
    scored.sort(reverse=True)
    return [{"action": name, "score": round(score, 3)} for score, name in scored[:k]]


def match_actions(det, results, min_iou: float = 0.3):
    """
    Actions for a current detection from the last inference results [(trackerID, box, actions)].
    Matched by trackerID when both have one, otherwise by the best box overlap (IoU >= min_iou).
    """
    tid = det.get("trackerID")
    if tid is not None:
        for r_tid, _, actions in results:
            if r_tid is not None and r_tid == tid:
                return actions
    box = box_xyxy(det)
    best, best_iou = [], min_iou
    for _, r_box, actions in results:
        v = iou(box, r_box)
        if v >= best_iou:
            best, best_iou = actions, v
    return best


class PersonClipBuffer:
    """Last clip_frames frames of one stream (resized, not cropped) with the person detections of each frame."""

    def __init__(self, clip_frames: int):
        from collections import deque

        self.frames = deque(maxlen=clip_frames)
        self.persons = deque(maxlen=clip_frames)
        self.scale = 1.0
        self.count = 0
        self.last_run = None
        self.results = []  # [(trackerID, box_xyxy in original pixels, actions)] from the last inference

    def add(self, frame_rgb, scale: float, persons):
        if self.frames and self.frames[-1].shape != frame_rgb.shape:  # stream resolution changed
            self.frames.clear()
            self.persons.clear()
        self.frames.append(frame_rgb)
        self.persons.append(persons)
        self.scale = scale
        self.count += 1

    @property
    def ready(self) -> bool:
        return len(self.frames) == self.frames.maxlen

    def due(self, interval: int) -> bool:
        if not self.ready:
            return False
        return self.last_run is None or self.count - self.last_run >= interval

    def middle_persons(self):
        """Detections of the middle frame: AVA labels the action at the centre of the clip."""
        return self.persons[len(self.persons) // 2]
