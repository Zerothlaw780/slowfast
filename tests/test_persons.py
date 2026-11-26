import numpy as np

from src.utils.clip import resize_short, to_inputs
from src.utils.labels import AVA
from src.utils.persons import (
    PersonClipBuffer,
    actions_from_probs,
    box_xyxy,
    iou,
    match_actions,
    person_detections,
)


def det(left, top, w, h, label="person", tid=None):
    d = {"boundingBox": {"left": left, "top": top, "width": w, "height": h}, "classLabel": label, "confidence": 0.9}
    if tid is not None:
        d["trackerID"] = tid
    return d


def test_ava_labels():
    assert len(AVA) == 60 and AVA[5] == "fall down" and max(AVA) == 80


def test_resize_short_keeps_aspect_and_scale():
    frame = np.zeros((480, 640, 3), np.uint8)
    out, scale = resize_short(frame)
    assert out.shape == (256, 341, 3) and abs(scale - 256 / 480) < 1e-9
    slow, fast = to_inputs([out] * 32)
    assert fast.shape == (1, 3, 32, 256, 341) and slow.shape == (1, 3, 8, 256, 341)


def test_person_filter():
    dets = [det(0, 0, 10, 10), det(0, 0, 10, 10, "car"), {"classLabel": "person"}]
    assert len(person_detections(dets, "person")) == 1
    assert len(person_detections(dets, "")) == 2  # every detection with a box
    assert len(person_detections(dets, "PERSON")) == 1


def test_actions_from_probs_multilabel():
    probs = np.zeros(80)
    probs[5 - 1], probs[80 - 1], probs[2 - 1] = 0.8, 0.5, 0.9  # fall down, watch (a person), id 2 is not an AVA-60 class
    rows = actions_from_probs(probs, 3, 0.3)
    assert [r["action"] for r in rows] == ["fall down", "watch (a person)"]
    assert actions_from_probs(probs, 3, 0.95) == []


def test_match_by_tracker_then_iou():
    results = [(7, [0, 0, 100, 200], [{"action": "stand", "score": 0.9}]),
               (None, [300, 0, 400, 200], [{"action": "sit", "score": 0.8}])]
    assert match_actions(det(250, 50, 100, 200, tid=7), results)[0]["action"] == "stand"  # moved, same tracker
    assert match_actions(det(305, 5, 100, 200), results)[0]["action"] == "sit"  # no tracker, overlaps
    assert match_actions(det(600, 0, 50, 50), results) == []
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9
    assert box_xyxy(det(1, 2, 3, 4)) == [1, 2, 4, 6]


def test_person_buffer_uses_middle_frame():
    buf = PersonClipBuffer(5)
    for i in range(5):
        buf.add(np.zeros((4, 4, 3), np.uint8), 1.0, [det(i, 0, 1, 1)])
    assert buf.ready and buf.due(3)
    assert buf.middle_persons()[0]["boundingBox"]["left"] == 2
    buf.add(np.zeros((8, 8, 3), np.uint8), 1.0, [])  # resolution change resets the buffer
    assert len(buf.frames) == 1
