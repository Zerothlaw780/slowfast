import numpy as np
import pytest
import torch

from src.utils.clip import ClipBuffer, WARMING_UP, resize_crop, sample_indices, to_inputs, top_actions
from src.utils.draw import draw_actions, result_lines


def test_resize_crop_shape_and_rgb():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[..., 0] = 255  # blue in BGR
    out = resize_crop(frame)
    assert out.shape == (256, 256, 3)
    assert out[..., 2].mean() == 255 and out[..., 0].mean() == 0  # blue ends up in the last channel (RGB)


def test_sample_indices_even_and_repeating():
    assert list(sample_indices(64, 32))[:3] == [0, 2, 4]
    idx = sample_indices(8, 32)
    assert len(idx) == 32 and idx.min() == 0 and idx.max() == 7


@pytest.mark.parametrize("n", [8, 32, 100])
def test_to_inputs_shapes(n):
    frames = [np.full((256, 256, 3), i, np.uint8) for i in range(n)]
    slow, fast = to_inputs(frames)
    assert fast.shape == (1, 3, 32, 256, 256)
    assert slow.shape == (1, 3, 8, 256, 256)
    assert torch.allclose(slow[0, :, 0], fast[0, :, 0]) and torch.allclose(slow[0, :, -1], fast[0, :, -1])


def test_top_actions_threshold_keeps_best():
    labels = ["a", "b", "c", "d"]
    probs = np.array([0.1, 0.6, 0.25, 0.05])
    assert [r["action"] for r in top_actions(probs, labels, 3, 0.2)] == ["b", "c"]
    assert [r["action"] for r in top_actions(probs, labels, 3, 0.9)] == ["b"]
    assert top_actions(probs, labels, 2, 0.0)[1] == {"rank": 2, "action": "c", "score": 0.25}


def test_buffer_runs_when_full_then_every_interval():
    buf = ClipBuffer(4)
    runs = []
    for i in range(1, 14):
        buf.add(np.zeros((2, 2, 3), np.uint8))
        if buf.due(3):
            buf.mark_run(np.array([1.0]))
            runs.append(i)
    assert runs == [4, 7, 10, 13]


def test_draw_does_not_change_size():
    frame = np.zeros((480, 640, 3), np.uint8)
    out = draw_actions(frame, result_lines([{"rank": 1, "action": "running", "score": 0.8}], 32, 32))
    assert out.shape == frame.shape and out.any()
    assert result_lines(WARMING_UP, 5, 32) == ["SlowFast: warming up 5/32"]
