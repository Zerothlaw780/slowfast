import numpy as np
import pytest

pytest.importorskip("pytorchvideo")

from pytorchvideo.models.hub import slowfast_r50

from src.utils.clip import to_inputs, top_actions
from src.utils.model import load_labels, predict, resolve_device


def test_labels():
    labels = load_labels()
    assert len(labels) == 400 and labels == sorted(labels) and labels[0] == "abseiling"


def test_predict_returns_probabilities():
    model = slowfast_r50(pretrained=False).eval()  # random weights: checks shapes, not accuracy
    frames = [np.random.randint(0, 255, (256, 256, 3), np.uint8) for _ in range(32)]
    probs = predict(model, to_inputs(frames))
    assert probs.shape == (400,) and abs(probs.sum() - 1) < 1e-4
    rows = top_actions(probs, load_labels(), 3, 0.0)
    assert [r["rank"] for r in rows] == [1, 2, 3]


def test_gpu_falls_back_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("GPU") == ("cpu", True)
    assert resolve_device("CPU") == ("cpu", False)
