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


def _fake_checkpoint(path, classes):
    """Checkpoint in the notebook's format with random weights."""
    import torch
    import torch.nn as nn

    model = slowfast_r50(pretrained=False)
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, len(classes))
    torch.save({"state_dict": model.state_dict(), "classes": classes, "arch": "slowfast_r50", "clip_frames": 64}, path)
    return model.eval()


def test_custom_checkpoint_roundtrip(tmp_path):
    from src.utils.model import load_custom_model

    classes = ["loading_pallet", "normal", "forklift_turning"]
    original = _fake_checkpoint(tmp_path / "ft.pth", classes)
    model, loaded_classes = load_custom_model(str(tmp_path / "ft.pth"))
    assert loaded_classes == classes
    frames = [np.random.randint(0, 255, (256, 256, 3), np.uint8) for _ in range(32)]
    inputs = to_inputs(frames)
    assert np.allclose(predict(model, inputs), predict(original, inputs), atol=1e-5)
    rows = top_actions(predict(model, inputs), loaded_classes, 5, 0.0)
    assert len(rows) == 3 and {r["action"] for r in rows} == set(classes)


def test_custom_checkpoint_rejects_other_files(tmp_path):
    import torch

    from src.utils.model import load_custom_model

    torch.save({"weights": 1}, tmp_path / "other.pth")
    with pytest.raises(ValueError, match="not a SlowFast checkpoint"):
        load_custom_model(str(tmp_path / "other.pth"))


def test_download_once(tmp_path):
    from src.utils.model import download

    src = tmp_path / "src.bin"
    src.write_bytes(b"abc")
    dst = tmp_path / "cache" / "w.pth"
    download(src.as_uri(), str(dst))
    src.write_bytes(b"changed")
    download(src.as_uri(), str(dst))  # already cached: not downloaded again
    assert dst.read_bytes() == b"abc"
