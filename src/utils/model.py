"""Loading and running the SlowFast model."""

import os
import shutil
import urllib.request

import numpy as np
import torch
import torch.nn as nn

from .labels import KINETICS400


def load_labels():
    """400 Kinetics class names, index = model output index."""
    return list(KINETICS400)


def load_model(weights_dir: str):
    """
    slowfast_r50 (8x8, Kinetics-400) from PyTorchVideo. The checkpoint (~264 MB) is downloaded once
    into weights_dir/checkpoints/ and reused on the next start.
    """
    os.makedirs(weights_dir, exist_ok=True)
    torch.hub.set_dir(weights_dir)
    from pytorchvideo.models.hub import slowfast_r50

    return slowfast_r50(pretrained=True).eval()


def build_custom_model(checkpoint: dict):
    """
    slowfast_r50 with a new head for the checkpoint's classes, as saved by notebooks/train_slowfast.ipynb:
    {"state_dict": ..., "classes": [...], "arch": "slowfast_r50", "clip_frames": ...}. Returns (model, classes).
    """
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "classes" not in checkpoint:
        raise ValueError(
            "SlowFast - the weights file is not a SlowFast checkpoint; create it with notebooks/train_slowfast.ipynb"
        )
    if checkpoint.get("arch", "slowfast_r50") != "slowfast_r50":
        raise ValueError(f"SlowFast - unsupported architecture in the weights file: {checkpoint.get('arch')}")
    from pytorchvideo.models.hub import slowfast_r50

    classes = list(checkpoint["classes"])
    model = slowfast_r50(pretrained=False)
    head = model.blocks[-1]
    head.proj = nn.Linear(head.proj.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), classes


def load_custom_model(path: str):
    """Model and class names from a fine-tuned checkpoint file."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return build_custom_model(checkpoint)


def download(url: str, path: str, timeout: int = 120):
    """Downloads url to path once (skipped if the file exists); writes to a temporary file first."""
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".partial"
    with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    os.replace(tmp, path)
    return path


def load_ava_model(weights_dir: str):
    """
    slowfast_r50_detection (8x8, AVA) from PyTorchVideo: actions per person box. The checkpoint is
    downloaded once into weights_dir/checkpoints/ and reused on the next start.
    """
    os.makedirs(weights_dir, exist_ok=True)
    torch.hub.set_dir(weights_dir)
    from pytorchvideo.models.hub import slowfast_r50_detection

    return slowfast_r50_detection(pretrained=True).eval()


def predict_boxes(model, inputs, boxes_xyxy) -> np.ndarray:
    """AVA probabilities (N, 80) for N boxes (x1, y1, x2, y2 in the clip's pixel coordinates). Multi-label: sigmoid per action."""
    boxes = torch.as_tensor(np.asarray(boxes_xyxy, dtype=np.float32)).reshape(-1, 4)
    boxes = torch.cat([torch.zeros(len(boxes), 1), boxes], dim=1).to(inputs[0].device)  # first column = clip index
    with torch.inference_mode():
        probs = model(list(inputs), boxes)  # the head already applies a sigmoid
    return probs.float().cpu().numpy()


def resolve_device(name: str):
    """'CPU' -> 'cpu'; 'GPU' -> 'cuda' if available, otherwise 'cpu'. Returns (device, fell_back)."""
    if name == "GPU":
        if torch.cuda.is_available():
            return "cuda", False
        return "cpu", True
    return "cpu", False


def predict(model, inputs) -> np.ndarray:
    """Class probabilities (400,) for one clip. The model returns logits, so softmax is applied here."""
    with torch.inference_mode():
        logits = model(list(inputs))  # the model replaces the list items in place; keep the caller's list intact
    return torch.softmax(logits, dim=1)[0].float().cpu().numpy()
