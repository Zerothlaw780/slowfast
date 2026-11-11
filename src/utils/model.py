"""Loading and running the SlowFast model."""

import os

import numpy as np
import torch

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
        logits = model(inputs)
    return torch.softmax(logits, dim=1)[0].float().cpu().numpy()
