"""
Frame buffer and preprocessing for SlowFast (slowfast_r50, Kinetics-400).

pytorchvideo.transforms does not work with torchvision >= 0.17, so the preprocessing
used by the PyTorchVideo model zoo is written here with OpenCV and torch:
short side -> 256, center crop 256x256, RGB, scale to [0, 1], normalize with
mean 0.45 / std 0.225, 32 frames for the Fast pathway and every 4th of them (8) for the Slow pathway.
"""

from collections import deque

import cv2
import numpy as np
import torch

SIDE = 256  # short side and crop size
MEAN = 0.45
STD = 0.225
NUM_FRAMES = 32  # frames the Fast pathway expects
ALPHA = 4  # Slow pathway sees 1 of every ALPHA frames (32 / 4 = 8)

WARMING_UP = [{"rank": 0, "action": "warming up", "score": 0.0}]


def resize_crop(frame_bgr: np.ndarray, size: int = SIDE) -> np.ndarray:
    """BGR frame of any size -> RGB uint8 frame of size x size (short side resize + center crop)."""
    h, w = frame_bgr.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = max(size, round(w * scale)), max(size, round(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top, left = (new_h - size) // 2, (new_w - size) // 2
    crop = resized[top : top + size, left : left + size]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def sample_indices(n: int, k: int = NUM_FRAMES) -> np.ndarray:
    """k evenly spaced indices in [0, n-1]; repeats indices when n < k."""
    return np.linspace(0, n - 1, k).round().astype(int)


def to_inputs(frames, device="cpu"):
    """List of RGB uint8 frames (already resize_crop'ed) -> [slow, fast] tensors of shape 1 x 3 x T x H x W."""
    idx = sample_indices(len(frames))
    clip = np.stack([frames[i] for i in idx])  # T, H, W, 3
    fast = torch.from_numpy(clip).float().div_(255.0).sub_(MEAN).div_(STD)
    fast = fast.permute(3, 0, 1, 2).contiguous()  # 3, T, H, W
    slow_idx = torch.linspace(0, fast.shape[1] - 1, fast.shape[1] // ALPHA).long()
    slow = fast.index_select(1, slow_idx)
    return [slow.unsqueeze(0).to(device), fast.unsqueeze(0).to(device)]


def top_actions(probs: np.ndarray, labels, k: int, threshold: float):
    """Rows [{rank, action, score}] for the k best classes with score >= threshold (the best one is always kept)."""
    order = np.argsort(probs)[::-1][:k]
    rows = [
        {"rank": r + 1, "action": labels[i], "score": round(float(probs[i]), 3)}
        for r, i in enumerate(order)
        if r == 0 or probs[i] >= threshold
    ]
    return rows


class ClipBuffer:
    """Keeps the last clip_frames frames of one stream and decides when the model should run."""

    def __init__(self, clip_frames: int):
        self.frames = deque(maxlen=clip_frames)
        self.count = 0  # frames seen
        self.last_run = None  # frame count at the last inference
        self.probs = None  # class probabilities from the last inference

    def add(self, frame_rgb: np.ndarray):
        self.frames.append(frame_rgb)
        self.count += 1

    @property
    def ready(self) -> bool:
        return len(self.frames) == self.frames.maxlen

    def due(self, interval: int) -> bool:
        """True when the clip is full and interval frames have passed since the last run (or it never ran)."""
        if not self.ready:
            return False
        return self.last_run is None or self.count - self.last_run >= interval

    def mark_run(self, probs: np.ndarray):
        self.probs = probs
        self.last_run = self.count


def resize_short(frame_bgr: np.ndarray, side: int = SIDE):
    """BGR frame -> (RGB uint8 frame with short side = side, scale). No crop, so people at the edges stay visible."""
    h, w = frame_bgr.shape[:2]
    scale = side / min(h, w)
    resized = cv2.resize(frame_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), scale
