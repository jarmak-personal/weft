"""Image loading/saving helpers with linear RGB conversion."""

from __future__ import annotations

import numpy as np
from PIL import Image


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1.0 + a) * np.power(np.clip(x, 0.0, 1.0), 1 / 2.4) - a)


def load_image_linear(path: str) -> tuple[np.ndarray, int, int]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    linear = srgb_to_linear(arr)
    h, w, _ = linear.shape
    return linear, w, h


def save_image_linear(path: str, image_linear: np.ndarray) -> None:
    srgb = np.clip(linear_to_srgb(image_linear), 0.0, 1.0)
    arr = (srgb * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
