"""Frame / video annotations for eval exports."""

from __future__ import annotations

import numpy as np


def _load_font(size: int = 11):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def annotate_frames_uint8(
    frames: np.ndarray,
    labels: list[str] | None = None,
    *,
    prefix: str = "step",
) -> np.ndarray:
    """Draw labels on each frame (top-right). frames: (T, H, W, C) uint8."""
    from PIL import Image, ImageDraw

    if frames.ndim != 4:
        raise ValueError(f"expected (T,H,W,C), got {frames.shape}")
    font = _load_font(max(10, min(14, frames.shape[2] // 8)))
    out: list[np.ndarray] = []
    for t, frame in enumerate(frames):
        text = labels[t] if labels is not None else f"{prefix} {t}"
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        x = max(2, frame.shape[1] - text_w - 4)
        draw.text(
            (x, 4),
            text,
            fill=(255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
            font=font,
        )
        out.append(np.asarray(img))
    return np.stack(out, axis=0)
