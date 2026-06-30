"""PIL-based frame and panel annotations for eval exports."""

from __future__ import annotations

import numpy as np


def load_font(size: int = 11):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_text(draw, xy: tuple[int, int], text: str, font) -> None:
    draw.text(
        xy,
        text,
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
        font=font,
    )


def annotate_panel_rows(
    panel_hwc: np.ndarray,
    row_h: int,
    n_samples: int,
    row_labels: tuple[str, ...],
) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(panel_hwc)
    draw = ImageDraw.Draw(img)
    font = load_font(11)
    n_rows = len(row_labels)
    for s in range(n_samples):
        for r, label in enumerate(row_labels):
            y = s * n_rows * row_h + r * row_h + 2
            _draw_text(draw, (4, y), label, font)
    return np.asarray(img)


def annotate_rollout_panel_rows(panel_hwc: np.ndarray, row_h: int, n_samples: int) -> np.ndarray:
    return annotate_panel_rows(panel_hwc, row_h, n_samples, ("gt", "pred"))


def annotate_multictx_panel(
    panel_hwc: np.ndarray,
    row_h: int,
    frame_w: int,
    ctx_lengths: list[int],
    n_samples: int,
    gap_px: int,
    total_frames: int,
    *,
    include_gt_row: bool = True,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(panel_hwc)
    draw = ImageDraw.Draw(img)
    font = load_font(11)
    n_ctx_rows = len(ctx_lengths)
    rows_per_sample = (1 + n_ctx_rows) if include_gt_row else n_ctx_rows
    for s in range(n_samples):
        row_offset = 0
        if include_gt_row:
            y0 = s * rows_per_sample * row_h
            _draw_text(draw, (4, y0 + 2), "gt", font)
            row_offset = 1
        for r, ctx in enumerate(ctx_lengths):
            y0 = s * rows_per_sample * row_h + (row_offset + r) * row_h
            _draw_text(draw, (4, y0 + 2), f"ctx={ctx}", font)
            if ctx <= 0 or ctx >= total_frames:
                continue
            ctx_w = ctx * frame_w
            has_gap = gap_px > 0 and 0 < ctx < total_frames
            first_rollout_x = ctx_w + gap_px if has_gap else ctx_w
            mid_y = y0 + row_h // 2 - 6
            if ctx_w > 48:
                _draw_text(draw, (max(4, ctx_w // 2 - 28), mid_y), "context", font)
            if first_rollout_x + frame_w <= panel_hwc.shape[1]:
                bbox = draw.textbbox((0, 0), "rollout", font=font)
                text_w = bbox[2] - bbox[0]
                text_x = first_rollout_x + max(4, (frame_w - text_w) // 2)
                _draw_text(draw, (text_x, mid_y), "rollout", font)
    return np.asarray(img)


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
    font = load_font(max(10, min(14, frames.shape[2] // 8)))
    out: list[np.ndarray] = []
    for t, frame in enumerate(frames):
        text = labels[t] if labels is not None else f"{prefix} {t}"
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        x = max(2, frame.shape[1] - text_w - 4)
        _draw_text(draw, (x, 4), text, font)
        out.append(np.asarray(img))
    return np.stack(out, axis=0)
