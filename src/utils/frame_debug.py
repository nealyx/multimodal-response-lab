"""Frame inspection utilities for debugging the capture and preprocessing path.

These functions are import-safe to use anywhere; they have no side effects
beyond writing to the logger or to a provided output path.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def frame_info(frame: np.ndarray, label: str = "") -> dict:
    """Return a dict of diagnostic properties for a BGR or RGB numpy frame."""
    tag = f"[{label}] " if label else ""
    if frame is None:
        return {f"{tag}frame": "None"}

    h, w = frame.shape[:2]
    ch   = frame.shape[2] if frame.ndim == 3 else 1
    info = {
        "label":         label,
        "shape":         frame.shape,
        "dtype":         str(frame.dtype),
        "C_contiguous":  bool(frame.flags["C_CONTIGUOUS"]),
        "strides":       frame.strides,
        "mean_per_ch":   [round(float(frame[:, :, c].mean()), 2) for c in range(ch)],
        "min":           int(frame.min()),
        "max":           int(frame.max()),
        "all_zero":      bool((frame == 0).all()),
        "all_same":      bool(frame.std() < 0.1),
    }
    logger.debug(
        "%sshape=%s dtype=%s C=%s min=%d max=%d mean=%s all_zero=%s",
        tag, info["shape"], info["dtype"], info["C_contiguous"],
        info["min"], info["max"], info["mean_per_ch"], info["all_zero"],
    )
    return info


def save_debug_frame(
    frame:    np.ndarray,
    out_dir:  str | Path,
    label:    str       = "",
    *,
    add_info: bool      = True,
) -> Path:
    """Save *frame* (BGR) to *out_dir* with a timestamped filename.

    Optionally burn diagnostic text (shape, dtype, channel means) onto the
    saved image so the file is self-describing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = label.replace(" ", "_") if label else "frame"
    path = out_dir / f"{slug}_{time.strftime('%H%M%S_%f')}.png"

    canvas = frame.copy() if add_info else frame
    if add_info and frame is not None:
        info = frame_info(frame, label)
        lines = [
            f"shape: {info['shape']}  dtype: {info['dtype']}",
            f"min={info['min']}  max={info['max']}  all_zero={info['all_zero']}",
            f"means: {info['mean_per_ch']}",
            f"C_contiguous: {info['C_contiguous']}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                canvas, line, (6, 20 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),   4, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, line, (6, 20 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

    cv2.imwrite(str(path), canvas)
    logger.info("Saved debug frame → %s", path)
    return path


def check_mediapipe_input(rgb: np.ndarray, label: str = "rgb_input") -> list[str]:
    """Validate a numpy array before passing to mp.Image.

    Returns a list of warning strings (empty = all checks pass).
    """
    warnings: list[str] = []

    if rgb.dtype != np.uint8:
        warnings.append(f"dtype is {rgb.dtype}, expected uint8")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        warnings.append(f"shape {rgb.shape} is not (H, W, 3)")
    if not rgb.flags["C_CONTIGUOUS"]:
        warnings.append("array is not C-contiguous — pass np.ascontiguousarray(rgb)")
    if rgb.max() == 0:
        warnings.append("all pixels are zero — camera may not have warmed up yet")
    if rgb.std() < 1.0:
        warnings.append(f"very low variance ({rgb.std():.2f}) — frame may be blank or uniform")

    for w in warnings:
        logger.warning("[%s] MediaPipe input check: %s", label, w)

    return warnings
