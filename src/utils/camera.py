"""Robust macOS-compatible camera acquisition.

Why macOS camera backends behave inconsistently
------------------------------------------------
OpenCV's VideoCapture on macOS wraps Apple's AVFoundation framework.
There are two distinct failure modes that both appear "working" to naive code:

1.  cap.isOpened() returns True, but cap.read() returns black frames.
    AVFoundation starts a capture session *asynchronously*.  isOpened()
    returns True as soon as the session was *requested*, not when frames are
    actually flowing.  The first 10–40 read() calls can return all-zero
    frames while the sensor initialises, auto-exposes, and white-balances.
    This window varies by camera model (USB: 200–800 ms; built-in: 50–200 ms).

2.  cap.isOpened() returns True, cap.read() returns ret=True, but frame is
    zeros anyway.  This happens when the requested resolution is unsupported
    by the camera's firmware and AVFoundation silently delivers empty buffers
    rather than negotiating a supported format.

Why VideoCapture "opens" while still returning black frames
------------------------------------------------------------
VideoCapture::open() calls AVCaptureSession.startRunning(), which returns
immediately.  Frames arrive later via an internal capture delegate callback.
The first cap.read() call races against that first callback — if it arrives
before the callback fires, OpenCV returns zeros (the initial buffer state).

How CAP_AVFOUNDATION differs from CAP_ANY
------------------------------------------
CAP_ANY = 0 lets OpenCV pick the backend at runtime.  On macOS, pip-installed
opencv-python sometimes resolves CAP_ANY to a GStreamer or Qt backend (if
those libraries appear in the Python environment) rather than AVFoundation.
Those alternative backends may not support macOS AVCaptureSession semantics
at all, producing permament black frames.  Specifying CAP_AVFOUNDATION = 1200
explicitly bypasses that selection.

What production systems do
---------------------------
1.  Try the platform-native backend first.
2.  Warm up: discard frames until a frame with real pixel content arrives,
    up to a timeout, before passing the camera handle to the consumer.
3.  Validate every incoming frame at capture time; discard blanks silently.
4.  Fall back to a lower resolution if the target resolution produces blanks
    (indicates unsupported mode rather than a timing issue).
5.  Log the backend and exact resolution that succeeded for post-mortem debugging.

Config keys (all optional, with defaults)
------------------------------------------
  device_id:       int   = 0          OS camera index
  backend:         str   = "auto"     "auto" | "avfoundation" | "any" | "qt"
  resolution:      list  = [640, 480]
  warmup_frames:   int   = 30         max frames to read during warm-up
  warmup_timeout_s float = 6.0        max wall-clock seconds for warm-up
  min_frame_mean:  float = 1.0        frames below this mean are blank
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Backend registry ──────────────────────────────────────────────────────────
# Ordered by macOS preference.  CAP_AVFOUNDATION must be first so it takes
# priority over the OS default picker (CAP_ANY) which may resolve to GStreamer.

_BACKEND_IDS: dict[str, int] = {
    "avfoundation": cv2.CAP_AVFOUNDATION,   # 1200 — macOS native, always try first
    "any":          cv2.CAP_ANY,             # 0    — OS default picker
    "qt":           cv2.CAP_QT,              # 500  — legacy QuickTime (macOS fallback)
    "dshow":        cv2.CAP_DSHOW,           # 700  — Windows DirectShow
    "v4l2":         cv2.CAP_V4L2,            # 200  — Linux Video4Linux2
}

# "auto" expands to this ordered list.
_AUTO_ORDER: list[str] = ["avfoundation", "any", "qt"]

# Resolution fallback ladder — tried in order if the requested resolution fails.
_RESOLUTION_LADDER: list[tuple[int, int]] = [(1280, 720), (640, 480), (320, 240)]


# ── Frame validation ──────────────────────────────────────────────────────────

def is_valid_frame(frame: Optional[np.ndarray], min_mean: float = 1.0) -> bool:
    """Return True if *frame* contains non-trivial pixel content.

    Rejects: None, wrong dtype, non-3-channel arrays, and all-zero (black)
    frames.  min_mean=1.0 means average pixel value ≥ 1 across all channels —
    a very conservative threshold that only rejects completely black frames.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return False
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False
    if frame.dtype != np.uint8:
        return False
    return float(frame.mean()) >= min_mean


# ── Public entry point ────────────────────────────────────────────────────────

def open_camera(capture_cfg: dict) -> cv2.VideoCapture:
    """Open and warm up a VideoCapture that reliably delivers valid frames.

    Tries backends in priority order (AVFoundation → Any → QT by default).
    For each backend, attempts the requested resolution first, then steps
    down the resolution ladder if that resolution produces blank frames.

    Returns a ready cv2.VideoCapture whose next read() will return a valid
    frame.  The caller is responsible for calling cap.release().

    Raises RuntimeError if no combination of backend and resolution succeeds.
    """
    device_id      = capture_cfg.get("device_id", 0)
    req_res        = capture_cfg.get("resolution", [640, 480])
    req_w, req_h   = int(req_res[0]), int(req_res[1])
    backend_pref   = str(capture_cfg.get("backend", "auto")).lower()
    warmup_frames  = int(capture_cfg.get("warmup_frames",  30))
    warmup_timeout = float(capture_cfg.get("warmup_timeout_s", 6.0))
    min_mean       = float(capture_cfg.get("min_frame_mean", 1.0))

    backend_order  = _resolve_backend_order(backend_pref)

    # Resolution list: requested first, then lower rungs we haven't tried yet.
    resolutions = [(req_w, req_h)] + [
        (w, h) for (w, h) in _RESOLUTION_LADDER if (w, h) != (req_w, req_h)
    ]

    for backend_name in backend_order:
        for width, height in resolutions:
            cap = _attempt(
                device_id, backend_name, _BACKEND_IDS[backend_name],
                width, height, warmup_frames, warmup_timeout, min_mean,
            )
            if cap is not None:
                return cap

    raise RuntimeError(
        f"Cannot open camera {device_id} with any backend or resolution. "
        f"Tried backends={backend_order}, resolutions={resolutions}. "
        "Check macOS camera permissions: System Preferences → Privacy & Security "
        "→ Camera → enable Terminal (or your IDE)."
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_backend_order(pref: str) -> list[str]:
    if pref == "auto":
        return _AUTO_ORDER
    if pref in _BACKEND_IDS:
        # Explicit preference first, then any as safety net.
        return [pref] + [b for b in _AUTO_ORDER if b != pref]
    logger.warning("Unknown camera backend '%s'; defaulting to auto order.", pref)
    return _AUTO_ORDER


def _attempt(
    device_id:    int,
    backend_name: str,
    backend_id:   int,
    width:        int,
    height:       int,
    warmup_frames:  int,
    warmup_timeout: float,
    min_mean:       float,
) -> Optional[cv2.VideoCapture]:
    """Try to open and warm up one backend/resolution combination.

    Returns a ready VideoCapture on success, None if the attempt fails.
    Always releases the VideoCapture before returning None.
    """
    logger.debug(
        "Attempting backend=%s (%d)  device=%d  %dx%d",
        backend_name, backend_id, device_id, width, height,
    )

    cap = cv2.VideoCapture(device_id, backend_id)
    if not cap.isOpened():
        logger.debug("  %s: VideoCapture.isOpened() returned False", backend_name)
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # CAP_PROP_BUFFERSIZE = 1 minimises internal buffering so cap.read() returns
    # the freshest available frame.  Some backends (e.g. V4L2) require ≥ 2;
    # if set(1) is ignored, OpenCV silently uses its default.
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    valid, n_read, n_blank = _warmup(cap, warmup_frames, warmup_timeout, min_mean)

    if not valid:
        logger.debug(
            "  %s %dx%d: %d frames read (%d blank) — no valid frame; releasing",
            backend_name, actual_w, actual_h, n_read, n_blank,
        )
        cap.release()
        return None

    logger.info(
        "Camera ready: backend=%s  device=%d  requested=%dx%d  actual=%dx%d  "
        "valid frame after %d/%d reads",
        backend_name, device_id, width, height, actual_w, actual_h,
        n_read, warmup_frames,
    )
    return cap


def _warmup(
    cap:           cv2.VideoCapture,
    max_frames:    int,
    timeout_s:     float,
    min_mean:      float,
) -> tuple[bool, int, int]:
    """Read frames until a valid one arrives, or limits are hit.

    Returns (got_valid_frame, total_frames_read, blank_frames_count).

    We intentionally consume the valid frame here.  The camera is live, so
    the next read() after warm-up will immediately produce another valid frame.
    Returning the cap mid-stream and letting the caller start from the next
    frame is correct for real-time capture.
    """
    t_start = time.perf_counter()
    n_read  = 0
    n_blank = 0

    while n_read < max_frames:
        elapsed = time.perf_counter() - t_start
        if elapsed >= timeout_s:
            logger.debug("  warmup: timeout after %.1f s (%d frames)", elapsed, n_read)
            break

        ret, frame = cap.read()
        n_read += 1

        if not ret:
            logger.debug("  warmup: cap.read() returned False on attempt %d", n_read)
            time.sleep(0.033)
            continue

        if is_valid_frame(frame, min_mean):
            logger.debug(
                "  warmup: valid frame at read #%d  mean=%.1f  elapsed=%.0f ms",
                n_read, float(frame.mean()), elapsed * 1000,
            )
            return True, n_read, n_blank

        n_blank += 1
        logger.debug(
            "  warmup: blank frame #%d (mean=%.3f)",
            n_read, float(frame.mean()) if frame is not None else -1,
        )

    return False, n_read, n_blank
