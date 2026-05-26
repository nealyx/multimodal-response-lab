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

import contextlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Camera enumeration ────────────────────────────────────────────────────────

# Keywords that identify Continuity Camera / iPhone / iPad devices.
# Matched case-insensitively against the name returned by system_profiler.
_CONTINUITY_KEYWORDS = ("iphone", "ipad", "continuity", "ios device")


@dataclass
class CameraInfo:
    index:          int
    name:           str    # display name from system_profiler; "unknown" if unavailable
    openable:       bool   # True if cv2.VideoCapture(index) succeeds
    width:          int    # reported frame width; 0 if not openable
    height:         int    # reported frame height; 0 if not openable
    is_continuity:  bool   # True if name matches Continuity Camera / iPhone / iPad


def _mac_camera_names() -> list[str]:
    """Return ordered camera display names from system_profiler (macOS only).

    Returns an empty list on non-macOS platforms or if system_profiler fails.
    The ordering matches AVFoundation's device enumeration order, which is also
    the order OpenCV uses for integer device indices (0, 1, 2 …).
    """
    try:
        raw = subprocess.check_output(
            ["system_profiler", "SPCameraDataType", "-json"],
            stderr=subprocess.DEVNULL, timeout=5.0,
        )
        data = json.loads(raw)
        return [c.get("_name", "unknown") for c in data.get("SPCameraDataType", [])]
    except Exception:
        return []


def _is_continuity_name(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in _CONTINUITY_KEYWORDS)


@contextlib.contextmanager
def _suppress_stderr():
    """Redirect C-level stderr to /dev/null (suppresses OpenCV probe noise)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_fd  = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(devnull)
        os.close(old_fd)


def enumerate_cameras(max_index: int = 8) -> list[CameraInfo]:
    """Probe device indices 0 … max_index-1 and return a CameraInfo for each.

    Uses AVFoundation on macOS.  Cameras that fail to open are still included
    in the result (openable=False) if system_profiler reports a name for them.
    Stops early once two consecutive indices fail to open and have no known name.
    """
    names = _mac_camera_names()
    result: list[CameraInfo] = []
    consecutive_unknown = 0

    for idx in range(max_index):
        with _suppress_stderr():
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        name = names[idx] if idx < len(names) else "unknown"

        if not cap.isOpened():
            cap.release()
            if name == "unknown":
                consecutive_unknown += 1
                if consecutive_unknown >= 2:
                    break   # no more cameras beyond this point
                continue
            consecutive_unknown = 0
            result.append(CameraInfo(idx, name, False, 0, 0, _is_continuity_name(name)))
            continue

        consecutive_unknown = 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        result.append(CameraInfo(idx, name, True, w, h, _is_continuity_name(name)))

    return result


def print_camera_candidates(capture_cfg: dict) -> None:
    """Enumerate available cameras, print a summary, and warn about Continuity Camera.

    Call this before opening the pipeline so the user sees what's available
    before any frames are captured.  Logs at INFO level so it appears in the
    default log output.

    Continuity Camera (macOS 13+) silently promotes the connected iPhone to
    device index 0, pushing the built-in FaceTime HD Camera to index 1.
    Running with the default device_id=0 will then open the iPhone instead
    of the Mac webcam.  This function makes that situation visible and tells
    the user which index to use instead.
    """
    selected = int(capture_cfg.get("device_id", 0))
    backend  = str(capture_cfg.get("backend", "auto"))
    cameras  = enumerate_cameras()

    logger.info("── Camera candidates ──────────────────────────────────────")
    if not cameras:
        logger.info("  (no cameras found via system_profiler or AVFoundation probe)")
    for cam in cameras:
        markers = []
        if cam.index == selected:
            markers.append("SELECTED")
        if cam.is_continuity:
            markers.append("CONTINUITY/IPHONE")
        tag = "  [" + ", ".join(markers) + "]" if markers else ""
        res = f"{cam.width}x{cam.height}" if cam.openable else "not openable"
        logger.info("  [%d] %-36s %s%s", cam.index, cam.name, res, tag)
    logger.info(
        "  backend: %s   (override with --backend; persistent in configs/default.yaml)",
        backend,
    )
    logger.info("───────────────────────────────────────────────────────────")

    # ── Warnings ─────────────────────────────────────────────────────────────
    selected_cam = next((c for c in cameras if c.index == selected), None)

    if selected_cam and selected_cam.is_continuity:
        mac_alt = next(
            (c.index for c in cameras if not c.is_continuity and c.openable),
            None,
        )
        hint = (
            f"  Use --device {mac_alt} to select the Mac built-in camera instead."
            if mac_alt is not None
            else "  Disable Continuity Camera in System Settings → General → AirPlay & Handoff."
        )
        logger.warning(
            "Device %d ('%s') appears to be a Continuity Camera (iPhone/iPad).\n"
            "  This may cause unexpected resolution, field of view, or latency.\n%s",
            selected, selected_cam.name, hint,
        )
    elif any(c.is_continuity for c in cameras):
        # Continuity Camera is present but not selected — just inform.
        for cc in (c for c in cameras if c.is_continuity):
            logger.info(
                "Note: Continuity Camera '%s' is present at index %d but not selected.",
                cc.name, cc.index,
            )


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
