"""Threaded webcam capture + MediaPipe Face Landmarker inference.

MediaPipe 0.10+ Task API
------------------------
The legacy mp.solutions.face_mesh is gone.  We now use:
  FaceLandmarker (mediapipe.tasks.python.vision) with RunningMode.VIDEO.

VIDEO mode vs LIVE_STREAM mode
-------------------------------
VIDEO mode   — synchronous: detect_for_video(image, timestamp_ms) blocks until
               done and returns FaceLandmarkerResult directly.  Simpler; use
               this when the calling thread controls the loop.
LIVE_STREAM  — asynchronous: detect_async(image, timestamp_ms) returns
               immediately; results arrive via result_callback on a MediaPipe
               internal thread.  Requires thread-safe result hand-off and is
               harder to reason about.  Deferred to a later iteration.

Timestamps must be monotonically increasing integers in milliseconds.
We derive them from time.perf_counter() scaled to ms.

Threading model (unchanged from Day 1 design)
----------------------------------------------

              daemon thread                       calling thread
         ┌──────────────────────┐          ┌──────────────────────────┐
         │  CaptureThread       │          │                          │
         │  ──────────────────  │─ write ─▶│  FrameBuffer             │
         │  cap.read() loop     │          │  (single-slot, Lock)     │
         └──────────────────────┘          └────────────┬─────────────┘
                                                        │ read
                                                ┌───────▼────────┐
                                                │  FacePipeline  │
                                                │  process_      │
                                                │  latest()      │
                                                │  detect_for_   │
                                                │  video()       │
                                                └────────────────┘

Frame ownership
---------------
  CaptureThread: cap.read() → FrameBuffer.write(frame.copy())
  FrameBuffer.read(): returns frame.copy() — caller owns this copy
  process_latest(): converts BGR→RGB for MediaPipe; FaceFrameResult.frame_raw
                    holds the BGR copy; caller owns the result

Model file
----------
  FaceLandmarker requires a .task model bundle downloaded separately.
  FacePipeline.start() auto-downloads it to the configured cache path if absent.
  Download URL is pinned to the float16 variant (~3 MB) which runs well on CPU.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

from src.utils.camera import is_valid_frame, open_camera
from src.utils.frame_debug import check_mediapipe_input, frame_info, save_debug_frame

logger = logging.getLogger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


# ── Result bundle ─────────────────────────────────────────────────────────────

@dataclass
class FaceFrameResult:
    """All outputs for one processed webcam frame."""
    frame_raw:         np.ndarray   # BGR, no overlay; owned by caller
    face_landmarks:    list         # list[list[NormalizedLandmark]], one list per face
    face_count:        int
    timestamp_capture: float        # time.perf_counter() in the capture thread
    timestamp_process: float        # time.perf_counter() after detect_for_video()
    frame_index:       int

    @property
    def latency_ms(self) -> float:
        return (self.timestamp_process - self.timestamp_capture) * 1000.0

    @property
    def face_detected(self) -> bool:
        return self.face_count > 0


# ── Single-slot frame buffer ──────────────────────────────────────────────────

class FrameBuffer:
    """Thread-safe buffer holding only the most recent captured frame.

    write() is called by the capture thread; read() by the processing thread.
    The single-slot design ensures latency stays bounded: we always process
    the newest frame, never a stale one that piled up in a FIFO queue.
    """

    def __init__(self) -> None:
        self._lock:      threading.Lock       = threading.Lock()
        self._frame:     Optional[np.ndarray] = None
        self._timestamp: float                = 0.0
        self._ready:     threading.Event      = threading.Event()

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame     = frame.copy()
            self._timestamp = time.perf_counter()
            self._ready.set()

    def read(self, timeout: float = 1.0) -> tuple[Optional[np.ndarray], float]:
        """Block until a frame is available or timeout elapses.

        Returns (frame_copy, capture_timestamp) or (None, 0.0) on timeout.
        Clears the ready flag so a second call without a new write() blocks.
        """
        if not self._ready.wait(timeout):
            return None, 0.0
        with self._lock:
            self._ready.clear()
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._timestamp


# ── Background capture thread ─────────────────────────────────────────────────

class CaptureThread(threading.Thread):
    """Daemon thread: opens the camera robustly, then feeds valid frames into FrameBuffer.

    Camera acquisition is delegated to open_camera() which handles backend
    selection, warm-up, and resolution fallback before the capture loop starts.
    The loop itself also validates each frame so that any sporadic blank frames
    that slip through after warm-up are silently discarded rather than
    forwarded to the inference thread.
    """

    def __init__(self, buffer: FrameBuffer, capture_cfg: dict) -> None:
        super().__init__(name="CaptureThread", daemon=True)
        self._buffer      = buffer
        self._capture_cfg = capture_cfg
        self._stop_evt    = threading.Event()

    def run(self) -> None:
        min_mean = float(self._capture_cfg.get("min_frame_mean", 1.0))

        try:
            cap = open_camera(self._capture_cfg)
        except RuntimeError as exc:
            logger.error("CaptureThread: %s", exc)
            return

        while not self._stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("cap.read() returned False — retrying")
                time.sleep(0.01)
                continue
            # Discard sporadic blank frames (can occur on some backends when the
            # camera momentarily drops exposure or the USB bus stalls).
            if not is_valid_frame(frame, min_mean):
                logger.debug("CaptureThread: discarding blank frame")
                continue
            self._buffer.write(frame)

        cap.release()
        logger.info("CaptureThread: VideoCapture released")

    def stop(self) -> None:
        self._stop_evt.set()


# ── Pipeline ──────────────────────────────────────────────────────────────────

class FacePipeline:
    """Webcam capture + MediaPipe Face Landmarker, designed as a context manager.

    Example
    -------
        with FacePipeline(cfg["face_mesh"], cfg["capture"]) as pipeline:
            while running:
                result = pipeline.process_latest()
                if result and result.face_detected:
                    lm_list = result.face_landmarks[0]  # first face
    """

    def __init__(
        self,
        face_cfg:   dict,
        capture_cfg: dict,
        *,
        debug_mode: bool = False,
        debug_dump_dir: str = "outputs/debug_frames",
    ) -> None:
        self._face_cfg      = face_cfg
        self._capture_cfg   = capture_cfg
        self._debug_mode    = debug_mode
        self._debug_dump_dir = debug_dump_dir
        self._buffer        = FrameBuffer()
        self._capture_thr: Optional[CaptureThread]               = None
        self._landmarker:  Optional[mp_vision.FaceLandmarker]    = None
        self._frame_idx:   int                                    = 0
        self._t0:          float                                  = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        model_path = self._ensure_model()

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=                    self._face_cfg.get("max_faces",                  1),
            min_face_detection_confidence=self._face_cfg.get("min_detection_confidence", 0.5),
            min_face_presence_confidence= self._face_cfg.get("min_presence_confidence",  0.5),
            min_tracking_confidence=      self._face_cfg.get("min_tracking_confidence",  0.5),
            output_face_blendshapes=      False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()

        self._capture_thr = CaptureThread(
            buffer=      self._buffer,
            capture_cfg= self._capture_cfg,
        )
        self._capture_thr.start()
        logger.info("FacePipeline started")

    def stop(self) -> None:
        if self._capture_thr is not None:
            self._capture_thr.stop()
            self._capture_thr.join(timeout=2.0)
        if self._landmarker is not None:
            self._landmarker.close()
        logger.info("FacePipeline stopped")

    def __enter__(self) -> "FacePipeline":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Processing ────────────────────────────────────────────────────────────

    def process_latest(self, timeout: float = 0.1) -> Optional[FaceFrameResult]:
        """Grab the newest frame and run Face Landmarker inference on it.

        Returns None if no frame arrived within *timeout* seconds.
        Must be called from one thread only — FaceLandmarker is not thread-safe.

        VIDEO mode requires monotonically increasing integer timestamps in ms.
        We compute them from perf_counter relative to the pipeline start time.
        """
        frame, ts_capture = self._buffer.read(timeout=timeout)
        if frame is None:
            return None

        if self._capture_cfg.get("flip_horizontal", True):
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Debug instrumentation (only when debug_mode=True) ─────────────────
        if self._debug_mode:
            fi = frame_info(frame, label=f"bgr_f{self._frame_idx}")
            ri = frame_info(rgb,   label=f"rgb_f{self._frame_idx}")
            warnings = check_mediapipe_input(rgb, label=f"frame_{self._frame_idx}")
            # Save the first 3 frames to disk for visual inspection.
            if self._frame_idx < 3:
                save_debug_frame(frame, self._debug_dump_dir, label=f"bgr_f{self._frame_idx}")
                save_debug_frame(rgb[:, :, ::-1].copy(), self._debug_dump_dir,
                                 label=f"rgb_as_bgr_f{self._frame_idx}")
            if warnings:
                logger.warning("Frame %d input check FAILED: %s", self._frame_idx, warnings)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp must be strictly increasing; derive from wall clock.
        timestamp_ms = int((time.perf_counter() - self._t0) * 1000)
        mp_result    = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if self._debug_mode:
            logger.debug(
                "frame=%d ts=%dms  mp_result.face_landmarks=%s",
                self._frame_idx, timestamp_ms,
                "None" if mp_result.face_landmarks is None
                else f"{len(mp_result.face_landmarks)} face(s)",
            )

        self._frame_idx += 1

        landmarks = mp_result.face_landmarks if mp_result.face_landmarks else []

        return FaceFrameResult(
            frame_raw=         frame,
            face_landmarks=    landmarks,
            face_count=        len(landmarks),
            timestamp_capture= ts_capture,
            timestamp_process= time.perf_counter(),
            frame_index=       self._frame_idx,
        )

    # ── Model management ──────────────────────────────────────────────────────

    def _ensure_model(self) -> Path:
        """Return model path, downloading from GCS if not cached."""
        model_path = Path(
            self._face_cfg.get("model_path", ".cache/models/face_landmarker.task")
        )
        if model_path.exists():
            logger.info("Model found at %s", model_path)
            return model_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Face Landmarker model → %s", model_path)
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        logger.info("Download complete: %.1f MB", model_path.stat().st_size / 1e6)
        return model_path
