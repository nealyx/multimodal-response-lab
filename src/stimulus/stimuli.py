"""Visual stimulus implementations for behavioral experiments.

Design contract
---------------
  - Every Stimulus subclass is purely presentational.  It draws itself onto
    a copy of the incoming frame and returns that copy.  No timing, no FSM,
    no event emission — those are the StimulusPresenter's responsibility.
  - render(frame, elapsed_ms) receives monotonic time since onset so that
    animated stimuli (moving target, pulsing prompt) can be frame-time-
    independent.
  - metadata() returns a JSON-serialisable dict logged inside StimulusEvent.

Why four stimulus types?
------------------------
  ColorFlash   — lowest cognitive demand; the whole field changes colour.
                 Elicits an orienting response that is detectable via gaze
                 or engagement state change even without a key press.

  ShapeStimulus — higher demand; requires recognising a specific shape.
                  Can be positioned off-centre to probe spatial attention.

  MovingTarget — tests smooth-pursuit behaviour.  Gaze should track the
                 target; failure to do so is detectable via gaze stability
                 and on-screen fraction dropping.

  ReactionPrompt — explicit key-press instruction; gives the most direct
                   latency measurement because both the stimulus onset and
                   the response are timestamped on the same monotonic clock.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class Stimulus(ABC):
    """Base class for all visual stimuli."""

    def __init__(
        self,
        stimulus_id: str,
        duration_ms: float,
        cfg: Dict[str, Any],
    ) -> None:
        self.stimulus_id = stimulus_id
        self.duration_ms = duration_ms
        self._cfg = cfg

    @property
    def stimulus_type(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def render(self, frame: np.ndarray, elapsed_ms: float) -> np.ndarray:
        """Render onto a copy of *frame*.  elapsed_ms = ms since onset."""
        ...

    def metadata(self) -> Dict[str, Any]:
        """Type-specific parameters for StimulusEvent logging."""
        return {}


# ── Stimulus implementations ──────────────────────────────────────────────────

class ColorFlash(Stimulus):
    """Full-screen or region colour overlay.

    Uses addWeighted blending so the face mesh remains partially visible,
    making it easier to track gaze changes caused by the flash.
    """

    def __init__(
        self,
        stimulus_id:  str,
        duration_ms:  float,
        cfg:          Dict[str, Any],
        *,
        color:  Tuple[int, int, int] = (0, 100, 255),   # BGR
        alpha:  float = 0.45,
        region: Optional[Tuple[int, int, int, int]] = None,  # (x, y, w, h)
    ) -> None:
        super().__init__(stimulus_id, duration_ms, cfg)
        self._color  = (int(color[0]), int(color[1]), int(color[2]))
        self._alpha  = float(alpha)
        self._region = region

    def render(self, frame: np.ndarray, elapsed_ms: float) -> np.ndarray:
        out = frame.copy()
        h, w = frame.shape[:2]
        x, y, rw, rh = self._region if self._region else (0, 0, w, h)
        overlay = out.copy()
        cv2.rectangle(overlay, (x, y), (x + rw, y + rh), self._color, -1)
        cv2.addWeighted(overlay, self._alpha, out, 1.0 - self._alpha, 0, out)
        return out

    def metadata(self) -> Dict[str, Any]:
        return {
            "color":  list(self._color),
            "alpha":  self._alpha,
            "region": list(self._region) if self._region else None,
        }


class ShapeStimulus(Stimulus):
    """A geometric shape at a fixed screen position.

    Supports: circle, square, cross, arrow_up, arrow_right.
    Position anchors: center, top, bottom, left, right.
    """

    _POSITIONS: Dict[str, Tuple[float, float]] = {
        "center": (0.50, 0.50),
        "top":    (0.50, 0.20),
        "bottom": (0.50, 0.80),
        "left":   (0.20, 0.50),
        "right":  (0.80, 0.50),
    }

    def __init__(
        self,
        stimulus_id: str,
        duration_ms: float,
        cfg:         Dict[str, Any],
        *,
        shape:       str = "circle",
        color:       Tuple[int, int, int] = (50, 50, 255),
        size:        int = 60,
        position:    str = "center",
        border_only: bool = False,
    ) -> None:
        super().__init__(stimulus_id, duration_ms, cfg)
        self._shape       = shape
        self._color       = (int(color[0]), int(color[1]), int(color[2]))
        self._size        = int(size)
        self._position    = position
        self._border_only = border_only

    def _center(self, w: int, h: int) -> Tuple[int, int]:
        fx, fy = self._POSITIONS.get(self._position, (0.5, 0.5))
        return (int(w * fx), int(h * fy))

    def render(self, frame: np.ndarray, elapsed_ms: float) -> np.ndarray:
        out = frame.copy()
        h, w = frame.shape[:2]
        cx, cy = self._center(w, h)
        fill = 1 if self._border_only else -1
        s = self._size

        if self._shape == "circle":
            cv2.circle(out, (cx, cy), s // 2, self._color, fill, cv2.LINE_AA)

        elif self._shape == "square":
            cv2.rectangle(
                out, (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 2),
                self._color, fill, cv2.LINE_AA,
            )

        elif self._shape == "cross":
            t = max(2, s // 8)
            cv2.line(out, (cx - s//2, cy), (cx + s//2, cy), self._color, t, cv2.LINE_AA)
            cv2.line(out, (cx, cy - s//2), (cx, cy + s//2), self._color, t, cv2.LINE_AA)

        elif self._shape == "arrow_up":
            pts = np.array([
                [cx,        cy - s // 2],
                [cx + s//3, cy],
                [cx + s//8, cy],
                [cx + s//8, cy + s // 2],
                [cx - s//8, cy + s // 2],
                [cx - s//8, cy],
                [cx - s//3, cy],
            ], dtype=np.int32)
            if self._border_only:
                cv2.polylines(out, [pts], True, self._color, 2, cv2.LINE_AA)
            else:
                cv2.fillPoly(out, [pts], self._color, cv2.LINE_AA)

        elif self._shape == "arrow_right":
            pts = np.array([
                [cx + s // 2,  cy],
                [cx,           cy - s // 3],
                [cx,           cy - s // 8],
                [cx - s // 2,  cy - s // 8],
                [cx - s // 2,  cy + s // 8],
                [cx,           cy + s // 8],
                [cx,           cy + s // 3],
            ], dtype=np.int32)
            if self._border_only:
                cv2.polylines(out, [pts], True, self._color, 2, cv2.LINE_AA)
            else:
                cv2.fillPoly(out, [pts], self._color, cv2.LINE_AA)

        return out

    def metadata(self) -> Dict[str, Any]:
        return {
            "shape":    self._shape,
            "color":    list(self._color),
            "size":     self._size,
            "position": self._position,
        }


class MovingTarget(Stimulus):
    """A dot that moves across the screen; tests smooth-pursuit gaze response.

    The target follows a continuous path (horizontal, vertical, diagonal, or
    circular) computed purely from elapsed_ms so rendering is deterministic
    regardless of frame rate drops.
    """

    def __init__(
        self,
        stimulus_id:   str,
        duration_ms:   float,
        cfg:           Dict[str, Any],
        *,
        color:         Tuple[int, int, int] = (50, 220, 50),
        size:          int   = 30,
        speed_px_per_s: float = 120.0,
        path:          str   = "horizontal",
    ) -> None:
        super().__init__(stimulus_id, duration_ms, cfg)
        self._color  = (int(color[0]), int(color[1]), int(color[2]))
        self._size   = int(size)
        self._speed  = float(speed_px_per_s)
        self._path   = path

    def _position(self, elapsed_ms: float, w: int, h: int) -> Tuple[int, int]:
        t      = elapsed_ms / 1000.0
        margin = self._size // 2 + 5

        if self._path == "horizontal":
            span   = w - 2 * margin
            period = span / self._speed * 2
            phase  = (t % period) / period if period > 0 else 0.0
            frac   = 1.0 - abs(2.0 * phase - 1.0)
            return (margin + int(frac * span), h // 2)

        elif self._path == "vertical":
            span   = h - 2 * margin
            period = span / self._speed * 2
            phase  = (t % period) / period if period > 0 else 0.0
            frac   = 1.0 - abs(2.0 * phase - 1.0)
            return (w // 2, margin + int(frac * span))

        elif self._path == "diagonal":
            sx     = w - 2 * margin
            sy     = h - 2 * margin
            period = min(sx, sy) / self._speed * 2
            phase  = (t % period) / period if period > 0 else 0.0
            frac   = 1.0 - abs(2.0 * phase - 1.0)
            return (margin + int(frac * sx), margin + int(frac * sy))

        else:  # circular
            r     = min(w, h) // 4
            omega = self._speed / max(r, 1)
            angle = omega * t
            return (w // 2 + int(r * math.cos(angle)), h // 2 + int(r * math.sin(angle)))

    def render(self, frame: np.ndarray, elapsed_ms: float) -> np.ndarray:
        out  = frame.copy()
        h, w = frame.shape[:2]
        cx, cy = self._position(elapsed_ms, w, h)
        r = self._size // 2
        cv2.circle(out, (cx, cy), r,     self._color,   -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), r + 2, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def metadata(self) -> Dict[str, Any]:
        return {
            "color":          list(self._color),
            "size":           self._size,
            "speed_px_per_s": self._speed,
            "path":           self._path,
        }


class ReactionPrompt(Stimulus):
    """Instruction prompt expecting an explicit key-press response.

    The prompt box pulses slightly to draw attention without causing a large
    luminance change that would confound gaze-based response detection.
    """

    def __init__(
        self,
        stimulus_id:  str,
        duration_ms:  float,
        cfg:          Dict[str, Any],
        *,
        text:         str = "PRESS SPACE",
        color:        Tuple[int, int, int] = (0, 220, 255),
        response_key: str = "space",
    ) -> None:
        super().__init__(stimulus_id, duration_ms, cfg)
        self._text         = text
        self._color        = (int(color[0]), int(color[1]), int(color[2]))
        self.response_key  = response_key

    def render(self, frame: np.ndarray, elapsed_ms: float) -> np.ndarray:
        out  = frame.copy()
        h, w = frame.shape[:2]
        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = 1.2
        thickness = 3
        (tw, th), _ = cv2.getTextSize(self._text, font, scale, thickness)
        tx = (w - tw) // 2
        ty = (h + th) // 2

        # Pulsing background box
        pulse   = 0.35 + 0.20 * math.sin(elapsed_ms / 160.0)
        pad     = 20
        overlay = out.copy()
        cv2.rectangle(
            overlay,
            (tx - pad, ty - th - pad),
            (tx + tw + pad, ty + pad),
            self._color, -1,
        )
        cv2.addWeighted(overlay, pulse, out, 1.0 - pulse, 0, out)

        # Text with black shadow
        cv2.putText(out, self._text, (tx, ty), font, scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(out, self._text, (tx, ty), font, scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)
        return out

    def metadata(self) -> Dict[str, Any]:
        return {
            "text":         self._text,
            "color":        list(self._color),
            "response_key": self.response_key,
        }
