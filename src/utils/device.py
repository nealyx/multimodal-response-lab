"""Device resolution with CUDA → MPS → CPU priority fallback.

Centralizing device selection here prevents the pattern where every module
re-implements its own `if torch.cuda.is_available()` chain, which diverges
over time and makes backend overrides (e.g. forcing CPU for a test run)
require changes in many places.

Usage
-----
    from src.utils.device import get_device, device_info

    device = get_device(preferred="auto", allow_fallback=True)
    # or, driven from config:
    device = get_device(**cfg["device"])
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class Backend(str, Enum):
    """Canonical backend names — used for config validation and comparisons."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"
    AUTO = "auto"


def get_device(
    preferred: str = "auto",
    allow_fallback: bool = True,
) -> torch.device:
    """Resolve the best available torch device.

    When *preferred* is ``"auto"`` the priority order is CUDA → MPS → CPU.
    When an explicit backend is requested, it is used if available; otherwise
    the function falls back to CPU (if *allow_fallback* is True) or raises.

    Args:
        preferred: Backend hint — ``"auto" | "cuda" | "mps" | "cpu"``.
        allow_fallback: Degrade to CPU rather than raising when the requested
            backend is unavailable.

    Returns:
        A :class:`torch.device` ready to pass to ``.to(device)`` calls.

    Raises:
        ValueError: If *preferred* is not a recognised backend string.
        RuntimeError: If the requested backend is unavailable and
            *allow_fallback* is ``False``.
    """
    preferred = preferred.lower()

    try:
        Backend(preferred)
    except ValueError:
        valid = [b.value for b in Backend]
        raise ValueError(
            f"Unknown backend '{preferred}'. Valid options: {valid}"
        )

    device = (
        _auto_detect() if preferred == Backend.AUTO
        else _resolve_explicit(preferred, allow_fallback)
    )

    logger.info("Device selected: %s", device)
    return device


def device_info(device: torch.device) -> dict[str, object]:
    """Return human-readable device metadata for logging and telemetry.

    Useful for recording hardware context alongside experiment results so
    that performance numbers are always interpretable.
    """
    info: dict[str, object] = {
        "device": str(device),
        "backend": device.type,
        "torch_version": torch.__version__,
    }

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["name"] = props.name
        info["vram_gb"] = round(props.total_memory / 1e9, 2)
        info["sm_count"] = props.multi_processor_count

    if device.type == "mps":
        # MPS does not expose hardware props via torch today; log what we can.
        info["name"] = "Apple Silicon GPU (MPS)"

    return info


# ── Internal helpers ──────────────────────────────────────────────────────────

def _auto_detect() -> torch.device:
    """Probe backends in priority order and return the first available."""
    if torch.cuda.is_available():
        d = torch.device("cuda")
        logger.debug("CUDA available — %s", torch.cuda.get_device_name(0))
        return d

    # torch.backends.mps.is_available() returns True on Apple Silicon with
    # macOS ≥ 12.3 and torch ≥ 1.12, but full op coverage landed in 2.0.
    if torch.backends.mps.is_available():
        logger.debug("MPS available — Apple Silicon GPU selected")
        return torch.device("mps")

    logger.debug("No hardware accelerator found — using CPU")
    return torch.device("cpu")


def _resolve_explicit(backend: str, allow_fallback: bool) -> torch.device:
    """Return *backend* as a device, or fall back to CPU with a warning."""
    if _is_available(backend):
        return torch.device(backend)

    msg = f"Requested backend '{backend}' is not available on this system."
    if not allow_fallback:
        raise RuntimeError(msg + " Set allow_fallback: true to degrade to CPU.")

    logger.warning("%s Falling back to CPU.", msg)
    return torch.device("cpu")


def _is_available(backend: str) -> bool:
    """Return True if the given backend is usable on the current machine."""
    if backend == Backend.CUDA:
        return torch.cuda.is_available()
    if backend == Backend.MPS:
        return torch.backends.mps.is_available()
    if backend == Backend.CPU:
        return True
    return False
