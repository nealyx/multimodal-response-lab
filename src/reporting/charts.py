"""Matplotlib chart generators for session analytics.

All functions return a matplotlib Figure or None (if matplotlib is unavailable
or there is insufficient data).  Charts use the Agg backend so they can be
generated without a display (server-side, CI, etc.).

Chart inventory
---------------
  engagement_timeline    — smoothed score over time, state background shading,
                           stimulus onset markers, threshold lines
  state_distribution     — horizontal bar chart of state fractions
  signal_channels        — gaze H/V, head yaw/pitch, blink rate over time
  stimulus_comparison    — mean during vs. baseline score per stimulus type
  reaction_latency       — per-trial key-press latencies (only if responses exist)
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    log.warning("matplotlib not available — charts will be skipped")

# State colour palette (BGR in OpenCV, but RGB here for matplotlib)
_STATE_COLORS = {
    "focused":    "#27AE60",
    "drifting":   "#F1C40F",
    "distracted": "#E67E22",
    "fatigued":   "#E74C3C",
    "unreliable": "#95A5A6",
    "unknown":    "#BDC3C7",
}
_THRESHOLD_FOCUSED  = 0.72
_THRESHOLD_DRIFTING = 0.45


def _require_mpl(func):
    """Decorator: returns None immediately if matplotlib is unavailable."""
    def wrapper(*args, **kwargs):
        if not HAS_MATPLOTLIB:
            return None
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            log.warning("Chart generation failed (%s): %s", func.__name__, exc)
            return None
    wrapper.__name__ = func.__name__
    return wrapper


# ── Individual chart generators ───────────────────────────────────────────────

@_require_mpl
def engagement_timeline(
    samples: List[Any],   # List[BehavioralSample]
    stimulus_events: List[Any] = None,
    start_ts: float = 0.0,
    title: str = "Engagement Score Timeline",
) -> Optional["Figure"]:
    """Line chart of smoothed engagement score with state shading and stimulus markers."""
    if not samples:
        return None

    times  = [s.timestamp - start_ts for s in samples]
    scores = [s.smoothed_score       for s in samples]
    states = [s.engagement_state     for s in samples]

    fig, ax = _plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    # State background shading (per-segment)
    for i in range(len(times) - 1):
        col = _STATE_COLORS.get(states[i], "#BDC3C7")
        ax.axvspan(times[i], times[i + 1], alpha=0.12, color=col, linewidth=0)

    # Threshold lines
    ax.axhline(_THRESHOLD_FOCUSED,  color="#27AE60", linewidth=0.8, linestyle="--",
               alpha=0.7, label=f"focused ≥ {_THRESHOLD_FOCUSED}")
    ax.axhline(_THRESHOLD_DRIFTING, color="#F1C40F", linewidth=0.8, linestyle="--",
               alpha=0.7, label=f"drifting ≥ {_THRESHOLD_DRIFTING}")

    # Score line
    ax.plot(times, scores, color="#3498DB", linewidth=1.5, label="smoothed score")

    # Stimulus onset markers
    if stimulus_events:
        for ev in stimulus_events:
            rel_t = ev.onset_ts - start_ts
            ax.axvline(rel_t, color="#E8DAEF", linewidth=1.0, alpha=0.6,
                       linestyle=":", zorder=5)
            ax.text(rel_t + 0.1, 0.95, ev.stimulus_id.split("_")[0][:5],
                    color="#E8DAEF", fontsize=6, va="top",
                    transform=ax.get_xaxis_transform(), alpha=0.8)

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)", color="#ECF0F1", fontsize=10)
    ax.set_ylabel("Score", color="#ECF0F1", fontsize=10)
    ax.set_title(title, color="#ECF0F1", fontsize=12, pad=10)
    ax.tick_params(colors="#BDC3C7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2C3E50")

    legend = ax.legend(loc="lower right", fontsize=8,
                       facecolor="#1A1A2E", edgecolor="#2C3E50", labelcolor="#BDC3C7")
    fig.tight_layout(pad=1.5)
    return fig


@_require_mpl
def state_distribution(
    metrics: Any,   # EngagementMetrics
    title: str = "Engagement State Distribution",
) -> Optional["Figure"]:
    """Horizontal stacked bar showing session state fractions."""
    labels = ["focused", "drifting", "distracted", "fatigued", "unreliable"]
    values = [
        metrics.focused_fraction,
        metrics.drifting_fraction,
        metrics.distracted_fraction,
        metrics.fatigued_fraction,
        metrics.unreliable_fraction,
    ]

    fig, ax = _plt.subplots(figsize=(8, 2.5))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    left = 0.0
    for label, val in zip(labels, values):
        if val > 0:
            ax.barh(0, val, left=left, color=_STATE_COLORS[label],
                    height=0.5, label=f"{label}  {val*100:.1f}%")
            if val > 0.04:
                ax.text(left + val / 2, 0, f"{val*100:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
        left += val

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel("Fraction of session", color="#ECF0F1", fontsize=10)
    ax.set_title(title, color="#ECF0F1", fontsize=12, pad=10)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors="#BDC3C7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2C3E50")

    legend = ax.legend(loc="upper right", fontsize=8, ncol=3,
                       facecolor="#1A1A2E", edgecolor="#2C3E50", labelcolor="#BDC3C7",
                       bbox_to_anchor=(1.0, 1.6))
    fig.tight_layout(pad=1.5)
    return fig


@_require_mpl
def signal_channels(
    samples: List[Any],
    start_ts: float = 0.0,
    title: str = "Signal Channels Over Time",
) -> Optional["Figure"]:
    """Three-panel chart: on-screen fraction, head yaw, blink rate."""
    if not samples:
        return None

    times     = [s.timestamp - start_ts for s in samples]
    on_screen = [1.0 if s.is_on_screen else 0.0 for s in samples]
    yaw       = [s.head_yaw    for s in samples]
    blink_rt  = [s.blink_rate  for s in samples]

    fig, axes = _plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    fig.patch.set_facecolor("#1A1A2E")

    configs = [
        (axes[0], on_screen, "#3498DB", "On-screen",  (0, 1.05)),
        (axes[1], yaw,       "#E67E22", "Head Yaw (°)", None),
        (axes[2], blink_rt,  "#9B59B6", "Blink rate/min", None),
    ]

    for ax, data, col, ylabel, ylim in configs:
        ax.set_facecolor("#16213E")
        ax.plot(times, data, color=col, linewidth=1.0)
        ax.set_ylabel(ylabel, color="#ECF0F1", fontsize=9)
        ax.tick_params(colors="#BDC3C7", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2C3E50")
        if ylim:
            ax.set_ylim(*ylim)

    axes[2].set_xlabel("Time (s)", color="#ECF0F1", fontsize=10)
    axes[0].set_title(title, color="#ECF0F1", fontsize=12, pad=6)
    # Mark zero for yaw
    axes[1].axhline(0, color="#BDC3C7", linewidth=0.5, linestyle="--", alpha=0.5)

    fig.tight_layout(pad=1.5)
    return fig


@_require_mpl
def stimulus_comparison(
    stimulus_types: Dict[str, Any],   # Dict[str, StimulusTypeMetrics]
    title: str = "Per-Stimulus Score: Baseline vs. During",
) -> Optional["Figure"]:
    """Grouped bar chart comparing mean baseline and during scores by stimulus type."""
    if not stimulus_types:
        return None

    types      = list(stimulus_types.keys())
    baselines  = [stimulus_types[t].mean_baseline_score for t in types]
    durings    = [stimulus_types[t].mean_during_score    for t in types]

    # Shorten type names for display
    short = [t.replace("Stimulus", "").replace("Prompt", "Pr.") for t in types]

    x   = range(len(types))
    w   = 0.35

    fig, ax = _plt.subplots(figsize=(max(6, len(types) * 2), 4))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    bars_b = ax.bar([xi - w/2 for xi in x], baselines, w,
                    label="Baseline", color="#2980B9", alpha=0.85)
    bars_d = ax.bar([xi + w/2 for xi in x], durings,   w,
                    label="During", color="#27AE60", alpha=0.85)

    for bar in list(bars_b) + list(bars_d):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8, color="#ECF0F1")

    ax.axhline(_THRESHOLD_FOCUSED, color="#27AE60", linewidth=0.8,
               linestyle="--", alpha=0.5)
    ax.axhline(_THRESHOLD_DRIFTING, color="#F1C40F", linewidth=0.8,
               linestyle="--", alpha=0.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(short, color="#ECF0F1", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Engagement score", color="#ECF0F1", fontsize=10)
    ax.set_title(title, color="#ECF0F1", fontsize=12, pad=10)
    ax.tick_params(colors="#BDC3C7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2C3E50")
    ax.legend(facecolor="#1A1A2E", edgecolor="#2C3E50", labelcolor="#BDC3C7", fontsize=9)

    fig.tight_layout(pad=1.5)
    return fig


@_require_mpl
def reaction_latency(
    stimulus_types: Dict[str, Any],
    title: str = "Reaction Latency by Stimulus Type",
) -> Optional["Figure"]:
    """Scatter/bar chart of mean reaction latency per stimulus type."""
    has_lat = {k: v for k, v in stimulus_types.items()
               if v.mean_latency_ms is not None}
    if not has_lat:
        return None

    types    = list(has_lat.keys())
    latencies = [has_lat[t].mean_latency_ms for t in types]
    stds      = [has_lat[t].std_latency_ms or 0.0 for t in types]
    short     = [t.replace("Stimulus", "").replace("Prompt", "Pr.") for t in types]

    fig, ax = _plt.subplots(figsize=(max(5, len(types) * 2), 4))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    bars = ax.bar(range(len(types)), latencies, color="#E74C3C", alpha=0.85,
                  yerr=stds, capsize=5, error_kw={"ecolor": "#ECF0F1", "elinewidth": 1.2})

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(stds) * 0.1 + 10,
                f"{val:.0f} ms", ha="center", va="bottom",
                fontsize=9, color="#ECF0F1")

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(short, color="#ECF0F1", fontsize=9)
    ax.set_ylabel("Mean latency (ms)", color="#ECF0F1", fontsize=10)
    ax.set_title(title, color="#ECF0F1", fontsize=12, pad=10)
    ax.tick_params(colors="#BDC3C7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2C3E50")

    fig.tight_layout(pad=1.5)
    return fig


# ── Batch generator ───────────────────────────────────────────────────────────

def generate_all(log: Any, metrics: Any) -> Dict[str, Any]:
    """Generate all charts and return them as a name → Figure dict.

    *log*     — SessionLog
    *metrics* — SessionMetrics
    """
    charts: Dict[str, Any] = {}

    fig = engagement_timeline(
        log.samples, log.stimulus_events, log.start_ts,
    )
    if fig is not None:
        charts["engagement_timeline"] = fig

    fig = state_distribution(metrics.engagement)
    if fig is not None:
        charts["state_distribution"] = fig

    fig = signal_channels(log.samples, log.start_ts)
    if fig is not None:
        charts["signal_channels"] = fig

    if metrics.stimulus_types:
        fig = stimulus_comparison(metrics.stimulus_types)
        if fig is not None:
            charts["stimulus_comparison"] = fig

        fig = reaction_latency(metrics.stimulus_types)
        if fig is not None:
            charts["reaction_latency"] = fig

    return charts


# ── PNG serialisation helper ──────────────────────────────────────────────────

def figure_to_png_bytes(fig: Any) -> bytes:
    """Render a matplotlib Figure to PNG bytes (for HTML embedding)."""
    if not HAS_MATPLOTLIB:
        return b""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()
