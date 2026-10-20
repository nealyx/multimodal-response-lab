"""Missing samples are not measurements of zero or a dominant state."""

import argparse
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

import analyze_session
from demo_offline import build_synthetic_session
from src.reporting.charts import generate_all
from src.reporting.html_report import generate_html
from src.reporting.reporter import SessionReporter
from src.stimulus.events import SessionLog, TrialSummary
from src.stimulus.exporter import SessionExporter


def empty_session():
    return SessionLog("empty", "Synthetic empty session", 0.0, 0.0, {})


def zero_session():
    session = build_synthetic_session()
    sample = replace(
        session.samples[0], smoothed_score=0.0, engagement_state="distracted",
        is_on_screen=False, gaze_h=0.0, gaze_v=0.0, head_yaw=0.0,
        head_pitch=0.0, blink_rate=0.0, mean_ear=0.0, is_fatigued=False,
    )
    session.samples = [sample, replace(sample, frame_index=1, timestamp=1.0)]
    session.trial_summaries = [
        TrialSummary("zero-latency", 0, "reaction_prompt", 0.0, 0.0,
                     0.0, "key_press", True, 0.0, 0.0, 0.0),
    ]
    return session


def test_empty_session_displays_unavailable_instead_of_zero_or_state(capsys):
    session = empty_session()
    metrics = SessionReporter().compute(session)
    html = generate_html(metrics, {})
    analyze_session._print_summary(metrics)
    console = capsys.readouterr().out
    assert "No samples loaded" in html
    assert '<div class="value">Unavailable</div>' in html
    assert 'class="badge badge-focused"' not in html
    assert "<td>Mean score</td>" not in html
    assert "0.000 ± 0.000" not in console
    assert "Sample metrics" in console and "unavailable" in console
    assert generate_all(session, metrics) == {}
    # Existing consumers retain the schema; zero-count summaries are sentinels.
    assert metrics.n_frames == metrics.engagement.score_summary.n == 0
    assert metrics.engagement.score_summary.mean == 0.0


def test_json_only_report_keeps_events_but_omits_sample_charts_and_scores(tmp_path, capsys):
    session = build_synthetic_session()
    source = tmp_path / "session_log.json"
    SessionExporter.to_json(session, str(source), session.trial_summaries)
    output = tmp_path / "reports"
    analyze_session.run(argparse.Namespace(
        input=str(source), output_dir=str(output), no_charts=False,
        no_html=False, open=False, profile=None,
    ))
    report = output / session.session_id
    html = (report / "report.html").read_text()
    metrics = json.loads((report / "session_metrics.json").read_text())
    console = capsys.readouterr().out
    assert "No samples loaded" in html
    assert "400 ms" in html and "350 ms" in html and "450 ms" in html
    assert 'class="badge badge-focused"' not in html
    assert "<td>Unavailable</td>" in html
    assert set(p.name for p in (report / "charts").iterdir()) == {"reaction_latency.png"}
    assert not (report / "timeline.csv").exists()
    assert metrics["n_frames"] == 0
    assert metrics["n_trials"] == metrics["n_responses"] == 2
    assert metrics["reaction_latencies_ms"] == [350.0, 450.0]
    assert "Δscore=unavailable" in console
    assert "0.000 ± 0.000" not in console


def test_zero_valued_measurements_remain_numeric(capsys):
    session = zero_session()
    metrics = SessionReporter().compute(session)
    html = generate_html(metrics, {}, [{
        "trial_index": 0, "stimulus_id": "zero-latency", "stimulus_type": "reaction_prompt",
        "reaction_latency_ms": 0.0, "response_type": "key_press", "hit": True,
        "baseline_score": 0.0, "during_score": 0.0, "recovery_s": 0.0,
    }])
    analyze_session._print_summary(metrics)
    console = capsys.readouterr().out
    assert "No samples loaded" not in html
    assert '<div class="value">0.000</div>' in html
    assert 'class="badge badge-distracted"' in html
    assert "<td>0 ms</td>" in html and "<td>0.0 s</td>" in html
    assert "0.000 ± 0.000" in console
    assert "RT=0 ms" in console
    assert metrics.n_frames == metrics.engagement.score_summary.n == 2


@pytest.mark.parametrize("has_samples", [False, True])
def test_baseline_comparison_requires_samples_but_accepts_measured_zero(monkeypatch, capsys, has_samples):
    session = zero_session() if has_samples else empty_session()
    metrics = SessionReporter().compute(session)
    calls = []

    def compare(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(summary_lines=lambda: ["comparison calculated"])

    monkeypatch.setattr(analyze_session.BaselineComparator, "compare", compare)
    analyze_session._print_baseline_comparison(metrics, SimpleNamespace(profile_id="synthetic"), "session")
    if has_samples:
        assert calls[0]["blink_rate"] == calls[0]["engagement_score"] == 0.0
        assert "comparison calculated" in capsys.readouterr().out
    else:
        assert calls == []
        assert "unavailable" in capsys.readouterr().out
