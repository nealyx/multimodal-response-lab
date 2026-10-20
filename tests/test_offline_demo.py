"""End-to-end checks of the public demo without cameras, models, or networking."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from demo_offline import build_synthetic_session
from src.reporting.html_report import generate_html
from src.reporting.reporter import SessionReporter


def test_synthetic_session_is_repeatable_and_keeps_quality_separate():
    first = build_synthetic_session()
    assert asdict(first) == asdict(build_synthetic_session())
    assert first.config["provenance"]["synthetic"] is True
    assert len(first.samples) == 600
    assert [e.latency_ms for e in first.response_events] == [350.0, 450.0]
    assert all(e.metadata["synthetic"] for e in first.stimulus_events + first.response_events)
    assert first.samples[0].confidence == 0.0
    assert first.samples[0].smoothed_score > 0.0
    missing = [s for s in first.samples if 45.5 <= s.timestamp < 50.0]
    assert all(s.confidence == 0.0 for s in missing)
    assert all(s.engagement_state == "unreliable" for s in missing)
    assert any(s.smoothed_score > 0.0 for s in missing)


def test_synthetic_report_discloses_fixture_origin():
    session = build_synthetic_session()
    html = generate_html(SessionReporter().compute(session), {})
    assert "SYNTHETIC DEMONSTRATION" in html
    assert "no participant recording" in html
    assert "does not establish causation" in html
    session.config = {}
    assert "SYNTHETIC DEMONSTRATION" not in generate_html(SessionReporter().compute(session), {})


def test_report_open_accepts_relative_output_paths(tmp_path, monkeypatch):
    import argparse
    import analyze_session
    from src.stimulus.exporter import SessionExporter

    monkeypatch.chdir(tmp_path)
    session = build_synthetic_session()
    SessionExporter.to_json(session, "session/session_log.json", session.trial_summaries)
    SessionExporter.to_csv(session, "session/samples.csv", "session/events.csv")
    opened = []
    monkeypatch.setattr(analyze_session.webbrowser, "open", opened.append)
    analyze_session.run(argparse.Namespace(
        input="session", output_dir="reports", no_charts=True,
        no_html=False, open=True, profile=None,
    ))
    expected = tmp_path / "reports" / session.session_id / "report.html"
    assert expected.exists()
    assert opened == [expected.as_uri()]


def test_demo_runs_without_optional_vision_imports_or_network(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "demo"
    script = r"""
import importlib.abc
import runpy
import socket
import sys

class NoVisionImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"cv2", "mediapipe", "torch", "sounddevice", "pyaudio"}:
            raise AssertionError("Offline demo imported optional hardware stack: " + fullname)

def no_network(*args, **kwargs):
    raise AssertionError("Offline demo attempted network access")

sys.meta_path.insert(0, NoVisionImports())
socket.socket = no_network
socket.create_connection = no_network
sys.argv = ["demo_offline.py", "--output-dir", sys.argv[1]]
runpy.run_path("demo_offline.py", run_name="__main__")
"""
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = subprocess.run(
        [sys.executable, "-c", script, str(output)],
        cwd=root, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "demo_manifest.json").read_text())
    report_dir = output / "reports" / manifest["session_id"]
    metrics = json.loads((report_dir / "session_metrics.json").read_text())
    assert metrics["n_frames"] == 600
    assert metrics["n_trials"] == 2
    assert metrics["config_snapshot"]["provenance"]["synthetic"] is True
    assert metrics["reaction_latencies_ms"] == [350.0, 450.0]
    assert len(list((report_dir / "charts").glob("*.png"))) == 5
    html = (report_dir / "report.html").read_text()
    assert "SYNTHETIC DEMONSTRATION" in html
    assert "data:image/png;base64," in html
    rerun = subprocess.run(
        [sys.executable, "demo_offline.py", "--output-dir", str(output)],
        cwd=root, env=env, capture_output=True, text=True, timeout=10,
    )
    assert rerun.returncode != 0
    assert "already exists" in rerun.stderr
    assert json.loads((output / "demo_manifest.json").read_text()) == manifest
