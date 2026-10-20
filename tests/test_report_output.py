"""Reports must not mix artifacts from separate analysis invocations."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import analyze_session
from demo_offline import build_synthetic_session
from src.stimulus.exporter import SessionExporter


@pytest.fixture
def saved_session(tmp_path):
    session = build_synthetic_session()
    directory = tmp_path / "session"
    SessionExporter.to_json(session, str(directory / "session_log.json"), session.trial_summaries)
    SessionExporter.to_csv(session, str(directory / "samples.csv"), str(directory / "events.csv"))
    return directory, session.session_id


def report_args(source, output):
    return argparse.Namespace(
        input=str(source), output_dir=str(output), no_charts=True,
        no_html=False, open=False, profile=None,
    )


def contents(directory):
    return {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("json_only", [False, True])
def test_existing_report_is_preserved_when_inputs_or_output_options_change(saved_session, tmp_path, json_only):
    source, session_id = saved_session
    output = tmp_path / "reports"
    analyze_session.run(report_args(source, output))
    report = output / session_id
    (report / "charts" / "previous.png").write_bytes(b"previous chart")
    before = contents(report)

    # JSON-only deliberately omits samples. Skipping HTML/charts must not leave
    # the previous full report beside new zero-frame metrics.
    args = report_args(source / "session_log.json" if json_only else source, output)
    args.no_html = True
    with pytest.raises(FileExistsError, match="Choose a new --output-dir"):
        analyze_session.run(args)
    assert contents(report) == before


def test_cli_explains_existing_destination_without_traceback(saved_session, tmp_path):
    source, session_id = saved_session
    output = tmp_path / "reports"
    destination = output / session_id
    destination.mkdir(parents=True)
    (destination / "keep.txt").write_text("existing report", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "analyze_session.py", str(source), "--output-dir", str(output), "--no-charts"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "Choose a new --output-dir" in result.stderr
    assert "Traceback" not in result.stderr
    assert contents(destination) == {Path("keep.txt"): b"existing report"}


def test_new_output_root_and_other_session_can_share_an_existing_parent(saved_session, tmp_path):
    source, session_id = saved_session
    output = tmp_path / "reports"
    analyze_session.run(report_args(source, output))
    original = contents(output / session_id)

    alternate = tmp_path / "reanalysis"
    analyze_session.run(report_args(source, alternate))
    assert json.loads((alternate / session_id / "session_metrics.json").read_text())["n_frames"] == 600
    assert contents(output / session_id) == original

    log_path = source / "session_log.json"
    document = json.loads(log_path.read_text())
    document["session_id"] = "second-synthetic-session"
    log_path.write_text(json.dumps(document), encoding="utf-8")
    analyze_session.run(report_args(source, output))
    assert (output / "second-synthetic-session" / "report.html").is_file()
    assert contents(output / session_id) == original
