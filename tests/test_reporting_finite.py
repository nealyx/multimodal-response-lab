"""Saved numeric inputs must not turn a successful report into NaN/Infinity."""

import csv
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from demo_offline import build_synthetic_session
from src.reporting.loader import load_session_dir, load_session_json
from src.reporting.reporter import SessionReporter
from src.stimulus.exporter import SessionExporter


@pytest.fixture
def saved_session(tmp_path):
    session = build_synthetic_session()
    source = tmp_path / "session"
    SessionExporter.to_json(session, str(source / "session_log.json"), session.trial_summaries)
    SessionExporter.to_csv(session, str(source / "samples.csv"), str(source / "events.csv"))
    return source, session


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_cli(source, output):
    return subprocess.run(
        [sys.executable, "analyze_session.py", str(source),
         "--output-dir", str(output), "--no-charts"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "MPLBACKEND": "Agg"},
        text=True, capture_output=True, timeout=30,
    )


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def test_nonfinite_csv_fields_are_skipped_with_row_warnings(saved_session, caplog):
    source, session = saved_session
    fields = ["timestamp", "smoothed_score", "confidence", "focused_fraction",
              "gaze_h", "gaze_v", "head_yaw", "head_pitch", "mean_ear", "blink_rate"]
    invalid = ["NaN", "inf", "-Infinity", "1e400", "-1e400"]
    rows = []
    for index, field in enumerate(fields):
        row = asdict(session.samples[index])
        row[field] = invalid[index % len(invalid)]
        rows.append(row)
    valid = [replace(session.samples[-2], smoothed_score=0.0, gaze_h=0.0,
                     head_yaw=-12.0, head_pitch=-3.5), session.samples[-1]]
    write_rows(source / "samples.csv", rows + [asdict(sample) for sample in valid])
    before = (source / "samples.csv").read_bytes()

    loaded = load_session_dir(str(source))

    assert loaded.samples == valid
    assert (source / "samples.csv").read_bytes() == before
    assert caplog.text.count("Skipping row") == len(fields)
    for line in range(2, len(fields) + 2):
        assert f"Skipping row {line} in samples.csv" in caplog.text
    assert "finite" in caplog.text


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"])
def test_nonfinite_json_numbers_are_rejected_even_in_nested_config(saved_session, token):
    source, _ = saved_session
    path = source / "session_log.json"
    document = json.loads(path.read_text())
    document["config"]["nested"] = [{"threshold": "NUMERIC_TOKEN"}]
    path.write_text(json.dumps(document).replace('"NUMERIC_TOKEN"', token))

    with pytest.raises(ValueError, match="finite"):
        load_session_json(str(path))


@pytest.mark.parametrize("field", ["start_ts", "end_ts"])
@pytest.mark.parametrize("value", ["-1e400", 10 ** 400, -(10 ** 400)],
                         ids=["overflow-string", "huge-integer", "negative-huge-integer"])
def test_unrepresentable_session_timestamps_are_rejected(saved_session, field, value):
    source, _ = saved_session
    path = source / "session_log.json"
    document = json.loads(path.read_text())
    document[field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="finite"):
        load_session_json(str(path))


def test_unrepresentable_event_numbers_follow_existing_skip_policy(saved_session, caplog):
    source, session = saved_session
    path = source / "session_log.json"
    document = json.loads(path.read_text())
    # Required and optional numeric fields are both checked; valid records stay.
    for collection, field, value in [
        ("stimulus_events", "onset_ts", "NaN"),
        ("stimulus_events", "offset_ts", "Infinity"),
        ("response_events", "latency_ms", "1e400"),
        ("trial_summaries", "baseline_score", "-Infinity"),
        ("trial_summaries", "recovery_s", "NaN"),
        ("stimulus_events", "duration_ms", 10 ** 400),
        ("response_events", "response_ts", -(10 ** 400)),
        ("trial_summaries", "reaction_latency_ms", 10 ** 400),
    ]:
        bad = dict(document[collection][0])
        bad[field] = value
        document[collection].append(bad)
    path.write_text(json.dumps(document))

    loaded = load_session_json(str(path))

    assert loaded.stimulus_events == session.stimulus_events
    assert loaded.response_events == session.response_events
    assert loaded.trial_summaries == session.trial_summaries
    warnings = [record.message for record in caplog.records if "Skipping malformed" in record.message]
    assert len(warnings) == 8
    assert all("finite" in warning for warning in warnings)


def test_finite_zero_negative_and_absent_optional_values_are_preserved(saved_session):
    source, _ = saved_session
    path = source / "session_log.json"
    document = json.loads(path.read_text())
    document.update(start_ts=-1.0, end_ts=0.0)
    document["stimulus_events"][0]["offset_ts"] = None
    document["response_events"][0]["latency_ms"] = 0.0
    document["trial_summaries"][0].update(
        reaction_latency_ms=0.0, recovery_s="", baseline_score=0.0,
    )
    path.write_text(json.dumps(document))

    loaded = load_session_json(str(path))

    assert loaded.start_ts == -1.0 and loaded.end_ts == 0.0
    assert loaded.stimulus_events[0].offset_ts is None
    assert loaded.response_events[0].latency_ms == 0.0
    assert loaded.trial_summaries[0].reaction_latency_ms == 0.0
    assert loaded.trial_summaries[0].recovery_s is None
    assert loaded.trial_summaries[0].baseline_score == 0.0


def test_cli_warns_and_reports_only_valid_samples_as_standard_json(saved_session, tmp_path):
    source, session = saved_session
    rows = [asdict(sample) for sample in session.samples]
    rows[0]["smoothed_score"] = "NaN"
    write_rows(source / "samples.csv", rows)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "reports"

    result = run_cli(source, output)

    assert result.returncode == 0, result.stderr
    assert "Skipping row 2 in samples.csv" in result.stderr
    report = output / session.session_id
    metrics = json.loads((report / "session_metrics.json").read_text(), parse_constant=reject_constant)
    session.samples = session.samples[1:]
    expected = asdict(SessionReporter().compute(session))
    assert {k: v for k, v in metrics.items() if k != "generated_at"} == {
        k: v for k, v in expected.items() if k != "generated_at"
    }
    assert metrics["n_frames"] == 599
    assert "nan" not in result.stdout.lower()
    assert '<div class="value">nan' not in (report / "report.html").read_text().lower()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_cli_rejects_nonfinite_json_before_creating_report(saved_session, tmp_path):
    source, _ = saved_session
    path = source / "session_log.json"
    document = json.loads(path.read_text())
    document["config"]["threshold"] = float("inf")
    path.write_text(json.dumps(document))
    output = tmp_path / "reports"

    result = run_cli(source, output)

    assert result.returncode != 0
    assert "finite" in result.stderr
    assert not output.exists()
