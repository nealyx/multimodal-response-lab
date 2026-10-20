"""Saved labels remain literal text in HTML, without changing recorded data."""

import argparse
from dataclasses import asdict, replace
from html.parser import HTMLParser
import json

import pytest

import analyze_session
from demo_offline import build_synthetic_session
from src.reporting.html_report import generate_html
from src.reporting.reporter import SessionReporter
from src.stimulus.exporter import SessionExporter


LABEL = 'A <script>window.reportLabel = true</script> & "B" — Δ'


class ParsedReport(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self.text = []
        self.feed(source)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


@pytest.mark.parametrize("field", ["session_id", "experiment_name"])
def test_session_labels_are_text_in_title_and_body(field):
    metrics = SessionReporter().compute(build_synthetic_session())
    setattr(metrics, field, LABEL)
    before = asdict(metrics)
    report = ParsedReport(generate_html(metrics, {}))
    assert LABEL in "".join(report.text)
    assert "script" not in [tag for tag, attrs in report.elements]
    assert asdict(metrics) == before


def test_trial_and_stimulus_labels_cannot_create_html_elements():
    session = build_synthetic_session()
    session.trial_summaries = [replace(
        session.trial_summaries[0], stimulus_id=LABEL,
        stimulus_type=LABEL, response_type=LABEL,
    )]
    metrics = SessionReporter().compute(session)
    rows = [asdict(session.trial_summaries[0])]
    before_metrics, before_rows = asdict(metrics), [dict(rows[0])]
    report = ParsedReport(generate_html(metrics, {}, rows))
    # One group label and three trial-table labels, all displayed literally.
    assert "".join(report.text).count(LABEL) == 4
    assert "script" not in [tag for tag, attrs in report.elements]
    assert asdict(metrics) == before_metrics and rows == before_rows


def test_chart_labels_preserve_quotes_without_creating_attributes():
    metrics = SessionReporter().compute(build_synthetic_session())
    name = 'angle <label> & " onerror="synthetic-label'
    expected = name.replace("_", " ").title()
    report = ParsedReport(generate_html(metrics, {name: b"synthetic PNG bytes"}))
    assert expected in "".join(report.text)
    images = [attrs for tag, attrs in report.elements if tag == "img"]
    assert len(images) == 1
    assert set(images[0]) == {"class", "src", "alt"}
    assert images[0]["alt"] == expected
    assert images[0]["src"].startswith("data:image/png;base64,")
    assert "label" not in [tag for tag, attrs in report.elements]


def test_saved_session_cli_keeps_literal_labels_and_original_metrics(tmp_path):
    session = build_synthetic_session()
    session.experiment_name = LABEL
    source = tmp_path / "session"
    SessionExporter.to_json(session, str(source / "session_log.json"), session.trial_summaries)
    SessionExporter.to_csv(session, str(source / "samples.csv"), str(source / "events.csv"))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "reports"
    analyze_session.run(argparse.Namespace(
        input=str(source), output_dir=str(output), no_charts=True,
        no_html=False, open=False, profile=None,
    ))
    report_dir = output / session.session_id
    report = ParsedReport((report_dir / "report.html").read_text())
    metrics = json.loads((report_dir / "session_metrics.json").read_text())
    assert LABEL in "".join(report.text)
    assert not {"script", "iframe", "img"} & {tag for tag, attrs in report.elements}
    assert metrics["experiment_name"] == LABEL
    expected = asdict(SessionReporter().compute(session))
    assert {k: v for k, v in metrics.items() if k != "generated_at"} == {
        k: v for k, v in expected.items() if k != "generated_at"
    }
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
