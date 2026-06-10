"""Tests for Day 13: multi-session dataset management.

Coverage:
  TestParticipant         ( 8 tests) — create, save/load, try_load, serialization
  TestSessionEntry        ( 7 tests) — to_dict/from_dict, is_complete, new fields
  TestSessionCatalog      ( 6 tests) — add, get, remove, for_participant, len
  TestDatasetManifest     ( 8 tests) — create, save/load, add_participant/session
  TestSessionIngestor     (14 tests) — ingest with real tmp files, on_screen, blink
  TestDatasetRegistry     (11 tests) — register, ingest, rebuild, directory structure
  TestDatasetStats        (10 tests) — compute stats, on_screen, blink, missing counts
  TestDatasetChecker      ( 8 tests) — errors, warnings, info, summary
  TestDatasetExporter     ( 7 tests) — JSON structure, CSV rows, new columns
  Total                   (79 tests)
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import List

import pytest

from src.dataset.catalog import SessionCatalog, SessionEntry
from src.dataset.checker import DatasetChecker, QualityIssue
from src.dataset.exporter import DatasetExporter
from src.dataset.ingestor import SessionIngestor
from src.dataset.manifest import DatasetManifest
from src.dataset.participant import Participant
from src.dataset.registry import DatasetRegistry
from src.dataset.stats import DatasetStats, DatasetStatsComputer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_entry(
    session_id:       str   = "sess_001",
    participant_id:   str   = "p_001",
    n_frames:         int   = 500,
    n_windows:        int   = 20,
    duration_s:       float = 60.0,
    has_session_log:  bool  = True,
    has_samples:      bool  = True,
    has_embeddings:   bool  = True,
    has_labels:       bool  = False,
    has_calibration:  bool  = False,
    mean_eng:         float = 0.72,
    on_screen_frac:   float = 0.85,
    mean_blink_rate:  float = 15.0,
    eng_fracs:        dict | None = None,
    label_summary:    dict | None = None,
) -> SessionEntry:
    return SessionEntry(
        session_id=            session_id,
        participant_id=        participant_id,
        ingested_at=           "2026-01-01T00:00:00+00:00",
        has_session_log=       has_session_log,
        has_samples=           has_samples,
        has_embeddings=        has_embeddings,
        has_labels=            has_labels,
        has_calibration=       has_calibration,
        n_frames=              n_frames,
        n_windows=             n_windows,
        duration_s=            duration_s,
        experiment_name=       "test",
        mean_engagement_score= mean_eng,
        on_screen_fraction=    on_screen_frac,
        mean_blink_rate=       mean_blink_rate,
        engagement_fractions=  eng_fracs if eng_fracs is not None
                               else {"focused": 0.6, "distracted": 0.2},
        label_summary=         label_summary if label_summary is not None else {},
    )


def _make_session_dir(tmp_path: Path, session_id: str = "20260101_120000") -> Path:
    """Create a minimal session directory with session_log.json + samples.csv."""
    session_dir = tmp_path / session_id
    session_dir.mkdir()

    # session_log.json
    session_log = {
        "session_id":      session_id,
        "experiment_name": "test_experiment",
        "start_ts":        1000.0,
        "end_ts":          1120.0,
        "config":          {},
    }
    (session_dir / "session_log.json").write_text(
        json.dumps(session_log), encoding="utf-8"
    )

    # samples.csv (5 rows, includes is_on_screen and blink_rate)
    with open(session_dir / "samples.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "frame_index", "timestamp", "engagement_state", "smoothed_score",
            "is_on_screen", "blink_rate",
        ])
        writer.writeheader()
        for i in range(5):
            writer.writerow({
                "frame_index": i, "timestamp": 1000.0 + i,
                "engagement_state": "focused", "smoothed_score": 0.75,
                "is_on_screen": "True", "blink_rate": "14.0",
            })

    return session_dir


def _make_embeddings_csv(path: Path, n_windows: int = 10) -> None:
    """Write a minimal behavioral_windows.csv with *n_windows* rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from src.embedding.features import FEATURE_NAMES
    import numpy as np
    fieldnames = ["window_id", "session_id", "start_ts", "end_ts",
                  "n_frames", "dominant_state", "active_stimulus_ids"] + FEATURE_NAMES
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_windows):
            row = {f: "0.0" for f in FEATURE_NAMES}
            row.update({
                "window_id": f"win_{i:04d}", "session_id": "sess",
                "start_ts": str(float(i)), "end_ts": str(float(i + 2)),
                "n_frames": "60", "dominant_state": "focused",
                "active_stimulus_ids": "",
            })
            writer.writerow(row)


# ── TestParticipant ───────────────────────────────────────────────────────────

class TestParticipant:
    def test_create_generates_id_if_empty(self):
        p = Participant.create()
        assert len(p.participant_id) > 0

    def test_create_preserves_explicit_id(self):
        p = Participant.create(participant_id="participant_001")
        assert p.participant_id == "participant_001"

    def test_to_dict_round_trips(self):
        p = Participant.create(participant_id="p1", age_range="25-34", notes="test")
        p2 = Participant.from_dict(p.to_dict())
        assert p2.participant_id == "p1"
        assert p2.age_range      == "25-34"
        assert p2.notes          == "test"

    def test_save_load_roundtrip(self, tmp_path):
        p = Participant.create(participant_id="p_save")
        path = str(tmp_path / "participant.json")
        p.save(path)
        p2 = Participant.load(path)
        assert p2.participant_id == "p_save"

    def test_try_load_missing_returns_none(self):
        assert Participant.try_load("/nonexistent/path.json") is None

    def test_try_load_none_returns_none(self):
        assert Participant.try_load(None) is None

    def test_session_ids_initializes_empty(self):
        p = Participant.create()
        assert p.session_ids == []

    def test_created_at_is_nonempty(self):
        p = Participant.create()
        assert len(p.created_at) > 0


# ── TestSessionEntry ──────────────────────────────────────────────────────────

class TestSessionEntry:
    def test_is_complete_when_all_present(self):
        e = _make_entry()
        assert e.is_complete is True

    def test_is_complete_false_when_missing_embeddings(self):
        e = _make_entry(has_embeddings=False)
        assert e.is_complete is False

    def test_focused_fraction(self):
        e = _make_entry(eng_fracs={"focused": 0.70, "distracted": 0.30})
        assert e.focused_fraction == pytest.approx(0.70)

    def test_focused_fraction_zero_when_missing(self):
        e = _make_entry(eng_fracs={})
        assert e.focused_fraction == pytest.approx(0.0)

    def test_to_dict_from_dict_roundtrip(self):
        e = _make_entry(has_labels=True, label_summary={"engagement": "HIGH"})
        e2 = SessionEntry.from_dict(e.to_dict())
        assert e2.session_id  == e.session_id
        assert e2.has_labels  == True
        assert e2.label_summary["engagement"] == "HIGH"

    def test_n_windows_preserved(self):
        e = _make_entry(n_windows=42)
        e2 = SessionEntry.from_dict(e.to_dict())
        assert e2.n_windows == 42


# ── TestSessionCatalog ────────────────────────────────────────────────────────

class TestSessionCatalog:
    def test_add_increases_len(self):
        c = SessionCatalog()
        c.add(_make_entry("s1"))
        c.add(_make_entry("s2"))
        assert len(c) == 2

    def test_get_returns_entry(self):
        c = SessionCatalog()
        e = _make_entry("s1")
        c.add(e)
        assert c.get("s1") is e

    def test_get_missing_returns_none(self):
        c = SessionCatalog()
        assert c.get("nope") is None

    def test_remove_decreases_len(self):
        c = SessionCatalog()
        c.add(_make_entry("s1"))
        removed = c.remove("s1")
        assert removed is True
        assert len(c) == 0

    def test_for_participant_filters(self):
        c = SessionCatalog()
        c.add(_make_entry("s1", participant_id="p1"))
        c.add(_make_entry("s2", participant_id="p2"))
        assert len(c.for_participant("p1")) == 1
        assert c.for_participant("p1")[0].session_id == "s1"

    def test_to_dict_from_dict_roundtrip(self):
        c = SessionCatalog()
        c.add(_make_entry("s1"))
        c.add(_make_entry("s2"))
        c2 = SessionCatalog.from_dict(c.to_dict())
        assert len(c2) == 2


# ── TestDatasetManifest ───────────────────────────────────────────────────────

class TestDatasetManifest:
    def test_create_sets_version(self):
        m = DatasetManifest.create("/tmp/test_ds")
        from src.dataset.manifest import MANIFEST_VERSION
        assert m.manifest_version == MANIFEST_VERSION

    def test_add_participant_stored(self):
        m = DatasetManifest.create("/tmp/test_ds")
        p = Participant.create(participant_id="p1")
        m.add_participant(p)
        assert m.get_participant("p1") is p

    def test_add_session_stored(self):
        m = DatasetManifest.create("/tmp/test_ds")
        e = _make_entry("s1")
        m.add_session(e)
        assert m.get_session("s1") is e

    def test_sessions_for_participant(self):
        m = DatasetManifest.create("/tmp/test_ds")
        m.add_session(_make_entry("s1", participant_id="p1"))
        m.add_session(_make_entry("s2", participant_id="p2"))
        assert len(m.sessions_for("p1")) == 1

    def test_save_load_roundtrip(self, tmp_path):
        m = DatasetManifest.create(str(tmp_path))
        m.add_participant(Participant.create(participant_id="p1"))
        m.add_session(_make_entry("s1"))
        path = str(tmp_path / "manifest.json")
        m.save(path)
        m2 = DatasetManifest.load(path)
        assert m2.get_participant("p1") is not None
        assert m2.get_session("s1") is not None

    def test_try_load_missing_returns_none(self):
        assert DatasetManifest.try_load("/nonexistent/manifest.json") is None

    def test_touch_updates_last_updated(self):
        m = DatasetManifest.create("/tmp/test_ds")
        original = m.last_updated
        import time; time.sleep(0.01)
        m.touch()
        assert m.last_updated >= original

    def test_all_sessions_returns_list(self):
        m = DatasetManifest.create("/tmp/test_ds")
        m.add_session(_make_entry("s1"))
        m.add_session(_make_entry("s2"))
        assert len(m.all_sessions()) == 2


# ── TestSessionIngestor ───────────────────────────────────────────────────────

class TestSessionIngestor:
    def test_ingest_with_session_log_sets_has_session_log(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        entry = SessionIngestor.ingest(str(session_dir), "p1")
        assert entry.has_session_log is True

    def test_ingest_reads_session_id(self, tmp_path):
        session_dir = _make_session_dir(tmp_path, session_id="20260101_120000")
        entry = SessionIngestor.ingest(str(session_dir), "p1")
        assert entry.session_id == "20260101_120000"

    def test_ingest_computes_duration(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        entry = SessionIngestor.ingest(str(session_dir), "p1")
        assert entry.duration_s == pytest.approx(120.0)

    def test_ingest_reads_samples(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        entry = SessionIngestor.ingest(str(session_dir), "p1")
        assert entry.has_samples is True
        assert entry.n_frames == 5

    def test_ingest_engagement_fractions(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        entry = SessionIngestor.ingest(str(session_dir), "p1")
        assert "focused" in entry.engagement_fractions
        assert entry.engagement_fractions["focused"] == pytest.approx(1.0)

    def test_ingest_counts_embeddings(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        emb_path = tmp_path / "embeddings" / "behavioral_windows.csv"
        _make_embeddings_csv(emb_path, n_windows=15)
        entry = SessionIngestor.ingest(
            str(session_dir), "p1", embeddings_path=str(emb_path)
        )
        assert entry.has_embeddings is True
        assert entry.n_windows == 15

    def test_ingest_reads_labels(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        labels_data = {"labels": {"engagement": "HIGH", "fatigue": "LOW"}, "recall_conf": 4}
        labels_path = session_dir / "labels.json"
        labels_path.write_text(json.dumps(labels_data), encoding="utf-8")
        entry = SessionIngestor.ingest(str(session_dir), "p1")
        assert entry.has_labels is True
        assert entry.label_summary["engagement"] == "HIGH"

    def test_ingest_missing_session_log_records_issue(self, tmp_path):
        session_dir = tmp_path / "empty_session"
        session_dir.mkdir()
        entry = SessionIngestor.ingest(str(session_dir), "p1")
        assert entry.has_session_log is False
        assert any("session_log" in i.lower() for i in entry.quality_issues)

    def test_ingest_missing_embeddings_records_issue(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        entry = SessionIngestor.ingest(
            str(session_dir), "p1", embeddings_path="/nonexistent/windows.csv"
        )
        assert entry.has_embeddings is False
        assert len(entry.quality_issues) > 0

    def test_ingest_calibration_path_stored(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        entry = SessionIngestor.ingest(
            str(session_dir), "p1", calibration_path="/some/profile.json"
        )
        assert entry.calibration_path == "/some/profile.json"

    def test_ingest_participant_id_stored(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        entry = SessionIngestor.ingest(str(session_dir), "participant_042")
        assert entry.participant_id == "participant_042"

    def test_ingest_session_dir_stored_as_absolute(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        entry = SessionIngestor.ingest(str(session_dir), "p1")
        assert Path(entry.session_dir).is_absolute()


# ── TestDatasetRegistry ───────────────────────────────────────────────────────

class TestDatasetRegistry:
    def test_register_participant_creates_entry(self, tmp_path):
        reg = DatasetRegistry(str(tmp_path / "ds"))
        p = reg.register_participant(participant_id="p1")
        assert p.participant_id == "p1"
        assert reg.get_participant("p1") is not None

    def test_register_same_participant_twice_returns_existing(self, tmp_path):
        reg = DatasetRegistry(str(tmp_path / "ds"))
        p1 = reg.register_participant(participant_id="p1", notes="first")
        p2 = reg.register_participant(participant_id="p1", notes="second")
        assert p1.participant_id == p2.participant_id
        # Notes should not have been overwritten
        assert reg.get_participant("p1").notes == "first"

    def test_ingest_session_adds_to_catalog(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        reg = DatasetRegistry(str(tmp_path / "ds"))
        reg.register_participant(participant_id="p1")
        entry = reg.ingest_session(str(session_dir), "p1")
        assert reg.get_session(entry.session_id) is not None

    def test_ingest_links_session_to_participant(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        reg = DatasetRegistry(str(tmp_path / "ds"))
        reg.register_participant(participant_id="p1")
        entry = reg.ingest_session(str(session_dir), "p1")
        p = reg.get_participant("p1")
        assert entry.session_id in p.session_ids

    def test_manifest_persisted_to_disk(self, tmp_path):
        ds_root = tmp_path / "ds"
        reg = DatasetRegistry(str(ds_root))
        reg.register_participant(participant_id="p_persist")
        # Reload from disk
        reg2 = DatasetRegistry(str(ds_root))
        assert reg2.get_participant("p_persist") is not None

    def test_sessions_for_participant(self, tmp_path):
        session_dir1 = _make_session_dir(tmp_path, "sess_A")
        session_dir2 = _make_session_dir(tmp_path, "sess_B")
        reg = DatasetRegistry(str(tmp_path / "ds"))
        reg.register_participant(participant_id="p1")
        reg.register_participant(participant_id="p2")
        reg.ingest_session(str(session_dir1), "p1")
        reg.ingest_session(str(session_dir2), "p2")
        assert len(reg.sessions_for("p1")) == 1
        assert len(reg.sessions_for("p2")) == 1

    def test_calibration_inherited_from_participant(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        reg = DatasetRegistry(str(tmp_path / "ds"))
        reg.register_participant(
            participant_id="p1",
            calibration_profile_path="/fake/profile.json"
        )
        entry = reg.ingest_session(str(session_dir), "p1")
        assert entry.calibration_path == "/fake/profile.json"

    def test_all_sessions_returns_all(self, tmp_path):
        s1 = _make_session_dir(tmp_path, "sess_001")
        s2 = _make_session_dir(tmp_path, "sess_002")
        reg = DatasetRegistry(str(tmp_path / "ds"))
        reg.register_participant("p1")
        reg.ingest_session(str(s1), "p1")
        reg.ingest_session(str(s2), "p1")
        assert len(reg.all_sessions()) == 2


# ── TestDatasetStats ──────────────────────────────────────────────────────────

class TestDatasetStats:
    def test_empty_entries_returns_zero_stats(self):
        s = DatasetStatsComputer.compute([], {})
        assert s.n_sessions == 0
        assert s.n_participants == 0

    def test_n_sessions_correct(self):
        entries = [_make_entry(f"s{i}") for i in range(5)]
        s = DatasetStatsComputer.compute(entries, {})
        assert s.n_sessions == 5

    def test_n_participants_counts_distinct(self):
        entries = [
            _make_entry("s1", participant_id="p1"),
            _make_entry("s2", participant_id="p1"),
            _make_entry("s3", participant_id="p2"),
        ]
        s = DatasetStatsComputer.compute(entries, {})
        assert s.n_participants == 2

    def test_total_frames_sum(self):
        entries = [_make_entry(f"s{i}", n_frames=100) for i in range(3)]
        s = DatasetStatsComputer.compute(entries, {})
        assert s.total_frames == 300

    def test_sessions_with_labels_count(self):
        entries = [
            _make_entry("s1", has_labels=True),
            _make_entry("s2", has_labels=False),
            _make_entry("s3", has_labels=True),
        ]
        s = DatasetStatsComputer.compute(entries, {})
        assert s.sessions_with_labels == 2

    def test_label_distribution(self):
        entries = [
            _make_entry("s1", has_labels=True, label_summary={"engagement": "HIGH"}),
            _make_entry("s2", has_labels=True, label_summary={"engagement": "LOW"}),
            _make_entry("s3", has_labels=True, label_summary={"engagement": "HIGH"}),
        ]
        s = DatasetStatsComputer.compute(entries, {})
        assert s.label_distribution["engagement"]["HIGH"] == 2
        assert s.label_distribution["engagement"]["LOW"]  == 1

    def test_engagement_distribution_weighted(self):
        entries = [
            _make_entry("s1", n_frames=100,
                         eng_fracs={"focused": 1.0, "distracted": 0.0}),
            _make_entry("s2", n_frames=100,
                         eng_fracs={"focused": 0.0, "distracted": 1.0}),
        ]
        s = DatasetStatsComputer.compute(entries, {})
        assert s.engagement_distribution.get("focused",    0) == pytest.approx(0.5)
        assert s.engagement_distribution.get("distracted", 0) == pytest.approx(0.5)

    def test_per_participant_stats_present(self):
        entries = [
            _make_entry("s1", participant_id="p1"),
            _make_entry("s2", participant_id="p1"),
        ]
        s = DatasetStatsComputer.compute(entries, {})
        assert "p1" in s.per_participant
        assert s.per_participant["p1"].n_sessions == 2


# ── TestDatasetChecker ────────────────────────────────────────────────────────

class TestDatasetChecker:
    def test_missing_session_log_is_error(self):
        e = _make_entry(has_session_log=False)
        issues = DatasetChecker.check_session(e)
        assert any(i.severity == "error" and "session_log" in i.issue_type for i in issues)

    def test_empty_samples_is_error(self):
        e = _make_entry(has_samples=True, n_frames=0)
        issues = DatasetChecker.check_session(e)
        assert any(i.issue_type == "empty_samples" for i in issues)

    def test_missing_embeddings_is_warning(self):
        e = _make_entry(has_embeddings=False)
        issues = DatasetChecker.check_session(e)
        assert any(i.severity == "warning" and "embeddings" in i.issue_type for i in issues)

    def test_short_session_is_warning(self):
        e = _make_entry(duration_s=10.0)
        issues = DatasetChecker.check_session(e)
        assert any(i.issue_type == "short_session" for i in issues)

    def test_no_calibration_is_info(self):
        e = _make_entry(has_calibration=False)
        issues = DatasetChecker.check_session(e)
        assert any(i.severity == "info" and "calibration" in i.issue_type for i in issues)

    def test_no_labels_is_info(self):
        e = _make_entry(has_labels=False)
        issues = DatasetChecker.check_session(e)
        assert any(i.severity == "info" and "labels" in i.issue_type for i in issues)

    def test_summary_counts_by_severity(self):
        issues = [
            QualityIssue("s1", "missing_session_log", "error",   "..."),
            QualityIssue("s2", "missing_embeddings",  "warning", "..."),
            QualityIssue("s3", "no_calibration",      "info",    "..."),
        ]
        counts = DatasetChecker.summary(issues)
        assert counts["error"]   == 1
        assert counts["warning"] == 1
        assert counts["info"]    == 1

    def test_complete_session_has_no_errors(self):
        e = _make_entry(
            has_session_log=True,
            has_samples=True,
            has_embeddings=True,
            has_labels=True,
            has_calibration=True,
            n_frames=500,
            n_windows=20,
            duration_s=120.0,
        )
        issues = DatasetChecker.check_session(e)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0


# ── TestDatasetExporter ───────────────────────────────────────────────────────

class TestDatasetExporter:
    def _make_manifest_with_entries(self, entries, tmp_path):
        m = DatasetManifest.create(str(tmp_path))
        for e in entries:
            m.add_session(e)
        return m

    def test_export_all_returns_two_paths(self, tmp_path):
        entries = [_make_entry("s1"), _make_entry("s2")]
        m = self._make_manifest_with_entries(entries, tmp_path)
        stats = DatasetStatsComputer.compute(entries, {})
        issues: list = []
        paths = DatasetExporter.export_all(m, stats, issues, str(tmp_path / "out"))
        assert len(paths) == 2
        assert "dataset_summary_json" in paths
        assert "dataset_summary_csv"  in paths

    def test_json_file_created(self, tmp_path):
        entries = [_make_entry("s1")]
        m = self._make_manifest_with_entries(entries, tmp_path)
        stats = DatasetStatsComputer.compute(entries, {})
        DatasetExporter.export_all(m, stats, [], str(tmp_path / "out"))
        assert (tmp_path / "out" / "dataset_summary.json").exists()

    def test_csv_file_created(self, tmp_path):
        entries = [_make_entry("s1")]
        m = self._make_manifest_with_entries(entries, tmp_path)
        stats = DatasetStatsComputer.compute(entries, {})
        DatasetExporter.export_all(m, stats, [], str(tmp_path / "out"))
        assert (tmp_path / "out" / "dataset_summary.csv").exists()

    def test_json_contains_summary_key(self, tmp_path):
        entries = [_make_entry("s1", n_frames=100)]
        m = self._make_manifest_with_entries(entries, tmp_path)
        stats = DatasetStatsComputer.compute(entries, {})
        out_dir = str(tmp_path / "out")
        paths = DatasetExporter.export_all(m, stats, [], out_dir)
        doc = json.loads(Path(paths["dataset_summary_json"]).read_text())
        assert "summary" in doc
        assert doc["summary"]["n_sessions"] == 1

    def test_csv_has_one_row_per_session(self, tmp_path):
        entries = [_make_entry(f"s{i}") for i in range(3)]
        m = self._make_manifest_with_entries(entries, tmp_path)
        stats = DatasetStatsComputer.compute(entries, {})
        paths = DatasetExporter.export_all(m, stats, [], str(tmp_path / "out"))
        with open(paths["dataset_summary_csv"], encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3

    def test_csv_contains_session_ids(self, tmp_path):
        entries = [_make_entry("sess_xyz")]
        m = self._make_manifest_with_entries(entries, tmp_path)
        stats = DatasetStatsComputer.compute(entries, {})
        paths = DatasetExporter.export_all(m, stats, [], str(tmp_path / "out"))
        with open(paths["dataset_summary_csv"], encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["session_id"] == "sess_xyz"

    def test_csv_has_on_screen_and_blink_columns(self, tmp_path):
        entries = [_make_entry("s1", on_screen_frac=0.90, mean_blink_rate=16.5)]
        m = self._make_manifest_with_entries(entries, tmp_path)
        stats = DatasetStatsComputer.compute(entries, {})
        paths = DatasetExporter.export_all(m, stats, [], str(tmp_path / "out"))
        with open(paths["dataset_summary_csv"], encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert "on_screen_fraction" in rows[0]
        assert "mean_blink_rate"    in rows[0]
        assert float(rows[0]["on_screen_fraction"]) == pytest.approx(0.90)

    def test_json_summary_has_missing_data_block(self, tmp_path):
        entries = [_make_entry("s1", has_labels=False, has_calibration=False)]
        m = self._make_manifest_with_entries(entries, tmp_path)
        stats = DatasetStatsComputer.compute(entries, {})
        paths = DatasetExporter.export_all(m, stats, [], str(tmp_path / "out"))
        doc = json.loads(Path(paths["dataset_summary_json"]).read_text())
        assert "missing_data" in doc["summary"]
        assert doc["summary"]["missing_data"]["no_labels"] == 1


# ── TestDirectoryStructure ────────────────────────────────────────────────────

class TestDirectoryStructure:
    """Verify the physical participants/ directory hierarchy is created."""

    def test_register_creates_participant_dir(self, tmp_path):
        reg = DatasetRegistry(str(tmp_path / "ds"))
        reg.register_participant(participant_id="participant_001")
        assert (tmp_path / "ds" / "participants" / "participant_001").is_dir()

    def test_register_creates_profile_json(self, tmp_path):
        reg = DatasetRegistry(str(tmp_path / "ds"))
        reg.register_participant(participant_id="participant_001", notes="test")
        profile_path = tmp_path / "ds" / "participants" / "participant_001" / "profile.json"
        assert profile_path.exists()
        doc = json.loads(profile_path.read_text())
        assert doc["participant_id"] == "participant_001"
        assert doc["notes"] == "test"

    def test_register_creates_calibration_subdir(self, tmp_path):
        reg = DatasetRegistry(str(tmp_path / "ds"))
        reg.register_participant(participant_id="p1")
        assert (tmp_path / "ds" / "participants" / "p1" / "calibration").is_dir()

    def test_register_creates_sessions_subdir(self, tmp_path):
        reg = DatasetRegistry(str(tmp_path / "ds"))
        reg.register_participant(participant_id="p1")
        assert (tmp_path / "ds" / "participants" / "p1" / "sessions").is_dir()

    def test_ingest_creates_session_entry_json(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        reg = DatasetRegistry(str(tmp_path / "ds"))
        reg.register_participant(participant_id="p1")
        entry = reg.ingest_session(str(session_dir), "p1")
        session_entry_path = (
            tmp_path / "ds" / "participants" / "p1"
            / "sessions" / entry.session_id / "session_entry.json"
        )
        assert session_entry_path.exists()

    def test_session_entry_json_contains_session_id(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)
        reg = DatasetRegistry(str(tmp_path / "ds"))
        reg.register_participant(participant_id="p1")
        entry = reg.ingest_session(str(session_dir), "p1")
        session_entry_path = (
            tmp_path / "ds" / "participants" / "p1"
            / "sessions" / entry.session_id / "session_entry.json"
        )
        doc = json.loads(session_entry_path.read_text())
        assert doc["session_id"] == entry.session_id
        assert doc["participant_id"] == "p1"

    def test_dataset_manifest_at_root(self, tmp_path):
        reg = DatasetRegistry(str(tmp_path / "ds"))
        assert (tmp_path / "ds" / "dataset_manifest.json").exists()


# ── TestOnScreenAndBlink ──────────────────────────────────────────────────────

class TestOnScreenAndBlink:
    """Verify on_screen_fraction and mean_blink_rate flow from CSV → stats."""

    def test_ingestor_computes_on_screen_fraction(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)  # all 5 rows have is_on_screen=True
        entry = SessionIngestor.ingest(str(session_dir), "p1")
        assert entry.on_screen_fraction == pytest.approx(1.0)

    def test_ingestor_computes_mean_blink_rate(self, tmp_path):
        session_dir = _make_session_dir(tmp_path)  # all 5 rows have blink_rate=14.0
        entry = SessionIngestor.ingest(str(session_dir), "p1")
        assert entry.mean_blink_rate == pytest.approx(14.0)

    def test_on_screen_fraction_partial(self, tmp_path):
        session_dir = tmp_path / "mixed_sess"
        session_dir.mkdir()
        (session_dir / "session_log.json").write_text(json.dumps({
            "session_id": "mixed", "experiment_name": "x",
            "start_ts": 0.0, "end_ts": 10.0, "config": {},
        }))
        with open(session_dir / "samples.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "frame_index", "timestamp", "engagement_state",
                "smoothed_score", "is_on_screen", "blink_rate",
            ])
            writer.writeheader()
            for i in range(4):
                writer.writerow({
                    "frame_index": i, "timestamp": float(i),
                    "engagement_state": "focused", "smoothed_score": "0.7",
                    "is_on_screen": "True" if i < 2 else "False",
                    "blink_rate": "10.0",
                })
        entry = SessionIngestor.ingest(str(session_dir), "p1")
        assert entry.on_screen_fraction == pytest.approx(0.5)

    def test_stats_mean_on_screen_weighted(self):
        entries = [
            _make_entry("s1", n_frames=100, on_screen_frac=1.0),
            _make_entry("s2", n_frames=100, on_screen_frac=0.0),
        ]
        stats = DatasetStatsComputer.compute(entries, {})
        assert stats.mean_on_screen_fraction == pytest.approx(0.5)

    def test_stats_mean_blink_rate_weighted(self):
        entries = [
            _make_entry("s1", n_frames=100, mean_blink_rate=10.0),
            _make_entry("s2", n_frames=100, mean_blink_rate=20.0),
        ]
        stats = DatasetStatsComputer.compute(entries, {})
        assert stats.mean_blink_rate == pytest.approx(15.0)

    def test_stats_missing_counts(self):
        entries = [
            _make_entry("s1", has_samples=True,  has_embeddings=False,
                        has_labels=False, has_calibration=False),
            _make_entry("s2", has_samples=False, has_embeddings=True,
                        has_labels=True,  has_calibration=True),
        ]
        stats = DatasetStatsComputer.compute(entries, {})
        assert stats.sessions_missing_samples    == 1
        assert stats.sessions_missing_embeddings == 1
        assert stats.sessions_missing_labels     == 1
        assert stats.sessions_missing_calibration == 1
