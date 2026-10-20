"""Check the actual CLI artifacts from the minimal-dependency CI walkthrough."""

import csv
import json
from pathlib import Path


root = Path("outputs/ci-demo")
manifest = json.loads((root / "demo_manifest.json").read_text(encoding="utf-8"))
assert manifest["synthetic"] is True
assert manifest["samples"] == 600
assert manifest["trials"] == 2
assert manifest["scripted_key_latencies_ms"] == [350.0, 450.0]

for reports in (root / "reports", Path("outputs/ci-reanalysis")):
    report = reports / manifest["session_id"]
    metrics = json.loads((report / "session_metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_frames"] == 600
    assert metrics["n_trials"] == 2
    assert metrics["config_snapshot"]["provenance"]["synthetic"] is True
    assert metrics["reaction_latencies_ms"] == [350.0, 450.0]
    assert len(list((report / "charts").glob("*.png"))) == 5
    html = (report / "report.html").read_text(encoding="utf-8")
    assert "SYNTHETIC DEMONSTRATION" in html
    assert "data:image/png;base64," in html

features = Path("outputs/ci-embeddings") / manifest["session_id"]
embedding = json.loads((features / "session_embedding.json").read_text(encoding="utf-8"))
assert embedding["n_windows"] == 116
assert embedding["n_features"] == 18
assert len(embedding["session_vector"]) == 18
with (features / "behavioral_windows.csv").open(encoding="utf-8", newline="") as stream:
    assert len(list(csv.DictReader(stream))) == 116
print("Synthetic provenance, 600 frames, scripted latencies, reports, charts, and feature export verified.")
