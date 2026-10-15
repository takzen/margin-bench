import json

from margin_bench.report import export


def test_demo_export_round_trip(tmp_path):
    records = [r for r in export.demo_records(range(1)) if r["family_id"].startswith(("dsp.", "control."))]
    summary, attempts = export.build(records, demo=True)
    export.write(tmp_path, summary, attempts)

    data = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert data["demo"] is True
    labels = {m["key"]: m for m in data["models"]}
    assert labels["demo/reference"]["overall"]["mean"] == 1.0
    assert labels["demo/negative"]["overall"]["mean"] < 1.0
    assert len(data["attempts"]) == len(records)
    one = json.loads((tmp_path / "attempts" / f"{data['attempts'][0]['id']}.json").read_text(encoding="utf-8"))
    assert one["plot_data"]["panels"]
    assert one["prompt"] == records[0]["prompt"]


def test_export_detects_generator_drift(tmp_path):
    records = export.demo_records(range(1))[:1]
    records[0]["prompt"] = records[0]["prompt"] + " (changed)"
    try:
        export.build(records)
    except export.ExportError as e:
        assert "generator changed" in str(e)
    else:
        raise AssertionError("drift was not detected")
