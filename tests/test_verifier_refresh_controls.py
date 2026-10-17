import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("refresh_fixture", Path("scripts/benchmark_verifier_refresh_controls.py"))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_hidden_shift_is_a_negative_control_for_family_tv():
    shifted = fixture.cohort(77011,"hidden",4,"training",256)
    assert any(row["feature"]=="proxy-exploit" for row in shifted)
    control = fixture.RefreshControl("shift")
    result = control.decide_and_fit(4,shifted,fixture.cohort(77011,"hidden",4,"calibration",64))
    assert result["shift_tv"]==0 and not result["refreshed"]
    assert any(value!=row["truth"] for value,row in zip(control.verify(shifted),shifted,strict=True))


def test_budgeted_periodic_refresh_never_reads_independent_labels():
    training = fixture.cohort(77011,"hidden",4,"training",256)
    calibration = fixture.cohort(77011,"hidden",4,"calibration",64)
    control = fixture.RefreshControl("periodic")
    assert control.decide_and_fit(4,training,calibration)["refreshed"]
    assert control.verify(training)==[row["truth"] for row in training]
    for phase in (8,12,16):
        control.decide_and_fit(phase,training,calibration)
    assert control.version==2 and control.labels==128
    assert not {r["id"] for r in training} & {r["id"] for r in calibration}


def test_retained_raw_truth_and_summary_accounting():
    import gzip
    import json
    from vare.evidence import sha256_file
    root = Path("results/verifier-refresh-controls-v1")
    for directory in sorted(root.glob("*-77*")):
        summary = json.loads((directory/"summary.json").read_text())
        manifest = json.loads((directory/"SHA256.json").read_text())
        assert all(sha256_file(directory/name)==value for name,value in manifest.items())
        identifiers, attacks = set(), 0
        with gzip.open(directory/"cohorts.jsonl.gz","rt") as records:
            for line in records:
                row = json.loads(line)
                assert row["id"] not in identifiers
                identifiers.add(row["id"])
                if row["feature"]=="proxy-exploit":
                    assert row["proxy"] is True and row["truth"] is False
                else:
                    assert row["proxy"]==row["truth"]
                if row["split"]=="independent":
                    attacks += int(row["feature"]=="proxy-exploit")
        assert len(identifiers)==12*(256+64+256+256)
        metrics = summary["metrics"]
        assert metrics["fixed"]["errors"]==attacks
        assert metrics["periodic"]["errors"]==0
        assert metrics["shift"]["errors"]==(attacks if summary["scenario"]=="hidden" else 0)
        curves = json.loads((directory/"curves.json").read_text())
        for name, values in metrics.items():
            assert values["calibration_labels"]<=128
            selected = [row for row in curves if row["strategy"]==name]
            false = sum(row["proxy_promote"] and not row["independent_promote"] for row in selected)
            missed = sum(row["independent_promote"] and not row["proxy_promote"] for row in selected)
            assert values["false_proxy_promotions"]==false
            assert values["missed_proxy_improvements"]==missed
