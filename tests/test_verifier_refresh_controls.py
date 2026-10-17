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
