"""The report joins archived MT5 evidence without rewriting the frozen campaign."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture
def assets():
    import build_x10_report_assets as builder

    return builder, builder.load_bundle()


def test_measured_mt5_failure_is_counted_without_mutating_campaign(assets):
    builder, bundle = assets
    before = copy.deepcopy(bundle["is_summary"])
    criteria = builder.decision_criteria(bundle)
    assert criteria["mt5_model_1"]["measured"]["expectancy_r"] == pytest.approx(
        -0.10342071659963598
    )
    assert not criteria["mt5_model_1"]["pass"]
    assert builder._n_failing(bundle) == 7
    assert builder.n_interpretable_failures(bundle) == 6
    assert criteria["oos_2026"]["measured"] == "non mesuré"
    assert bundle["is_summary"] == before
    assert "non mesuré" not in builder._decision_measure(
        "mt5_model_1", criteria["mt5_model_1"]
    )


def test_absent_mt5_stays_unmeasured(assets):
    builder, bundle = assets
    bundle["mt5_reference"] = None
    assert builder.decision_criteria(bundle)["mt5_model_1"]["measured"] == "non mesuré"
    assert builder.n_interpretable_failures(bundle) == 5
    assert builder.execution_pending(bundle) == 1


@pytest.mark.parametrize("mt5,python,expected", [(0.1, 0.2, True), (0.1, -0.2, False),
                                                     (-0.1, -0.2, False)])
def test_mt5_threshold_and_same_sign_are_both_required(assets, mt5, python, expected):
    builder, bundle = assets
    summary = bundle["mt5_reference"]["summary"]
    summary.update(mt5_expectancy_r=mt5, py_same_window_expectancy_r=python,
                   same_sign=(mt5 > 0) == (python > 0), criterion_8_pass=expected)
    assert builder.decision_criteria(bundle)["mt5_model_1"]["pass"] is expected
    macros = builder.build_macros(bundle)
    assert macros["XTenCriteriaPassing"] == builder.fmt_int(2 + expected)
    assert builder._n_failing(bundle) == 7 - expected


def test_contradictory_mt5_attestation_is_rejected(assets):
    builder, bundle = assets
    bundle["mt5_reference"]["summary"]["criterion_8_pass"] = True
    with pytest.raises(ValueError, match="MT5"):
        builder.decision_criteria(bundle)


def test_all_execution_macros_resolve_against_archives(assets):
    builder, bundle = assets
    macros = builder.build_macros(bundle)
    assert builder.execution_pending(bundle) == 0
    assert macros["XTenTradesOOS"] == "non lu"
    assert macros["XTenCriteriaFailingInterpretable"] == "$6$"
    assert bundle["is_summary"]["n_trials_logged"] == 34
    assert bundle["is_summary"]["holdout"]["state"] == "LOCKED"


def test_engine_expectancy_weights_trade_counts_not_years_or_matched_summary(assets):
    builder, _ = assets
    bundle = {"source": {"summary": {"qc_expectancy_r": 99}, "by_year": [
        {"qc_trades": 1, "qc_expectancy_r": 1.0},
        {"qc_trades": 9, "qc_expectancy_r": -1.0},
    ]}}
    assert builder.engine_expectancy(bundle, "source", "qc") == pytest.approx(-0.8)


@pytest.mark.parametrize("field,value", [("qc_expectancy_r", float("nan")),
                                        ("qc_trades", float("inf")), ("qc_trades", -1)])
def test_nonfinite_or_negative_engine_inputs_are_rejected(assets, field, value):
    builder, bundle = assets
    bundle["reconciliation_qc_2024"]["by_year"][0][field] = value
    with pytest.raises(ValueError, match="Engine"):
        builder.build_macros(bundle)


def test_check_rejects_a_published_table_after_its_source_disappears(assets, monkeypatch):
    builder, bundle = assets
    bundle["is_summary_v1"] = None
    monkeypatch.setattr(builder, "load_bundle", lambda: bundle)
    assert builder.main(["--check"]) == 1


def test_present_execution_sources_must_produce_all_expected_tables(assets):
    builder, bundle = assets
    tables = builder.build_tables(bundle, examples=None)
    assert {"x10_engines", "x10_qc_by_year", "x10_mt5_by_year", "x10_mt5_by_scenario",
            "x10_rungs", "x10_mt5_attribution", "x10_spread_measured", "x10_v1_v2"} <= tables.keys()


def test_malformed_present_source_cannot_silently_remove_a_table(assets):
    builder, bundle = assets
    del bundle["reconciliation_mt5"]["attribution"]["posts"]
    with pytest.raises(KeyError):
        builder.build_tables(bundle, examples=None)


def test_reconciliation_does_not_invent_equality_or_freeze_a_pass(assets):
    builder, bundle = assets
    table = builder.table_rungs(bundle)
    for line in table.splitlines():
        if " & " in line:
            assert line.endswith("\\\\"), line
    assert "barres des deux côtés" not in table
    assert builder.fmt_usd(2.17574, 3) in table
    assert builder.fmt_usd(0.7302, 3) in table
    row = next(line for line in table.splitlines() if "Python vers QC 2024" in line)
    assert "atteinte" in row
    bundle["reconciliation_qc_2024"]["summary"]["match_rate_py_to_qc"] = 0.5
    row = next(line for line in builder.table_rungs(bundle).splitlines()
               if "Python vers QC 2024" in line)
    assert "manquée" in row


def test_worst_year_is_selected_by_measure_not_by_position(assets):
    builder, bundle = assets
    bundle["walkforward"]["years"] = [
        {"year": 2023, "expectancy_r": -0.4},
        {"year": 2021, "expectancy_r": -0.1},
        {"year": 2022, "expectancy_r": -0.9},
    ]
    assert builder.MACRO_SOURCES["XTenWorstYearExpectancy"].value(bundle) == -0.9
    assert builder.MACRO_SOURCES["XTenWorstYear"].value(bundle) == 2022
