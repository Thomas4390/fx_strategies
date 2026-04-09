"""Test prepare_fn hook and native VBT pre-computation."""

import numpy as np

from framework.spec import IndicatorSpec, ParamDef, StrategySpec


def test_spec_accepts_prepare_fn():
    def dummy_prepare(raw, data):
        return {"vwap": np.ones(10)}

    spec = StrategySpec(
        name="Test",
        indicator=IndicatorSpec(
            class_name="T",
            short_name="t",
            input_names=("close_minute",),
            param_names=(),
            output_names=("out",),
            kernel_func=lambda close: close,
        ),
        signal_func=lambda c: (False, False, False, False),
        signal_args_map=(("close_arr", "data.close"),),
        params={},
        prepare_fn=dummy_prepare,
    )
    assert spec.prepare_fn is not None


def test_spec_without_prepare_fn():
    spec = StrategySpec(
        name="Test",
        indicator=IndicatorSpec(
            class_name="T",
            short_name="t",
            input_names=("close_minute",),
            param_names=(),
            output_names=("out",),
            kernel_func=lambda close: close,
        ),
        signal_func=lambda c: (False, False, False, False),
        signal_args_map=(("close_arr", "data.close"),),
        params={},
    )
    assert spec.prepare_fn is None


def test_pre_prefix_valid_in_signal_args_map():
    def dummy_prepare(raw, data):
        return {"vwap": np.ones(10)}

    spec = StrategySpec(
        name="Test",
        indicator=IndicatorSpec(
            class_name="T",
            short_name="t",
            input_names=("close_minute",),
            param_names=(),
            output_names=("out",),
            kernel_func=lambda close: close,
        ),
        signal_func=lambda c: (False, False, False, False),
        signal_args_map=(
            ("close_arr", "data.close"),
            ("vwap_arr", "pre.vwap"),
        ),
        params={},
        prepare_fn=dummy_prepare,
    )
    assert ("vwap_arr", "pre.vwap") in spec.signal_args_map
