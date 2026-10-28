import copy
import logging
import re
from types import SimpleNamespace

import pydantic
import pytest

from dca_helpers.validation.versions.v0_1 import models as models_v01
from dca_helpers.validation.versions.v0_1.models import (
    DCAScaleLevelSpec,
    raw_array_rule_messages,
    raw_rule_messages,
)
from dca_helpers.validation.versions.v0_2 import models as models_v02
from dca_helpers.validation.versions.v0_2.models import (
    DCALabelScaleLevelSpecV0_2,
    DCALabelStoreSpecV0_2,
    DCAStoreSpecV0_2,
    StoreRuleInputs,
    raw_store_rule_messages,
    timepoint_statistics_messages,
)

from .test_error_hiding import BY_NAME, DEFECTS, build_store
from .test_models import (
    valid_label_level,
    valid_label_store,
    valid_level,
    valid_store,
    validate_label_level,
    validate_store,
)

pytestmark = pytest.mark.unit

STORE_LEVEL = [d for d in DEFECTS if not d.nested]


def without_prefix(messages):
    return sorted(m.removeprefix("[SHOULD] ") for m in messages)


@pytest.mark.parametrize("defect", STORE_LEVEL, ids=lambda d: d.name)
def test_raw_store_rules_match_the_model_for_one_defect(defect):
    store = build_store([defect])
    expected = without_prefix(i.message for i in validate_store(store))
    assert without_prefix(raw_store_rule_messages(store)) == expected


def test_raw_store_rules_match_the_model_for_every_store_level_defect():
    store = build_store(STORE_LEVEL)
    expected = without_prefix(i.message for i in validate_store(store))
    assert without_prefix(raw_store_rule_messages(store)) == expected


def test_clean_store_has_no_raw_messages():
    assert raw_store_rule_messages(valid_store()) == []


@pytest.mark.parametrize(
    "hiding",
    ["dca_missing", "axes_order", "channel_field", "level_chunk_should"],
)
def test_raw_rules_find_what_the_model_hides(hiding):
    store = build_store([BY_NAME["no_coarse_level"]])
    if hiding == "dca_missing":
        del store["dca"]
    elif hiding == "axes_order":
        store["axes"] = ["t", "c", "y", "x", "z"]
    elif hiding == "channel_field":
        store["dca"]["channels"][0].pop("name")
    else:
        store["levels"][0]["chunk_shape"] = [1, 1, 64, 64, 64]
    messages = raw_store_rule_messages(store)
    assert any("No scale level has all spatial" in m for m in messages)


def with_field(path, value):
    store = copy.deepcopy(valid_store())
    target = store
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return store


MALFORMED_STORES = {
    "none": None,
    "string": "store",
    "list": [],
    "no_levels_key": {"axes": []},
    "empty_levels": with_field(("levels",), []),
    "levels_string": with_field(("levels",), "levels"),
    "levels_of_strings": with_field(("levels",), ["a", "b"]),
    "level_shape_short": with_field(("levels", 0, "shape"), [1, 2]),
    "axis_units_list": with_field(("axis_units",), ["t"]),
    "axis_units_nonstring_values": with_field(("axis_units",), {"t": 5, "z": [1]}),
    "axis_units_nonstring_keys": with_field(("axis_units",), {1: "second"}),
    "axis_scales_strings": with_field(("axis_scales",), {"t": "fast"}),
    "dca_string": with_field(("dca",), "dca"),
    "dca_list": with_field(("dca",), []),
    "channels_none": with_field(("dca", "channels"), None),
    "statistics_string": with_field(("dca", "normalization_statistics"), "stats"),
    "statistics_entry_not_dict": with_field(
        ("dca", "normalization_statistics"), {"0": 5, "1": None}
    ),
    "timepoint_statistics_list": with_field(
        ("dca", "normalization_statistics", "0", "timepoint_statistics"), []
    ),
    "acquisition_string": with_field(("dca", "acquisition"), "acq"),
    "time_sampling_number": with_field(
        ("dca", "acquisition"), {"time_sampling": 5, "timestamps": ["a", "b"]}
    ),
    "variable_with_bad_timestamps": with_field(
        ("dca", "acquisition"),
        {"time_sampling": "variable", "timestamps": [3, "x", None]},
    ),
}


@pytest.mark.parametrize("data", MALFORMED_STORES.values(), ids=MALFORMED_STORES.keys())
def test_malformed_store_never_raises(data):
    assert isinstance(raw_store_rule_messages(data), list)


@pytest.mark.parametrize("data", [None, "store", [], {}, {"levels": []}])
def test_store_without_usable_levels_is_not_judged(data):
    assert StoreRuleInputs.from_raw(data) is None
    assert raw_store_rule_messages(data) == []


def test_malformed_dca_still_runs_the_level_rules():
    store = build_store([BY_NAME["no_coarse_level"]])
    store["dca"] = "dca"
    inputs = StoreRuleInputs.from_raw(store)
    assert inputs.n_channels is None
    assert inputs.channel_statistics is None
    assert any("No scale level" in m for m in raw_store_rule_messages(store))


def test_label_level_raw_rules_use_the_label_dtype_rule():
    level = valid_label_level(dtype="uint16")
    expected = without_prefix(i.message for i in validate_label_level(level))
    raw = raw_array_rule_messages(level, DCALabelScaleLevelSpecV0_2.dtype_rule)
    assert without_prefix(raw) == expected
    assert any("label array" in m for m in raw)


def test_timepoint_rule_does_not_enumerate_every_timepoint():
    levels = [SimpleNamespace(path="0", shape=[10**9, 1, 1, 1, 1], chunk_shape=[1] * 5)]
    statistics = {"0": {"timepoint_statistics": {"0": {}, "7": {}, "x": {}}}}
    inputs = StoreRuleInputs(
        levels=levels, axis_units={}, axis_scales={}, channel_statistics=statistics
    )
    missing, unexpected = timepoint_statistics_messages(inputs)
    assert "missing 999999998 of 1000000000 timepoints (first missing: '1')" in missing
    assert "1 keys that are not timepoint indices" in unexpected
    assert "first unexpected: 'x'" in unexpected


def test_failing_store_raw_rule_keeps_the_original_errors(monkeypatch):
    def failing_rule(data):
        raise RuntimeError("rule bug")

    monkeypatch.setattr(models_v02, "raw_store_rule_messages", failing_rule)
    store = build_store([BY_NAME["no_coarse_level"]])
    del store["dca"]
    with pytest.raises(pydantic.ValidationError) as info:
        DCAStoreSpecV0_2.model_validate(store)
    assert any("'dca' key" in error["msg"] for error in info.value.errors())


def test_failing_level_raw_rule_keeps_the_original_errors(monkeypatch):
    def failing_rule(data, dtype_rule):
        raise RuntimeError("rule bug")

    monkeypatch.setattr(models_v01, "raw_array_rule_messages", failing_rule)
    with pytest.raises(pydantic.ValidationError) as info:
        DCAScaleLevelSpec.model_validate({**valid_level(), "codec_id": "bad"})
    assert any("not permitted" in error["msg"] for error in info.value.errors())


def reference_timepoint_findings(n_timepoints, keys):
    expected = {str(t) for t in range(n_timepoints)}
    findings = []
    missing = sorted(expected - set(keys), key=int)
    if missing:
        findings.append(("missing", len(missing), missing[0]))
    unexpected = sorted((k for k in keys if k not in expected), key=repr)
    if unexpected:
        findings.append(("unexpected", len(unexpected), repr(unexpected[0])))
    return findings


def timepoint_findings(n_timepoints, keys):
    levels = [
        SimpleNamespace(path="0", shape=[n_timepoints, 1, 1, 1, 1], chunk_shape=[1] * 5)
    ]
    statistics = {"0": {"timepoint_statistics": {k: {} for k in keys}}}
    inputs = StoreRuleInputs(
        levels=levels, axis_units={}, axis_scales={}, channel_statistics=statistics
    )
    findings = []
    for message in timepoint_statistics_messages(inputs):
        missing = re.search(r"missing (\d+) of .*\(first missing: '(\d+)'\)", message)
        if missing:
            findings.append(("missing", int(missing.group(1)), missing.group(2)))
            continue
        unexpected = re.search(
            r"has (\d+) keys .*\(first unexpected: (.+?)\)\. Keys", message
        )
        findings.append(("unexpected", int(unexpected.group(1)), unexpected.group(2)))
    return findings


KEY_SETS = {
    "empty": [],
    "in_range": ["0", "1", "2"],
    "gap": ["0", "2"],
    "leading_zero": ["00", "01", "1"],
    "out_of_range": ["3", "10", "99"],
    "signs_and_spaces": ["-1", "+1", " 1", "1 ", "1.0"],
    "non_digit": ["x", ""],
    "unicode_digits": ["٣", "²", "１"],
    "long_digits": ["9" * 30, "9" * 5000],
    "mixed": ["0", "5", "x", "01", "9" * 5000],
}


@pytest.mark.parametrize("n_timepoints", [-1, 0, 1, 3, 10])
@pytest.mark.parametrize("keys", KEY_SETS.values(), ids=KEY_SETS.keys())
def test_timepoint_rule_matches_the_set_based_reference(n_timepoints, keys):
    assert timepoint_findings(n_timepoints, keys) == reference_timepoint_findings(
        n_timepoints, keys
    )


def test_swallowed_raw_rule_is_logged_with_its_traceback(caplog):
    def failing_rule(data):
        raise RuntimeError("rule bug")

    with caplog.at_level(logging.WARNING):
        assert raw_rule_messages(failing_rule, {}) == []
    records = [r for r in caplog.records if "failing_rule" in r.getMessage()]
    assert len(records) == 1
    assert records[0].exc_info[0] is RuntimeError


def test_failing_label_store_raw_rule_keeps_the_original_errors(monkeypatch):
    def failing_rule(data):
        raise RuntimeError("rule bug")

    monkeypatch.setattr(models_v02, "raw_label_store_rule_messages", failing_rule)
    store = copy.deepcopy(valid_label_store())
    store["axes"] = ["t", "c", "y", "x", "z"]
    with pytest.raises(pydantic.ValidationError) as info:
        DCALabelStoreSpecV0_2.model_validate(store)
    assert any("Axes must be" in error["msg"] for error in info.value.errors())
