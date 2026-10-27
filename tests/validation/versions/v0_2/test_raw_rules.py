import copy

import pytest

from dca_helpers.validation.versions.v0_1.models import raw_array_rule_messages
from dca_helpers.validation.versions.v0_2.models import (
    DCALabelScaleLevelSpecV0_2,
    StoreRuleInputs,
    raw_store_rule_messages,
)

from .test_error_hiding import BY_NAME, DEFECTS, build_store
from .test_models import (
    valid_label_level,
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
