"""
Combination tests: one validation defect must never hide another.

Pydantic stops a model's mode="after" validator chain at the first raise, and
runs nested-model validation before the parent's after-validators. Each test
here builds a valid store, applies one or more single-defect mutators, and
asserts that every defect is reported with its own severity.

Pairs that still lose a defect because a nested model failed first are marked
as strict expected failures.
"""

import copy
import itertools
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from dca_helpers.validation.result import Severity

from .test_models import (
    valid_label_store,
    valid_store,
    validate_label_store,
    validate_store,
)

pytestmark = pytest.mark.unit

SAMPLE_STATS = {"p1": 1.0, "p5": 5.0, "p95": 95.0, "p99": 99.0}
NESTED_HIDING = "nested-model failures hide store-level rules"


@dataclass(frozen=True)
class Defect:
    name: str
    severity: Severity
    marker: str
    apply: Callable[[dict], None]
    nested: bool = False
    level_rule: bool = False


def shard_below_1gb(store):
    level = store["levels"][0]
    level["shape"][2:] = [1024, 1024, 1024]
    level["shard_shape"] = [1, 1, 128, 128, 128]


def partial_timepoint_statistics(store):
    for level in store["levels"]:
        level["shape"][0] = 4
    for channel in store["dca"]["normalization_statistics"].values():
        channel["timepoint_statistics"] = {"0": dict(SAMPLE_STATS)}


def time_downsampled(store):
    store["levels"][1]["shape"][0] = 2


def dimension_names_mismatch(store):
    store["levels"][0]["dimension_names"] = ["t", "z", "c", "y", "x"]


def channel_missing_name(store):
    store["dca"]["channels"][0].pop("name")


def axis_units_missing(store):
    store["axis_units"] = {}


def no_coarse_level(store):
    for level, size in zip(store["levels"], (16384, 8192, 4096)):
        level["shape"][2:] = [size, size, size]


def chunk_below_1mb(store):
    store["levels"][0]["chunk_shape"] = [1, 1, 64, 64, 64]


def channel_count_mismatch(store):
    store["dca"]["channels"].append({"name": "extra", "index": 1, "description": "x"})


def off_factor_z(store):
    store["levels"][1]["shape"][2] = store["levels"][0]["shape"][2] // 4


DEFECTS = [
    Defect(
        "shard_below_1gb",
        Severity.ERROR,
        "Uncompressed shard size",
        shard_below_1gb,
        nested=True,
        level_rule=True,
    ),
    Defect(
        "partial_timepoint_statistics",
        Severity.ERROR,
        "timepoint_statistics is",
        partial_timepoint_statistics,
    ),
    Defect(
        "time_downsampled",
        Severity.ERROR,
        "downsampling factor for 't'",
        time_downsampled,
    ),
    Defect(
        "dimension_names_mismatch",
        Severity.ERROR,
        "dimension_names",
        dimension_names_mismatch,
        nested=True,
    ),
    Defect(
        "channel_missing_name",
        Severity.ERROR,
        "Field required",
        channel_missing_name,
        nested=True,
    ),
    Defect("axis_units_missing", Severity.ERROR, "has no unit", axis_units_missing),
    Defect(
        "no_coarse_level",
        Severity.ERROR,
        "No scale level has all spatial",
        no_coarse_level,
    ),
    Defect(
        "chunk_below_1mb",
        Severity.WARNING,
        "recommended 1 MB",
        chunk_below_1mb,
        nested=True,
        level_rule=True,
    ),
    Defect(
        "channel_count_mismatch",
        Severity.WARNING,
        "dca.channels has",
        channel_count_mismatch,
    ),
    Defect(
        "off_factor_z",
        Severity.WARNING,
        "downsampling factor for 'z'",
        off_factor_z,
    ),
]

BY_NAME = {d.name: d for d in DEFECTS}


def build_store(defects):
    store = copy.deepcopy(valid_store())
    for defect in DEFECTS:
        if defect in defects:
            defect.apply(store)
    return store


def unreported(defects):
    issues = validate_store(build_store(defects))
    return [
        d.name
        for d in defects
        if not any(d.marker in i.message and i.severity == d.severity for i in issues)
    ]


def hides(a, b):
    """True when the validator is expected to still lose a defect for this pair."""
    if a.nested != b.nested:
        return True
    has_dimension_names = "dimension_names_mismatch" in {a.name, b.name}
    return has_dimension_names and (a.level_rule or b.level_rule)


def pair_params():
    for a, b in itertools.combinations(DEFECTS, 2):
        marks = []
        if hides(a, b):
            marks.append(pytest.mark.xfail(strict=True, reason=NESTED_HIDING))
        yield pytest.param(a, b, marks=marks, id=f"{a.name}+{b.name}")


@pytest.mark.parametrize("defect", DEFECTS, ids=lambda d: d.name)
def test_single_defect_is_reported(defect):
    assert unreported([defect]) == []


@pytest.mark.parametrize("a,b", pair_params())
def test_pair_of_defects_is_reported(a, b):
    assert unreported([a, b]) == []


def test_every_store_level_defect_is_reported_together():
    store_level = [d for d in DEFECTS if not d.nested]
    assert unreported(store_level) == []


def test_every_level_spec_rule_is_reported_together():
    level_rules = [d for d in DEFECTS if d.level_rule]
    assert unreported(level_rules) == []


@pytest.mark.xfail(strict=True, reason=NESTED_HIDING)
def test_all_defects_are_reported_together():
    assert unreported(DEFECTS) == []


def test_label_store_reports_every_rule_together():
    store = copy.deepcopy(valid_label_store())
    for level, size in zip(store["levels"], (16384, 8192, 4096)):
        level["shape"][2:] = [size, size, size]
    store["levels"][1]["shape"][0] = 2
    store["levels"][2]["shape"][2] = store["levels"][1]["shape"][2] // 4
    issues = validate_label_store(store)
    expected = [
        (Severity.ERROR, "No scale level has all spatial"),
        (Severity.ERROR, "downsampling factor for 't'"),
        (Severity.WARNING, "downsampling factor for 'z'"),
    ]
    for severity, marker in expected:
        assert any(i.severity == severity and marker in i.message for i in issues)
