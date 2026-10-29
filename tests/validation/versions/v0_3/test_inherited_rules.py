import copy

import pydantic
import pytest

from dca_helpers.validation.result import Severity
from dca_helpers.validation.validator import _convert_errors
from dca_helpers.validation.versions.v0_3.models import (
    DCALabelStoreSpecV0_3,
    DCAStoreSpecV0_3,
)

from ..v0_2.test_models import valid_label_store, valid_store

pytestmark = pytest.mark.unit


def v03_store():
    store = copy.deepcopy(valid_store())
    store["spec_version"] = "0.3"
    store["dca"]["version"] = "0.3"
    return store


def v03_label_store(n_levels=3):
    store = copy.deepcopy(valid_label_store(n_levels))
    store["spec_version"] = "0.3"
    return store


def issues_of(model, data):
    try:
        model.model_validate(data)
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)
    return []


def test_clean_v03_store_passes():
    assert issues_of(DCAStoreSpecV0_3, v03_store()) == []


def test_v03_level_warning_does_not_hide_a_missing_axis_unit():
    store = v03_store()
    store["levels"][0]["chunk_shape"] = [1, 1, 64, 64, 64]
    store["axis_units"] = {**store["axis_units"], "z": None}
    issues = issues_of(DCAStoreSpecV0_3, store)
    assert any(
        i.severity == Severity.WARNING and "recommended 1 MB" in i.message
        for i in issues
    )
    assert any(
        i.severity == Severity.ERROR and "Spatial axis 'z' has no unit" in i.message
        for i in issues
    )


def test_v03_channel_warning_does_not_hide_a_missing_axis_unit():
    store = v03_store()
    store["dca"]["channels"][0]["channel_type"] = "predicted"
    store["axis_units"] = {**store["axis_units"], "z": None}
    issues = issues_of(DCAStoreSpecV0_3, store)
    assert any(i.severity == Severity.WARNING for i in issues)
    assert any("Spatial axis 'z' has no unit" in i.message for i in issues)


def test_v03_wrong_axis_order_reports_only_the_order_errors():
    store = v03_store()
    names = ["x", "y", "z", "c", "t"]
    store["axes"] = names
    for level in store["levels"]:
        level["shape"] = list(reversed(level["shape"]))
        level["chunk_shape"] = list(reversed(level["chunk_shape"]))
        level["dimension_names"] = names
    issues = issues_of(DCAStoreSpecV0_3, store)
    assert issues
    assert all(
        i.message.startswith(("Axes must be", "dimension_names")) for i in issues
    )


def test_v03_label_warning_does_not_hide_the_coarse_level_error():
    store = v03_label_store(n_levels=1)
    store["levels"][0]["shape"][2:] = [4096, 4096, 4096]
    store["levels"][0]["dtype"] = "uint16"
    issues = issues_of(DCALabelStoreSpecV0_3, store)
    assert any(i.severity == Severity.WARNING for i in issues)
    assert any("No scale level has all spatial" in i.message for i in issues)
