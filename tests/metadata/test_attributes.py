"""Tests for the attribute writers: the ``dca`` block and normalization statistics."""

import pytest

from dca_helpers import CURRENT_SPEC_VERSION
from dca_helpers.metadata.attributes import (
    DCA_SCHEMA_URL,
    merge_dca_attributes,
    write_normalization_statistics,
)
from tests.metadata.conftest import min_dca_block

STATS = {
    "0": {
        "dataset_statistics": {"p1": 1.0, "p5": 2.0, "p95": 9.0, "p99": 10.0},
        "timepoint_statistics": {"0": {"p1": 1.0, "p5": 2.0, "p95": 9.0, "p99": 10.0}},
    }
}


def test_merge_stamps_version_and_url_and_keeps_existing_stats():
    attrs = {"dca": {"normalization_statistics": {"x": 1}}, "ome": {}}
    merge_dca_attributes(
        attrs,
        min_dca_block(),
        [{"index": 0, "name": "GFP", "description": "d", "color": "00FF00"}],
    )
    assert attrs["dca"]["version"] == CURRENT_SPEC_VERSION
    assert attrs["dca"]["url"] == DCA_SCHEMA_URL
    assert attrs["dca"]["normalization_statistics"] == {"x": 1}
    assert attrs["dca"]["channels"] == [{"index": 0, "name": "GFP", "description": "d"}]
    assert attrs["ome"] == {}


def test_merge_with_nothing_to_write_leaves_attrs_untouched():
    attrs = {"ome": {}}
    merge_dca_attributes(attrs, None, None)
    assert attrs == {"ome": {}}


RUN = {"version": "1.2.0", "code_git_sha": "abc123"}


def test_run_attrs_are_written_beside_dca_not_inside_it():
    attrs = {"ome": {}}
    merge_dca_attributes(attrs, min_dca_block(), None, run_attrs={"mytool": RUN})
    assert attrs["mytool"] == RUN
    assert "mytool" not in attrs["dca"]
    assert attrs["dca"]["study"] == min_dca_block()["study"]


def test_run_attrs_replace_a_previous_run_wholesale():
    attrs = {"mytool": {"version": "1.0.0", "code_git_sha": "old"}, "dca": {"x": 1}}
    merge_dca_attributes(attrs, None, None, run_attrs={"mytool": {"version": "1.1.0"}})
    assert attrs["mytool"] == {"version": "1.1.0"}
    assert attrs["dca"] == {"x": 1}


def test_without_run_attrs_an_existing_sibling_is_kept():
    attrs = {"mytool": RUN}
    merge_dca_attributes(attrs, min_dca_block(), None)
    assert attrs["mytool"] == RUN


@pytest.mark.parametrize("key", ["dca", "ome"])
def test_run_attrs_cannot_name_a_reserved_key(key):
    attrs = {"ome": {"keep": 1}}
    with pytest.raises(ValueError, match=key):
        merge_dca_attributes(attrs, min_dca_block(), None, run_attrs={key: RUN})
    assert attrs == {"ome": {"keep": 1}}


def test_write_stats_replaces_the_block_and_adds_metadata():
    attrs = {"dca": {"normalization_statistics": {"old": 1}}}
    write_normalization_statistics(attrs, STATS, metadata={"grid": 16})
    assert attrs["dca"]["normalization_statistics"] == {
        **STATS,
        "metadata": {"grid": 16},
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_write_stats_rejects_non_finite_values(value):
    bad = {
        "0": {"dataset_statistics": {"p1": value, "p5": 2.0, "p95": 9.0, "p99": 10.0}}
    }
    with pytest.raises(ValueError, match="finite"):
        write_normalization_statistics({"dca": {}}, bad)


@pytest.mark.parametrize("value", ["nan", "inf", "-Infinity"])
def test_write_stats_rejects_a_non_finite_string_pydantic_would_coerce(value):
    bad = {
        "0": {"dataset_statistics": {"p1": value, "p5": 2.0, "p95": 9.0, "p99": 10.0}}
    }
    with pytest.raises(ValueError, match="finite"):
        write_normalization_statistics({"dca": {}}, bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_write_stats_rejects_non_finite_metadata(value):
    with pytest.raises(ValueError, match="finite"):
        write_normalization_statistics({"dca": {}}, STATS, metadata={"scale": value})


def test_write_stats_rejects_non_finite_timepoint_values():
    stat = {"p1": 1.0, "p5": 2.0, "p95": 9.0, "p99": float("nan")}
    bad = {
        "0": {
            "dataset_statistics": STATS["0"]["dataset_statistics"],
            "timepoint_statistics": {"0": stat},
        }
    }
    with pytest.raises(ValueError, match="finite"):
        write_normalization_statistics({"dca": {}}, bad)


def test_schema_url_names_the_spec_version_and_the_schema_file():
    assert f"/v{CURRENT_SPEC_VERSION}/" in DCA_SCHEMA_URL
    assert DCA_SCHEMA_URL.endswith("schema.json")


def test_write_stats_rejects_a_non_index_channel_key():
    with pytest.raises(ValueError, match="channel index"):
        write_normalization_statistics({"dca": {}}, {"GFP": STATS["0"]})
