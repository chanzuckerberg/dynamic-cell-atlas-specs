"""Tests for the attribute writers: the ``dca`` block and normalization statistics."""

from typing import get_args

import pytest

from dca_helpers.metadata.attributes import (
    DCA_SCHEMA_URL,
    SPEC_VERSION,
    merge_dca_attributes,
    write_normalization_statistics,
)
from dca_helpers.validation.versions.v0_2.models import DCAGroupMetadataV0_2
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
    assert attrs["dca"]["version"] == SPEC_VERSION
    assert attrs["dca"]["url"] == DCA_SCHEMA_URL
    assert attrs["dca"]["normalization_statistics"] == {"x": 1}
    assert attrs["dca"]["channels"] == [{"index": 0, "name": "GFP", "description": "d"}]
    assert attrs["ome"] == {}


def test_merge_replaces_the_previous_block_and_keeps_only_stats():
    attrs = {"ome": {}}
    first = min_dca_block(perturbation={"control_class": "negcon"})
    merge_dca_attributes(
        attrs, first, [{"index": 0, "name": "GFP", "description": "d"}]
    )
    write_normalization_statistics(attrs, STATS)
    merge_dca_attributes(attrs, min_dca_block(), None)
    assert "perturbation" not in attrs["dca"]
    assert "channels" not in attrs["dca"]
    assert attrs["dca"]["normalization_statistics"] == STATS
    assert attrs["dca"]["study"] == min_dca_block()["study"]


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
    assert f"/v{SPEC_VERSION}/" in DCA_SCHEMA_URL
    assert DCA_SCHEMA_URL.endswith("schema.json")


def test_the_stamped_version_is_the_one_the_v0_2_models_accept():
    (accepted,) = get_args(DCAGroupMetadataV0_2.model_fields["version"].annotation)
    assert SPEC_VERSION == accepted


def test_write_stats_rejects_a_non_index_channel_key():
    with pytest.raises(ValueError, match="Unexpected key 'GFP'"):
        write_normalization_statistics({"dca": {}}, {"GFP": STATS["0"]})


def test_write_stats_rejects_a_grid_below_one():
    with pytest.raises(ValueError, match="grid"):
        write_normalization_statistics({"dca": {}}, STATS, metadata={"grid": 0})


def test_write_stats_keeps_free_form_metadata_keys():
    attrs = {"dca": {}}
    metadata = {"grid": 16, "source": "fov sample", "dataset_statistics": True}
    write_normalization_statistics(attrs, STATS, metadata=metadata)
    assert attrs["dca"]["normalization_statistics"]["metadata"] == metadata


def test_write_stats_rejects_metadata_passed_as_a_channel_key():
    with pytest.raises(ValueError, match="metadata="):
        write_normalization_statistics({"dca": {}}, {**STATS, "metadata": {"grid": 4}})


def test_write_stats_coerces_numeric_strings_in_channel_entries():
    stats = {
        "0": {"dataset_statistics": {"p1": "1", "p5": 2.0, "p95": 9.0, "p99": 10.0}}
    }
    attrs = {"dca": {}}
    write_normalization_statistics(attrs, stats)
    assert (
        attrs["dca"]["normalization_statistics"]["0"]["dataset_statistics"]["p1"] == 1.0
    )
