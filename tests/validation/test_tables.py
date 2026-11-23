"""
Functional tests for the v0.3 store-level table checks (validation/tables.py),
against plates and single-image stores written to tmp_path, with tables built by
table_fixtures. Assertions look only at issues under the ``tables`` location, since
the base fixtures carry unrelated provenance warnings.
"""

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pytest

from dca_helpers.validation.result import Severity
from dca_helpers.validation.runner import validate

from .table_fixtures import (
    CONDITION_ATTRS,
    add_condition_table,
    add_table,
    object_attrs,
    object_rows,
    write_listing,
)
from .test_labels import add_labels

FIXTURES = Path(__file__).parent.parent / "fixtures"
PLATE_V02 = FIXTURES / "v0_2" / "valid_plate.ome.zarr"
MULTIWELL_V02 = FIXTURES / "v0_2" / "valid_plate_multiwell.ome.zarr"
IMAGE_V02 = FIXTURES / "v0_2" / "valid_two_level_pyramid.ome.zarr"
FIELD = "A/1/0"
MULTIWELL_FIELDS = ["A/1/0", "A/1/1", "B/1/0", "B/1/1"]


def _to_v03(store: Path) -> None:
    """Set dca.version "0.3" on every image group of a copied v0.2 store."""
    for zj in store.rglob("zarr.json"):
        doc = json.loads(zj.read_text())
        dca = doc.get("attributes", {}).get("dca")
        if isinstance(dca, dict):
            dca["version"] = "0.3"
            zj.write_text(json.dumps(doc))


def _copy(base: Path, tmp_path: Path, name: str, v03: bool = True) -> Path:
    store = tmp_path / name
    shutil.copytree(base, store)
    if v03:
        _to_v03(store)
    return store


@pytest.fixture
def plate(tmp_path):
    """A v0.3 plate (one field, A/1/0) with a valid condition table."""
    store = _copy(PLATE_V02, tmp_path, "plate.ome.zarr")
    add_condition_table(store, [FIELD])
    return store


@pytest.fixture
def labelled_plate(plate):
    """The plate with a nuclei label image and a valid object index of it."""
    add_labels(plate / FIELD, ["nuclei"])
    add_table(
        plate,
        "nuclei_ROI_table",
        object_attrs("nuclei"),
        object_rows([FIELD, FIELD], [1, 2]),
    )
    return plate


def _run(path, spec_version="0.3", **kw):
    return validate(str(path), spec_version=spec_version, show_progress=False, **kw)


def _table_issues(run, severity=None):
    return [
        i
        for r in run
        for i in r.issues
        if i.loc
        and i.loc[0] == "tables"
        and (severity is None or i.severity == severity)
    ]


def _errors(run):
    return [i.message for i in _table_issues(run, Severity.ERROR)]


def _warnings(run):
    return [i.message for i in _table_issues(run, Severity.WARNING)]


def _has(messages, *fragments):
    return any(all(f in m for f in fragments) for m in messages)


@pytest.mark.functional
class TestConditionTable:
    def test_valid_plate_passes(self, plate):
        run = _run(plate)
        assert _table_issues(run) == []

    def test_no_tables_is_error(self, tmp_path):
        store = _copy(PLATE_V02, tmp_path, "plate.ome.zarr")
        run = _run(store)
        assert _has(_errors(run), "no tables/obs condition table")
        assert run.summary.stores_failed == 1

    def test_v02_obs_parquet_is_error(self, tmp_path):
        store = _copy(PLATE_V02, tmp_path, "plate.ome.zarr")
        add_condition_table(store, [FIELD], filename="obs.parquet")
        errors = _errors(_run(store))
        assert _has(errors, "v0.2 obs.parquet instead of table.parquet")
        assert not _has(errors, "no tables/obs condition table")

    def test_obs_of_another_type_is_error(self, plate):
        add_table(
            plate,
            "obs",
            {"type": "generic_table", "table_version": "1", "backend": "parquet"},
            {"zarr_path": [FIELD]},
        )
        assert _has(_errors(_run(plate)), "reserved for the condition table")

    def test_second_condition_table_is_error(self, plate):
        add_table(plate, "conditions", CONDITION_ATTRS, {"zarr_path": [FIELD]})
        assert _has(_errors(_run(plate)), "tables/conditions is a condition_table")

    def test_no_zarr_path_column_is_error(self, plate):
        add_table(plate, "obs", CONDITION_ATTRS, {"well": ["A/1"]})
        assert _has(_errors(_run(plate)), "tables/obs has no 'zarr_path' column")

    def test_duplicate_zarr_path_is_error(self, plate):
        add_condition_table(plate, [FIELD, FIELD])
        assert _has(_errors(_run(plate)), "more than one row", repr(FIELD))

    def test_missing_and_extra_rows_are_errors(self, plate):
        add_condition_table(plate, ["A/1/9"])
        errors = _errors(_run(plate))
        assert _has(errors, "no row for 1 image group", repr(FIELD))
        assert _has(errors, "not image groups of the store", "'A/1/9'")

    def test_every_field_of_every_well_is_needed(self, tmp_path):
        store = _copy(MULTIWELL_V02, tmp_path, "plate.ome.zarr")
        add_condition_table(store, ["A/1/0"])
        assert _has(_errors(_run(store)), "no row for 3 image group")
        add_condition_table(store, MULTIWELL_FIELDS)
        assert _table_issues(_run(store)) == []

    def test_field_subset_still_checks_every_field(self, tmp_path):
        # --field-limit narrows deep validation only; every opened well's
        # declared fields are still known, so the full set is compared.
        store = _copy(MULTIWELL_V02, tmp_path, "plate.ome.zarr")
        add_condition_table(store, ["A/1/0", "B/1/0"])
        assert _has(_errors(_run(store, field_limit=1)), "no row for 2 image group")

    def test_well_subset_checks_presence_only(self, tmp_path):
        store = _copy(MULTIWELL_V02, tmp_path, "plate.ome.zarr")
        add_condition_table(store, ["A/1/0", "A/1/1", "Z/9/0"])
        run = _run(store, wells="A/1")
        assert _table_issues(run) == []
        add_condition_table(store, ["A/1/0"])
        assert _has(_errors(_run(store, wells="A/1")), "'A/1/1'")

    def test_single_image_store_has_one_row(self, tmp_path):
        store = _copy(IMAGE_V02, tmp_path, "image.ome.zarr")
        add_condition_table(store, [""])
        assert _table_issues(_run(store)) == []
        add_condition_table(store, ["", "x"])
        assert _has(_errors(_run(store)), "tables/obs has 2 rows")

    def test_single_image_store_without_obs_is_error(self, tmp_path):
        store = _copy(IMAGE_V02, tmp_path, "image.ome.zarr")
        run = _run(store)
        assert _has(_errors(run), "no tables/obs condition table")
        assert run.summary.stores_failed == 1


@pytest.mark.functional
class TestLayout:
    def test_missing_listing_is_error(self, plate):
        (plate / "tables" / "zarr.json").unlink()
        assert _has(_errors(_run(plate)), "tables/ has no zarr.json")

    def test_listing_mismatch_names_both_sides(self, plate):
        add_table(plate, "extra", CONDITION_ATTRS | {"type": "generic_table"}, {})
        write_listing(plate, ["obs", "ghost"])
        errors = _errors(_run(plate))
        assert _has(errors, "lists table(s) with no group", "'ghost'")
        assert _has(errors, "does not list table group(s)", "'extra'")

    def test_listing_not_a_list_is_error(self, plate):
        write_listing(plate, "obs")
        assert _has(_errors(_run(plate)), "'tables' list of names")

    def test_table_group_without_zarr_json_is_error(self, plate):
        (plate / "tables" / "orphan").mkdir()
        write_listing(plate)
        assert _has(_errors(_run(plate)), "tables/orphan/ has no zarr.json")

    def test_attributes_failing_store_table_are_errors(self, plate):
        add_table(
            plate,
            "feats",
            {**object_attrs("nuclei", "feature_table"), "backend": "anndata_v1"},
            object_rows([FIELD], [1]),
        )
        assert _has(_errors(_run(plate)), "tables/feats", "parquet")

    def test_missing_table_parquet_is_error(self, plate):
        add_table(plate, "nuclei_ROI_table", object_attrs("nuclei"), None)
        assert _has(
            _errors(_run(plate)), "tables/nuclei_ROI_table has no table.parquet"
        )

    def test_unreadable_table_parquet_is_error(self, plate):
        add_table(plate, "nuclei_ROI_table", object_attrs("nuclei"), None)
        (plate / "tables" / "nuclei_ROI_table" / "table.parquet").write_bytes(b"x")
        assert _has(_errors(_run(plate)), "table.parquet is unreadable")

    def test_tables_inside_a_plate_field_is_error(self, plate):
        add_table(plate / FIELD, "obs", CONDITION_ATTRS, {"zarr_path": [FIELD]})
        run = _run(plate)
        assert _has(_errors(run), f"Image group '{FIELD}' has its own tables/")
        assert run.summary.stores_failed == 1

    def test_tables_at_a_single_image_root_is_fine(self, tmp_path):
        # A store whose root is the image group: the store root is the image group.
        store = _copy(IMAGE_V02, tmp_path, "image.ome.zarr")
        add_condition_table(store, [""])
        assert not _has(_errors(_run(store)), "has its own tables/")


@pytest.mark.functional
class TestTablesOfObjects:
    def test_valid_object_index_passes(self, labelled_plate):
        assert _table_issues(_run(labelled_plate)) == []

    def test_label_without_object_index_warns(self, plate):
        add_labels(plate / FIELD, ["nuclei"])
        run = _run(plate)
        assert _has(_warnings(run), "label image 'nuclei' has no object index")
        assert _errors(run) == []

    def test_feature_table_is_not_an_object_index(self, plate):
        add_labels(plate / FIELD, ["nuclei"])
        add_table(
            plate,
            "nuclei_features",
            object_attrs("nuclei", "feature_table"),
            object_rows([FIELD], [1], area=[3.0]),
        )
        assert _has(_warnings(_run(plate)), "'nuclei' has no object index")

    @pytest.mark.parametrize("column", ["zarr_path", "label"])
    def test_missing_key_column_is_error(self, labelled_plate, column):
        rows = object_rows([FIELD], [1])
        del rows[column]
        add_table(labelled_plate, "nuclei_ROI_table", object_attrs("nuclei"), rows)
        assert _has(_errors(_run(labelled_plate)), f"no '{column}' column")

    def test_non_integer_label_is_error(self, labelled_plate):
        rows = object_rows([FIELD], [1.0])
        add_table(labelled_plate, "nuclei_ROI_table", object_attrs("nuclei"), rows)
        assert _has(_errors(_run(labelled_plate)), "must be an integer column")

    @pytest.mark.parametrize("table_type", ["masking_roi_table", "feature_table"])
    @pytest.mark.parametrize("labels", [[0, 1], [-3], [None, 2]])
    def test_nonpositive_or_null_label_is_error(self, plate, table_type, labels):
        rows = object_rows([FIELD] * len(labels), labels)
        rows["label"] = pa.array(labels, pa.int64())
        add_table(plate, "nuclei_tab", object_attrs("nuclei", table_type), rows)
        assert _has(_errors(_run(plate)), "whose 'label' is null or <= 0")

    def test_duplicate_object_key_is_error(self, labelled_plate):
        rows = object_rows([FIELD, FIELD, FIELD], [1, 1, 2])
        add_table(labelled_plate, "nuclei_ROI_table", object_attrs("nuclei"), rows)
        assert _has(_errors(_run(labelled_plate)), "1 row(s) that repeat")

    def test_same_label_at_another_timepoint_is_not_a_duplicate(self, labelled_plate):
        rows = object_rows([FIELD, FIELD], [1, 1])
        rows["t"] = [0, 1]
        add_table(labelled_plate, "nuclei_ROI_table", object_attrs("nuclei"), rows)
        assert _errors(_run(labelled_plate)) == []

    def test_duplicates_in_a_feature_table_are_allowed(self, labelled_plate):
        # One row per value: a feature table may repeat an object (tables.rst).
        rows = object_rows([FIELD, FIELD], [1, 1], guide=["g1", "g2"])
        add_table(
            labelled_plate, "calls", object_attrs("nuclei", "feature_table"), rows
        )
        assert _errors(_run(labelled_plate)) == []

    @pytest.mark.parametrize("column", ["parent_label_name", "parent_label"])
    def test_half_a_parent_reference_is_error(self, labelled_plate, column):
        rows = object_rows(
            [FIELD], [1], **{column: ["cells" if "name" in column else 4]}
        )
        add_table(labelled_plate, "nuclei_ROI_table", object_attrs("nuclei"), rows)
        assert _has(_errors(_run(labelled_plate)), "only one of 'parent_label_name'")

    def test_full_parent_reference_passes(self, labelled_plate):
        rows = object_rows([FIELD], [1], parent_label_name=["cells"], parent_label=[4])
        add_table(labelled_plate, "nuclei_ROI_table", object_attrs("nuclei"), rows)
        assert _errors(_run(labelled_plate)) == []

    def test_plate_table_without_well_columns_warns(self, labelled_plate):
        rows = object_rows([FIELD], [1], plate=False)
        add_table(labelled_plate, "nuclei_ROI_table", object_attrs("nuclei"), rows)
        run = _run(labelled_plate)
        assert _has(_warnings(run), "should have the key columns", "path_in_well")
        assert _errors(run) == []

    def test_single_image_table_needs_no_well_columns(self, tmp_path):
        store = _copy(IMAGE_V02, tmp_path, "image.ome.zarr")
        add_condition_table(store, [""])
        add_labels(store, ["nuclei"])
        rows = object_rows([""], [1], plate=False)
        add_table(store, "nuclei_ROI_table", object_attrs("nuclei"), rows)
        assert _table_issues(_run(store)) == []

    def test_no_t_column_with_one_timepoint_passes(self, labelled_plate):
        rows = object_rows([FIELD], [1])
        del rows["t"]
        add_table(labelled_plate, "nuclei_ROI_table", object_attrs("nuclei"), rows)
        assert _errors(_run(labelled_plate)) == []

    def test_no_t_column_with_several_timepoints_is_error(self, labelled_plate):
        level0 = labelled_plate / FIELD / "labels" / "nuclei" / "0" / "zarr.json"
        doc = json.loads(level0.read_text())
        doc["shape"][0] = 3  # t is the first axis of the fixture
        level0.write_text(json.dumps(doc))
        rows = object_rows([FIELD], [1])
        del rows["t"]
        add_table(labelled_plate, "nuclei_ROI_table", object_attrs("nuclei"), rows)
        assert _has(_errors(_run(labelled_plate)), "no 't' column", "3 timepoints")


@pytest.mark.functional
class TestV02Unaffected:
    """A v0.2 store gets none of the table checks, whatever it has under tables/."""

    def test_v02_plate_without_tables_passes(self, tmp_path):
        store = _copy(PLATE_V02, tmp_path, "plate.ome.zarr", v03=False)
        add_labels(store / FIELD, ["nuclei"])
        run = _run(store, spec_version="0.2")
        assert _table_issues(run) == []
        assert run.summary.stores_failed == 0

    def test_v02_layout_with_obs_parquet_and_field_tables_passes(self, tmp_path):
        store = _copy(PLATE_V02, tmp_path, "plate.ome.zarr", v03=False)
        add_condition_table(store, [FIELD], filename="obs.parquet")
        (store / "tables" / "zarr.json").unlink()
        add_table(store / FIELD, "obs", {"anything": True}, {"x": [1]})
        run = _run(store, spec_version="0.2")
        assert _table_issues(run) == []
        assert run.summary.stores_failed == 0

    def test_v02_single_image_without_tables_passes(self, tmp_path):
        store = _copy(IMAGE_V02, tmp_path, "image.ome.zarr", v03=False)
        run = _run(store, spec_version="0.2")
        assert _table_issues(run) == []
        assert run.summary.stores_failed == 0
