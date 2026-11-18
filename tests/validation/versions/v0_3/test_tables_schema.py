"""
docs/v0.3/schema.json ``$defs/StoreTable`` (a table group's attributes, tables.rst):
every JSON example on the Tables page validates against it, and the rules it encodes
reject what the page forbids.
"""

import json
import re

import pytest

jsonschema = pytest.importorskip("jsonschema")

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def table_validator(root_dir):
    schema = json.loads((root_dir / "docs" / "v0.3" / "schema.json").read_text("utf-8"))
    return jsonschema.Draft202012Validator(
        {**schema["$defs"]["StoreTable"], "$defs": schema["$defs"]}
    )


def _page_examples(root_dir) -> list[dict]:
    """The table-group attributes in the page's full ``zarr.json`` examples."""
    text = (root_dir / "docs" / "v0.3" / "tables.rst").read_text("utf-8")
    blocks = re.findall(r"\.\. code-block:: json\n\n((?:   .*\n|\n)+)", text)
    examples = []
    for block in blocks:
        doc = json.loads("\n".join(line[3:] for line in block.splitlines()))
        if "attributes" in doc:
            examples.append(doc["attributes"])
    return examples


def test_page_has_examples(root_dir):
    assert len(_page_examples(root_dir)) >= 2


def test_every_page_example_validates(root_dir, table_validator):
    for attrs in _page_examples(root_dir):
        errors = [e.message for e in table_validator.iter_errors(attrs)]
        assert errors == [], (attrs["type"], errors)


_OBJECTS = {
    "type": "masking_roi_table",
    "table_version": "1",
    "backend": "parquet",
    "index_key": "label",
    "index_type": "int",
    "instance_key": "label",
    "region": {"path": "../labels/nuclei"},
}

INVALID = {
    "unknown type": {**_OBJECTS, "type": "cell_index"},
    "condition table without its zarr_path index": {
        "type": "condition_table",
        "table_version": "1",
        "backend": "parquet",
    },
    "condition table indexed by something else": {
        "type": "condition_table",
        "table_version": "1",
        "backend": "parquet",
        "index_key": "group_id",
        "index_type": "str",
    },
    "not parquet": {**_OBJECTS, "backend": "anndata_v1"},
    "wrong table version": {**_OBJECTS, "table_version": "2"},
    "unknown index type": {**_OBJECTS, "index_type": "uuid"},
    "renamed instance key": {**_OBJECTS, "instance_key": "cell_id"},
    "renamed index key": {**_OBJECTS, "index_key": "object_id"},
    "object table without region": {k: v for k, v in _OBJECTS.items() if k != "region"},
    "feature table without region": {
        k: v for k, v in {**_OBJECTS, "type": "feature_table"}.items() if k != "region"
    },
    "region outside labels/": {**_OBJECTS, "region": {"path": "../nuclei"}},
    "region with an absolute URL": {
        **_OBJECTS,
        "region": {"path": "s3://bucket/plate.ome.zarr/A/1/0/labels/nuclei"},
    },
    "region without a path": {**_OBJECTS, "region": {}},
    "a dca attribute block": {**_OBJECTS, "dca": {"region_key": "zarr_path"}},
}


@pytest.mark.parametrize("name", list(INVALID))
def test_schema_rejects(table_validator, name):
    assert not table_validator.is_valid(INVALID[name]), name


def test_ngio_combined_plate_table_validates(table_validator):
    """What ngio 1.1.5 writes for a plate table combined from per-image tables."""
    attrs = {
        **_OBJECTS,
        "type": "feature_table",
        "index_type": "str",
        "categorical_columns": [],
        "measurement_columns": [],
        "metadata_columns": [],
    }
    assert table_validator.is_valid(attrs)


def test_condition_table_validates(table_validator):
    attrs = {
        "type": "condition_table",
        "table_version": "1",
        "backend": "parquet",
        "index_key": "zarr_path",
        "index_type": "str",
    }
    assert table_validator.is_valid(attrs)


def test_generic_table_needs_no_object_keys(table_validator):
    attrs = {"type": "generic_table", "table_version": "1", "backend": "parquet"}
    assert table_validator.is_valid(attrs)
