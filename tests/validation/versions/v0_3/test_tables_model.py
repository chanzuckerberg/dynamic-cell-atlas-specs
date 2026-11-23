"""
StoreTableV0_3 (the runtime model of a table group's attributes) agrees with
docs/v0.3/schema.json ``$defs/StoreTable``, the JSON Schema oracle, on the Tables
page's examples and on every case of test_tables_schema.py.
"""

import pydantic
import pytest

from dca_helpers.validation.versions.v0_3.tables import StoreTableV0_3

from .test_tables_schema import _OBJECTS, INVALID, _page_examples

jsonschema = pytest.importorskip("jsonschema")

pytestmark = pytest.mark.unit

VALID = {
    "object index": _OBJECTS,
    "feature table": {**_OBJECTS, "type": "feature_table"},
    "ngio combined plate table": {
        **_OBJECTS,
        "type": "feature_table",
        "index_type": "str",
        "categorical_columns": [],
        "measurement_columns": [],
        "metadata_columns": [],
    },
    "condition table": {
        "type": "condition_table",
        "table_version": "1",
        "backend": "parquet",
        "index_key": "zarr_path",
        "index_type": "str",
    },
    "generic table": {
        "type": "generic_table",
        "table_version": "1",
        "backend": "parquet",
    },
    "roi table": {"type": "roi_table", "table_version": "1", "backend": "parquet"},
}

EXTRA_INVALID = {
    "explicit null index_key": {**_OBJECTS, "index_key": None},
    "empty instance_key": {**_OBJECTS, "instance_key": ""},
    "condition table indexed by int": {**VALID["condition table"], "index_type": "int"},
    "integer table_version": {**_OBJECTS, "table_version": 1},
    "region path with a sub-path": {**_OBJECTS, "region": {"path": "../labels/a/b"}},
    "not an object": ["type", "generic_table"],
}


def _model_accepts(attrs) -> bool:
    try:
        StoreTableV0_3.model_validate(attrs)
    except pydantic.ValidationError:
        return False
    return True


@pytest.fixture(scope="module")
def oracle(root_dir):
    import json

    schema = json.loads((root_dir / "docs" / "v0.3" / "schema.json").read_text("utf-8"))
    return jsonschema.Draft202012Validator(
        {**schema["$defs"]["StoreTable"], "$defs": schema["$defs"]}
    )


def test_page_examples_agree(root_dir, oracle):
    examples = _page_examples(root_dir)
    assert examples
    for attrs in examples:
        assert oracle.is_valid(attrs)
        assert _model_accepts(attrs), attrs


@pytest.mark.parametrize("name", list(VALID))
def test_valid_cases_agree(oracle, name):
    assert oracle.is_valid(VALID[name])
    assert _model_accepts(VALID[name])


@pytest.mark.parametrize("name", list(INVALID) + list(EXTRA_INVALID))
def test_invalid_cases_agree(oracle, name):
    attrs = INVALID.get(name, EXTRA_INVALID.get(name))
    assert not oracle.is_valid(attrs), name
    assert not _model_accepts(attrs), name
