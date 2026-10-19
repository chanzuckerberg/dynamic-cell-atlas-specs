"""
Unit tests for versions/v0_3/experimental_metadata.py: the processing_variant
list rules, and the guard that keeps the v0.2-class reuse honest.
"""

import json
from pathlib import Path

import pytest

from dca_helpers.validation.result import Severity
from dca_helpers.validation.versions.v0_3.experimental_metadata import (
    PROCESSING_STEPS,
    validate_dca_experimental_metadata_v0_3,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_FIELD = (
    REPO_ROOT
    / "tests/fixtures/v0_2/valid_plate_with_provenance.ome.zarr/A/1/0/zarr.json"
)
PV_LOC = ("dca", "image_data", "processing_variant")


def _dca(steps) -> dict:
    dca = json.loads(FIXTURE_FIELD.read_text())["attributes"]["dca"]
    dca["image_data"] = {"processing_variant": steps}
    dca["provenance"] = {
        **dca["provenance"],
        "image_data.processing_variant": {"source_kind": "original_data"},
    }
    return dca


def _at(issues, loc=PV_LOC):
    return [i for i in issues if tuple(i.loc[: len(loc)]) == loc]


@pytest.mark.unit
class TestProcessingVariantList:
    def test_fixture_without_image_data_is_clean(self):
        dca = json.loads(FIXTURE_FIELD.read_text())["attributes"]["dca"]
        assert validate_dca_experimental_metadata_v0_3(dca) == []

    def test_ordered_list_of_known_steps_passes(self):
        steps = ["deconvolved", "registered", "maximum intensity projection"]
        assert validate_dca_experimental_metadata_v0_3(_dca(steps)) == []

    def test_raw_alone_passes(self):
        assert validate_dca_experimental_metadata_v0_3(_dca(["raw"])) == []

    def test_bare_string_is_error(self):
        issues = _at(validate_dca_experimental_metadata_v0_3(_dca("deconvolved")))
        assert [i.severity for i in issues] == [Severity.ERROR]

    def test_empty_list_is_error(self):
        issues = _at(validate_dca_experimental_metadata_v0_3(_dca([])))
        assert [i.severity for i in issues] == [Severity.ERROR]

    def test_raw_with_other_steps_is_error(self):
        issues = _at(validate_dca_experimental_metadata_v0_3(_dca(["raw", "denoised"])))
        assert [i.severity for i in issues] == [Severity.ERROR]
        assert "'raw'" in issues[0].message

    def test_step_outside_vocabulary_is_warning(self):
        steps = ["deconvolved (legacy pipeline)"]
        issues = _at(validate_dca_experimental_metadata_v0_3(_dca(steps)))
        assert [i.severity for i in issues] == [Severity.WARNING]
        assert "deconvolved (legacy pipeline)" in issues[0].message

    def test_processing_description_alongside_steps_passes(self):
        dca = _dca(["deconvolved", "registered", "maximum intensity projection"])
        dca["image_data"][
            "processing_description"
        ] = "deconvolved, then registered on the label channel (current pipeline)"
        dca["provenance"]["image_data.processing_description"] = {
            "source_kind": "original_data"
        }
        assert validate_dca_experimental_metadata_v0_3(dca) == []

    def test_vocabulary_contains_mip_and_raw(self):
        assert {"raw", "maximum intensity projection"} <= PROCESSING_STEPS


@pytest.mark.unit
def test_v0_3_schema_differs_from_v0_2_only_in_image_data():
    """v0_3/experimental_metadata.py reuses the v0.2 rule classes for every block
    but image_data. That is only correct while the two generated schemas agree
    everywhere else; if this fails, override the changed block in the v0.3 module."""

    def load(version):
        path = (
            REPO_ROOT
            / f"schema/v{version}/generated/dca_experimental_metadata.schema.json"
        )
        return json.loads(path.read_text())

    v2, v3 = load("0.2"), load("0.3")
    changed = {"ImageData", "ProcessingStep"}
    assert set(v3["$defs"]) - set(v2["$defs"]) == {"ProcessingStep"}
    for name in v2["$defs"]:
        if name not in changed:
            assert v3["$defs"][name] == v2["$defs"][name], name
    for key in set(v2) | set(v3):
        if key not in {"$defs", "description", "metamodel_version"}:
            assert v3.get(key) == v2.get(key), key
