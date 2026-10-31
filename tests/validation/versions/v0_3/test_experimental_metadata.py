"""
Unit tests for versions/v0_3/experimental_metadata.py: the processing_variant
list rules, the required study privacy flags, and the guard that keeps the
v0.2-class reuse honest.
"""

import json
from pathlib import Path

import pytest

from dca_helpers.validation.result import Severity
from dca_helpers.validation.versions.v0_3.experimental_metadata import (
    PROCESSING_STEPS,
    validate_dca_experimental_metadata_v0_3,
)

from ...projection_fixtures import add_privacy_flags

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_FIELD = (
    REPO_ROOT
    / "tests/fixtures/v0_2/valid_plate_with_provenance.ome.zarr/A/1/0/zarr.json"
)
PV_LOC = ("dca", "image_data", "processing_variant")


def _base_dca() -> dict:
    """The v0.2 fixture's ``dca`` block with the v0.3-required privacy flags."""
    dca = json.loads(FIXTURE_FIELD.read_text())["attributes"]["dca"]
    add_privacy_flags(dca)
    return dca


def _dca(steps) -> dict:
    dca = _base_dca()
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
        assert validate_dca_experimental_metadata_v0_3(_base_dca()) == []

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
        assert {
            "raw",
            "maximum intensity projection",
            "mean intensity projection",
            "sum intensity projection",
            "z slice",
        } <= PROCESSING_STEPS


@pytest.mark.unit
class TestPrivacyFlags:
    @pytest.mark.parametrize("flag", ["is_pii", "is_phi"])
    def test_missing_flag_is_error(self, flag):
        dca = _base_dca()
        del dca["study"][flag]
        del dca["provenance"][f"study.{flag}"]
        issues = _at(
            validate_dca_experimental_metadata_v0_3(dca), ("dca", "study", flag)
        )
        assert [i.severity for i in issues] == [Severity.ERROR]

    @pytest.mark.parametrize("flag", ["is_pii", "is_phi"])
    @pytest.mark.parametrize("value", ["unknown", "na", None])
    def test_sentinel_or_null_is_error(self, flag, value):
        dca = _base_dca()
        dca["study"][flag] = value
        issues = _at(
            validate_dca_experimental_metadata_v0_3(dca), ("dca", "study", flag)
        )
        assert [i.severity for i in issues] == [Severity.ERROR]

    @pytest.mark.parametrize(
        "is_pii, is_phi", [(False, False), (True, False), (False, True), (True, True)]
    )
    def test_flags_are_independent(self, is_pii, is_phi):
        dca = _base_dca()
        add_privacy_flags(dca, is_pii=is_pii, is_phi=is_phi)
        assert validate_dca_experimental_metadata_v0_3(dca) == []

    def test_flag_without_provenance_is_warning(self):
        dca = _base_dca()
        del dca["provenance"]["study.is_pii"]
        issues = validate_dca_experimental_metadata_v0_3(dca)
        assert [i.severity for i in issues] == [Severity.WARNING]
        assert "study.is_pii" in issues[0].message

    def test_v0_2_study_rules_still_apply(self):
        dca = _base_dca()
        dca["study"]["study_type"] = ["Other"]
        dca["study"].pop("study_type_other", None)
        issues = _at(validate_dca_experimental_metadata_v0_3(dca), ("dca", "study"))
        assert any("study_type_other" in i.message for i in issues)


@pytest.mark.unit
def test_v0_3_schema_differs_from_v0_2_only_in_overridden_blocks():
    """v0_3/experimental_metadata.py reuses the v0.2 rule classes for every block
    but image_data and study. That is only correct while the two generated schemas
    agree everywhere else; if this fails, override the changed block in the v0.3
    module."""

    def load(version):
        path = (
            REPO_ROOT
            / f"schema/v{version}/generated/dca_experimental_metadata.schema.json"
        )
        return json.loads(path.read_text())

    v2, v3 = load("0.2"), load("0.3")
    changed = {"ImageData", "ProcessingStep", "Study"}
    assert set(v3["$defs"]) - set(v2["$defs"]) == {"ProcessingStep"}
    for name in v2["$defs"]:
        if name not in changed:
            assert v3["$defs"][name] == v2["$defs"][name], name
    for key in set(v2) | set(v3):
        if key not in {"$defs", "description", "metamodel_version"}:
            assert v3.get(key) == v2.get(key), key
