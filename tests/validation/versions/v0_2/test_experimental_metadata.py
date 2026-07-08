"""
Unit tests for DCAExperimentalMetadataV0_2 provenance-coverage validation.

DCAExperimentalMetadataV0_2 flags BOTH layers in one pass:

  * Pydantic-enforced rules — the generated base covers required-ness, types, enums,
    and CURIE patterns; v0.2 subclasses add the if/then rules gen-pydantic omits
    (e.g. source_kind: agent -> citation). These tests spot-check parity with the
    JSON Schema; the exhaustive per-rule coverage lives in
    tests/validation/test_experimental_metadata.py.
  * Provenance coverage (the cross-field rule the schema cannot express):
      - orphan field_path (key with no matching field)  -> ERROR
      - missing coverage (field without a record)        -> [SHOULD] WARNING

``provenance`` is a map keyed by field_path, so each record is a value keyed by its
dotted path (duplicate paths are structurally impossible).
"""

from __future__ import annotations

import copy

import pytest

from dca_helpers.validation.result import Severity
from dca_helpers.validation.versions.v0_2.experimental_metadata import (
    DCAExperimentalMetadataV0_2,
    validate_experimental_metadata_v0_2,
)


def _base_object() -> dict:
    """A schema-valid ExperimentalMetadata with full provenance coverage.

    Mirrors the curated `field_path` convention: one record per populated
    top-level attribute of each block (a multivalued or object-valued attribute
    such as `study.experimenter` still gets a single key).
    """
    blocks = {
        "study": {
            "source": "Test Lab",
            "study_type": ["FP"],
            "description": "A test study.",
            "year_imaged": "2022",
            "dataset_name": "test-dataset",
            "release_date": "2022-07-13",
            "license": "CC-BY-4.0",
            "experimenter": [{"first_name": "Ada", "last_name": "Lovelace"}],
        },
        "biosample": {
            "group_id": "grp__001",
            "organism": [
                {"ontology_term_id": "NCBITaxon:9606", "label": "Homo sapiens"}
            ],
            "tissue_type": "tissue",
            "tissue": [
                {"ontology_term_id": "UBERON:0002108", "label": "small intestine"}
            ],
            "disease": [{"ontology_term_id": "PATO:0000461", "label": "normal"}],
            "development_stage": [
                {"ontology_term_id": "HsapDv:0000087", "label": "adult"}
            ],
        },
        "specimen": {
            "specimen_state": "live",
            "preparation_method": [
                {"ontology_term_id": "FBbi:00100027", "label": "adherent live cells"}
            ],
        },
        "study_component": {
            "imaging_method": [
                {"ontology_term_id": "FBbi:00000369", "label": "light sheet"}
            ],
            "assay": [{"ontology_term_id": "EFO:0002909", "label": "microscopy assay"}],
        },
    }
    field_paths = [
        "study.source",
        "study.study_type",
        "study.description",
        "study.year_imaged",
        "study.dataset_name",
        "study.release_date",
        "study.license",
        "study.experimenter",
        "biosample.group_id",
        "biosample.tissue_type",
        "biosample.organism",
        "biosample.tissue",
        "biosample.disease",
        "biosample.development_stage",
        "specimen.specimen_state",
        "specimen.preparation_method",
        "study_component.imaging_method",
        "study_component.assay",
    ]
    blocks["provenance"] = {p: {"source_kind": "human"} for p in field_paths}
    return blocks


# --- The validator class extends the generated model ----------------------


def test_extends_generated_experimental_metadata():
    from dca_helpers.validation.versions.v0_2.experimental_metadata import (
        _base_experimental_metadata,
    )

    assert issubclass(DCAExperimentalMetadataV0_2, _base_experimental_metadata())


# --- Happy path -----------------------------------------------------------


def test_fully_covered_object_has_no_issues():
    assert validate_experimental_metadata_v0_2(_base_object()) == []


# --- Missing coverage -> [SHOULD] WARNING ---------------------------------


def test_missing_provenance_record_is_a_should_warning():
    obj = _base_object()
    del obj["provenance"]["biosample.tissue"]
    issues = validate_experimental_metadata_v0_2(obj)
    assert len(issues) == 1
    assert issues[0].severity is Severity.WARNING
    assert "biosample.tissue" in issues[0].message


# --- Orphan field_path -> ERROR -------------------------------------------


def test_orphan_field_path_is_an_error():
    obj = _base_object()
    obj["provenance"]["biosample.not_a_real_field"] = {"source_kind": "human"}
    issues = validate_experimental_metadata_v0_2(obj)
    errors = [i for i in issues if i.severity is Severity.ERROR]
    assert any("biosample.not_a_real_field" in e.message for e in errors)


def test_orphan_on_absent_block_is_an_error():
    # image_data is not present in the base object, so a record for it is orphaned.
    obj = _base_object()
    obj["provenance"]["image_data.processing_variant"] = {"source_kind": "human"}
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(
        i.severity is Severity.ERROR and "image_data.processing_variant" in i.message
        for i in issues
    )


# --- additional_metadata expands per item ---------------------------------


def test_additional_metadata_requires_per_item_provenance():
    obj = _base_object()
    obj["biosample"]["additional_metadata"] = [
        {"name": "donor_id", "value": "D-001"},
    ]
    # No matching record yet -> missing coverage for the per-item path.
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(
        i.severity is Severity.WARNING
        and "biosample.additional_metadata.donor_id" in i.message
        for i in issues
    )
    # Supplying the per-item record clears it.
    obj["provenance"]["biosample.additional_metadata.donor_id"] = {
        "source_kind": "human",
    }
    assert validate_experimental_metadata_v0_2(obj) == []


# --- Coverage is independent of base-schema validity ----------------------


def test_schema_violation_still_reported_through_subclass():
    # A genuinely invalid object (missing required block) still fails — the
    # subclass does not weaken the inherited required floor.
    obj = copy.deepcopy(_base_object())
    del obj["study"]
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(i.severity is Severity.ERROR for i in issues)


# --- Parity with the JSON Schema: the if/then rules gen-pydantic omits ------


def test_agent_source_without_citation_is_flagged():
    # The schema's `source_kind: agent -> citation` if/then rule. gen-pydantic
    # records it only as metadata; the subclass must flag it like the JSON Schema.
    obj = _base_object()
    obj["provenance"]["biosample.tissue"] = {"source_kind": "agent"}  # no citation
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(i.severity is Severity.ERROR and "citation" in i.message for i in issues)
    # Same finding as the standalone JSON Schema oracle (parity check). The oracle
    # needs jsonschema (a dev/test-only dep), so skip the parity half when it is absent.
    pytest.importorskip("jsonschema")
    from tests.validation.json_schema_oracle import validate_experimental_metadata

    schema_msgs = {i.message for i in validate_experimental_metadata(obj)}
    assert any("citation" in m for m in schema_msgs)
