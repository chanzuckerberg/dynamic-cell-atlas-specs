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
    validate_well_perturbation_v0_2,
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
                {"ontology_term_id": "HsapDv:0000258", "label": "adult stage"}
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


# --- perturbation block (image-level placement of the assignment) ----------
# Nesting PerturbationAssignment under the root is what brings the perturbation
# rules into the ordinary validation pass. Before that they were reachable only
# through validate_well_perturbation_v0_2, which the store walk never calls, so a
# malformed perturbation written into a store's dca block went unreported.


def _treated(definition: dict | None = None) -> dict:
    assignment: dict = {"control_class": "treatment"}
    if definition is not None:
        assignment["definition"] = definition
    return assignment


def _covered(obj: dict, *paths: str) -> dict:
    for path in paths:
        obj["provenance"][path] = {"source_kind": "human"}
    return obj


def test_perturbation_block_is_optional():
    # Required-when-present: a dataset with no perturbations simply omits it.
    obj = _base_object()
    assert "perturbation" not in obj
    assert validate_experimental_metadata_v0_2(obj) == []


def test_valid_perturbation_block_has_no_issues():
    obj = _base_object()
    obj["perturbation"] = _treated(
        {
            "perturbation_id": "Broad:BRD-K12345678",
            "modality": "compound",
            "chemical": {"pubchem_cid": "PubChem:2244"},
        }
    )
    _covered(obj, "perturbation.control_class", "perturbation.definition")
    assert validate_experimental_metadata_v0_2(obj) == []


def test_perturbation_rule_enforced_through_the_root_model():
    # non-empty control_class -> definition, fired by the same pass that checks
    # every other block (no standalone validate_well_perturbation_v0_2 call).
    obj = _base_object()
    obj["perturbation"] = _treated()
    _covered(obj, "perturbation.control_class")
    errors = [
        i
        for i in validate_experimental_metadata_v0_2(obj)
        if i.severity is Severity.ERROR
    ]
    assert any("definition" in e.message for e in errors)


def test_empty_control_class_is_exempt_through_the_root_model():
    obj = _base_object()
    obj["perturbation"] = {"control_class": "empty"}
    _covered(obj, "perturbation.control_class")
    assert validate_experimental_metadata_v0_2(obj) == []


def test_nested_perturbation_modality_rule_is_enforced():
    # compound -> chemical, two levels down at perturbation.definition.
    obj = _base_object()
    obj["perturbation"] = _treated(
        {"perturbation_id": "P1", "modality": "compound"}  # no chemical block
    )
    _covered(obj, "perturbation.control_class", "perturbation.definition")
    errors = [
        i
        for i in validate_experimental_metadata_v0_2(obj)
        if i.severity is Severity.ERROR
    ]
    assert any("chemical" in e.message for e in errors)


def test_perturbation_paths_join_provenance_coverage():
    # perturbation is in _METADATA_BLOCKS, so its populated fields want a record.
    obj = _base_object()
    obj["perturbation"] = {"control_class": "empty"}
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(
        i.severity is Severity.WARNING and "perturbation.control_class" in i.message
        for i in issues
    )


def test_orphan_perturbation_field_path_is_an_error():
    obj = _base_object()
    obj["provenance"]["perturbation.control_class"] = {"source_kind": "human"}
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(
        i.severity is Severity.ERROR and "perturbation.control_class" in i.message
        for i in issues
    )


# --- the standalone well-level entry point, post-inheritance-refactor ------
# Regression guard: _WellPerturbationV0_2 no longer DECLARES the
# non-empty rule or the nested _PerturbationV0_2 field overrides — it inherits both
# from _PerturbationAssignmentV0_2 through the MRO. The well-level rule tests in
# tests/validation/test_experimental_metadata.py go through the JSON Schema oracle
# (jsonschema), so they would stay green even if the Python class silently stopped
# enforcing. These exercise validate_well_perturbation_v0_2 itself, which is the only
# caller-facing path for the well-level form and had no runtime coverage at all.


def _well_row(**overrides) -> dict:
    row = {"plate": "Plate1", "well": "A03", "control_class": "treatment"}
    row.update(overrides)
    return row


def _compound(**overrides) -> dict:
    pert = {
        "perturbation_id": "Broad:BRD-K12345678",
        "modality": "compound",
        "chemical": {"pubchem_cid": "PubChem:2244"},
    }
    pert.update(overrides)
    return pert


def test_well_perturbation_validator_accepts_a_valid_row():
    assert validate_well_perturbation_v0_2(_well_row(definition=_compound())) == []


def test_well_perturbation_validator_still_enforces_the_nonempty_rule():
    # Inherited from _PerturbationAssignmentV0_2, not declared on the subclass.
    issues = validate_well_perturbation_v0_2(_well_row())  # treated, no definition
    assert any("definition" in i.message for i in issues)


def test_well_perturbation_validator_still_exempts_an_empty_well():
    assert validate_well_perturbation_v0_2(_well_row(control_class="empty")) == []


def test_well_perturbation_validator_still_enforces_nested_modality_rules():
    # compound => chemical, on the inherited _PerturbationV0_2 override.
    row = _well_row(definition={"perturbation_id": "P1", "modality": "compound"})
    assert any("chemical" in i.message for i in validate_well_perturbation_v0_2(row))


def test_well_perturbation_validator_checks_the_cotreatment_too():
    # The second inherited nested override — equally silent if the MRO stopped
    # delivering it, and not covered by any other test.
    row = _well_row(
        definition=_compound(),
        cotreatment={"perturbation_id": "P2", "modality": "crispr"},
    )
    assert any("genetic" in i.message for i in validate_well_perturbation_v0_2(row))


def test_well_perturbation_validator_still_requires_the_hcs_key():
    # Pydantic puts the offending field in `loc`, not in the message ("Field
    # required"), so assert on the locs.
    issues = validate_well_perturbation_v0_2({"control_class": "empty"})
    locs = {loc for i in issues for loc in i.loc}
    assert {"plate", "well"} <= locs


# --- Zebrafish ontologies (ZFA anatomy in tissue, ZFS stages in dev-stage) ---


def test_zebrafish_ontologies_validate():
    obj = _base_object()
    obj["biosample"]["organism"] = [
        {"ontology_term_id": "NCBITaxon:7955", "label": "Danio rerio"}
    ]
    obj["biosample"]["tissue"] = [
        {"ontology_term_id": "ZFA:0001056", "label": "myotome"}
    ]
    obj["biosample"]["development_stage"] = [
        {"ontology_term_id": "ZFS:0000048", "label": "larva"}
    ]
    issues = validate_experimental_metadata_v0_2(obj)
    assert [i for i in issues if i.severity is Severity.ERROR] == []


def test_zebrafish_stage_ontology_not_allowed_in_tissue():
    # ZFS is a developmental-stage ontology; it must not validate as a tissue term.
    obj = _base_object()
    obj["biosample"]["tissue"] = [{"ontology_term_id": "ZFS:0000048", "label": "larva"}]
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(i.severity is Severity.ERROR for i in issues)


def test_zebrafish_unspecified_anatomy_is_rejected():
    """ZFA:0001093 ("unspecified") is a fourth spelling of "no value".

    Cross-modality excludes it from the anatomy role; DCA excludes it by pattern so
    `unknown` stays the one sentinel, same treatment as ZFS:0000000 on dev stage.
    """
    obj = _base_object()
    obj["biosample"]["tissue"] = [
        {"ontology_term_id": "ZFA:0001093", "label": "unspecified"}
    ]
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(i.severity is Severity.ERROR for i in issues)


def test_zebrafish_cell_branch_term_is_accepted_for_cell_culture():
    """The ZFA:0009000 ("cell") branch is the cell-type role, which DCA folds into
    `tissue` when tissue_type is "cell culture" (cross-modality routes it to a
    separate field DCA does not have)."""
    obj = _base_object()
    obj["biosample"]["organism"] = [
        {"ontology_term_id": "NCBITaxon:7955", "label": "Danio rerio"}
    ]
    obj["biosample"]["tissue_type"] = "cell culture"
    obj["biosample"]["tissue"] = [
        {"ontology_term_id": "ZFA:0009285", "label": "podocyte"}
    ]
    issues = validate_experimental_metadata_v0_2(obj)
    assert [i for i in issues if i.severity is Severity.ERROR] == []


# --- Per-organism development-stage ontologies (cross-modality v1.1.0) ---


@pytest.mark.parametrize(
    "term_id,label",
    [
        ("HsapDv:0000258", "adult stage"),  # human
        ("MmusDv:0000036", "Theiler stage 27"),  # mouse
        ("ZFS:0000049", "segmentation"),  # zebrafish; no UBERON equivalent
        ("FBdv:00000054", "cleavage stage"),  # fly; descendant of FBdv:00005259
        ("WBls:0000669", "unfertilized egg Ce"),  # worm; named by cross-modality
        ("UBERON:0000068", "embryo stage"),  # fallback life-cycle term
        ("unknown", "unknown"),
        ("na", "na"),
    ],
)
def test_development_stage_accepts_per_organism_ontologies(term_id, label):
    obj = _base_object()
    obj["biosample"]["development_stage"] = [
        {"ontology_term_id": term_id, "label": label}
    ]
    issues = validate_experimental_metadata_v0_2(obj)
    assert [i for i in issues if i.severity is Severity.ERROR] == []


@pytest.mark.parametrize("term_id", ["ZFS:0000000", "FBdv:00007012"])
def test_development_stage_rejects_ontology_unknown_roots(term_id):
    """Cross-modality excludes these by name; the `unknown` sentinel is the one spelling.

    Guards against two ways of saying "unknown" now that ZFS/FBdv are permitted.
    """
    obj = _base_object()
    obj["biosample"]["development_stage"] = [
        {"ontology_term_id": term_id, "label": "unknown"}
    ]
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(i.severity is Severity.ERROR for i in issues)


# --- tissue_type "unknown" (corpora with no per-image manifest) ---


def test_tissue_type_unknown_validates():
    """A pooled training corpus can say the sample type is not recorded."""
    obj = _base_object()
    obj["biosample"]["tissue_type"] = "unknown"
    for slot in ("organism", "tissue", "disease", "development_stage"):
        obj["biosample"][slot] = [{"ontology_term_id": "unknown", "label": "unknown"}]
    issues = validate_experimental_metadata_v0_2(obj)
    assert [i for i in issues if i.severity is Severity.ERROR] == []


def test_tissue_type_rejects_unrecognized_value():
    obj = _base_object()
    obj["biosample"]["tissue_type"] = "not-a-tissue-type"
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(i.severity is Severity.ERROR for i in issues)


# --- cell_line_id is RECOMMENDED, cell_line_name REQUIRED, for a cell line ---


def _cell_line_object() -> dict:
    obj = _base_object()
    obj["biosample"]["tissue_type"] = "cell line"
    obj["biosample"]["tissue"] = [{"ontology_term_id": "CVCL:0042", "label": "U2OS"}]
    obj["biosample"]["cell_line_name"] = "U2OS"
    obj["biosample"]["cell_line_id"] = "CVCL:0042"
    return obj


def test_cell_line_id_is_optional_for_a_cell_line():
    """Pins the documented asymmetry: only `cell_line_name` is required.

    `cell_line_id` may be absent for iPSC / primary lines with no Cellosaurus
    accession, so the block header must not claim it is required.
    """
    obj = _cell_line_object()
    del obj["biosample"]["cell_line_id"]
    issues = validate_experimental_metadata_v0_2(obj)
    assert [i for i in issues if i.severity is Severity.ERROR] == []


def test_cell_line_name_is_required_for_a_cell_line():
    obj = _cell_line_object()
    del obj["biosample"]["cell_line_name"]
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(
        i.severity is Severity.ERROR and "cell_line_name" in i.message for i in issues
    )


def test_cell_line_id_divergence_from_tissue_is_a_should_warning():
    """The mirror disagreeing with its authoritative source is a SHOULD, not a MUST.

    Warning rather than error so the rule can land in a patch release: nothing that
    validated before starts failing.
    """
    obj = _cell_line_object()
    obj["biosample"]["cell_line_id"] = "CVCL:0030"  # HeLa, disagrees with tissue U2OS
    issues = validate_experimental_metadata_v0_2(obj)
    assert [i for i in issues if i.severity is Severity.ERROR] == []
    assert any(
        i.severity is Severity.WARNING and "cell_line_id" in i.message for i in issues
    )


def test_cell_line_tissue_without_a_cvcl_accession_is_a_should_warning():
    """The common real-world error: `tissue` holding anatomy for a cell-line sample.

    Caught independently of `cell_line_id`, since a wrong ontology in the
    authoritative slot is wrong whether or not the mirror is populated.
    """
    obj = _cell_line_object()
    obj["biosample"]["tissue"] = [
        {"ontology_term_id": "UBERON:0002371", "label": "bone marrow"}
    ]
    issues = validate_experimental_metadata_v0_2(obj)
    assert [i for i in issues if i.severity is Severity.ERROR] == []
    assert any(
        i.severity is Severity.WARNING and "no CVCL: accession" in i.message
        for i in issues
    )


def test_cell_line_tissue_check_fires_without_cell_line_id():
    obj = _cell_line_object()
    del obj["biosample"]["cell_line_id"]
    obj["biosample"]["tissue"] = [{"ontology_term_id": "CL:0000000", "label": "cell"}]
    issues = validate_experimental_metadata_v0_2(obj)
    assert any("no CVCL: accession" in i.message for i in issues)


def test_cell_line_tissue_sentinel_is_exempt():
    """Sentinels stand in for a CURIE, so they are the escape hatch, not a wrong term."""
    obj = _cell_line_object()
    obj["biosample"]["tissue"] = [{"ontology_term_id": "unknown", "label": "unknown"}]
    assert not [
        i for i in validate_experimental_metadata_v0_2(obj) if "CVCL" in i.message
    ]


def test_cell_line_id_agreeing_with_tissue_is_silent():
    # Provenance-coverage warnings for the two cell_line_* fields are expected here
    # (the fixture does not add records for them); only the agreement rule is asserted.
    issues = validate_experimental_metadata_v0_2(_cell_line_object())
    assert [i for i in issues if i.severity is Severity.ERROR] == []
    assert not [i for i in issues if "does not match" in i.message]


# --- study_type_other is REQUIRED when study_type includes "Other" --------


def _other_study_type_object() -> dict:
    obj = _base_object()
    obj["study"]["study_type"] = ["Other"]
    obj["study"]["study_type_other"] = "Custom modality"
    obj["provenance"]["study.study_type_other"] = {"source_kind": "human"}
    return obj


def test_other_study_type_with_study_type_other_has_no_error():
    assert validate_experimental_metadata_v0_2(_other_study_type_object()) == []


def test_other_study_type_requires_study_type_other():
    obj = _other_study_type_object()
    del obj["study"]["study_type_other"]
    del obj["provenance"]["study.study_type_other"]
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(
        i.severity is Severity.ERROR and "study_type_other" in i.message for i in issues
    )


def test_non_other_study_type_does_not_require_study_type_other():
    # _base_object() has study_type=["FP"] and no study_type_other.
    issues = validate_experimental_metadata_v0_2(_base_object())
    assert not [
        i
        for i in issues
        if i.severity is Severity.ERROR and "study_type_other" in i.message
    ]


def test_other_among_multiple_study_types_still_requires_study_type_other():
    obj = _other_study_type_object()
    obj["study"]["study_type"] = ["FP", "Other"]
    del obj["study"]["study_type_other"]
    del obj["provenance"]["study.study_type_other"]
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(
        i.severity is Severity.ERROR and "study_type_other" in i.message for i in issues
    )


def test_jump_cp_and_other_study_type_violations_are_both_reported():
    # A single accumulating validator enforces both Study MUST rules, so an object
    # violating both must not have one error masked behind the other.
    obj = _base_object()
    obj["study"]["dataset_name"] = "JUMP-CP_2023"
    obj["study"]["study_type"] = ["Other"]
    issues = validate_experimental_metadata_v0_2(obj)
    error_messages = [i.message for i in issues if i.severity is Severity.ERROR]
    assert any("cp_version" in m for m in error_messages)
    assert any("study_type_other" in m for m in error_messages)


# --- id/label agreement, sentinel half (the part needing no ontology service) ---


def _with_term(term_id: str, label: str) -> dict:
    obj = _base_object()
    obj["biosample"]["organism"] = [{"ontology_term_id": term_id, "label": label}]
    return obj


@pytest.mark.parametrize("sentinel", ["unknown", "na"])
def test_sentinel_id_with_a_real_label_is_a_should_warning(sentinel):
    """The case that matters: `label` is what consumers display.

    An id of `unknown` beside a label of "Homo sapiens" reads as a value the record
    explicitly does not have.
    """
    issues = validate_experimental_metadata_v0_2(_with_term(sentinel, "Homo sapiens"))
    assert [i for i in issues if i.severity is Severity.ERROR] == []
    assert any(
        i.severity is Severity.WARNING and "should repeat the sentinel" in i.message
        for i in issues
    )


@pytest.mark.parametrize("sentinel", ["unknown", "na"])
def test_sentinel_id_with_a_matching_label_is_silent(sentinel):
    assert validate_experimental_metadata_v0_2(_with_term(sentinel, sentinel)) == []


def test_unavailable_with_free_text_is_silent():
    """`unavailable` means the value is known but unmapped, so free text is correct."""
    obj = _with_term("unavailable", "an in-house strain with no NCBITaxon entry")
    assert validate_experimental_metadata_v0_2(obj) == []


@pytest.mark.parametrize("label", ["unknown", "na", "unavailable"])
def test_unavailable_with_a_sentinel_label_is_a_should_warning(label):
    issues = validate_experimental_metadata_v0_2(_with_term("unavailable", label))
    assert any(
        i.severity is Severity.WARNING and "free-text value" in i.message
        for i in issues
    )


def test_real_curie_label_is_left_to_the_ontology_lookup():
    """A wrong label on a real term needs term resolution, which is the ontology lookup's job.

    Pinned so that adding a naive string check here later is a deliberate choice.
    """
    obj = _with_term("NCBITaxon:9606", "not the right label")
    assert validate_experimental_metadata_v0_2(obj) == []


def test_sentinel_label_check_reaches_nested_blocks():
    # The walk is structural, so it must find terms outside `biosample` too.
    obj = _base_object()
    obj["study_component"]["assay"] = [
        {"ontology_term_id": "unknown", "label": "microscopy assay"}
    ]
    issues = validate_experimental_metadata_v0_2(obj)
    assert any("should repeat the sentinel" in i.message for i in issues)
    assert any("study_component" in str(i.loc) for i in issues)


# --- development_stage ontology is keyed to organism (cross-modality v1.1.0) ---


def _organism_stage(taxid: str, stage_id: str) -> dict:
    obj = _base_object()
    obj["biosample"]["organism"] = [{"ontology_term_id": taxid, "label": "organism"}]
    # A sentinel id must carry itself as the label, or the id/label rule fires and
    # masks what this fixture is actually exercising.
    label = stage_id if stage_id in ("unknown", "na") else "stage"
    obj["biosample"]["development_stage"] = [
        {"ontology_term_id": stage_id, "label": label}
    ]
    return obj


def test_mismatched_stage_ontology_for_organism_is_a_should_warning():
    # Human stage ontology on a zebrafish biosample: the flat-union pattern accepts
    # it, the cross-slot rule does not.
    obj = _organism_stage("NCBITaxon:7955", "HsapDv:0000258")
    issues = validate_experimental_metadata_v0_2(obj)
    assert [i for i in issues if i.severity is Severity.ERROR] == []
    assert any(
        i.severity is Severity.WARNING and "development_stage" in i.message
        for i in issues
    )


@pytest.mark.parametrize(
    "taxid,stage_id",
    [
        ("NCBITaxon:9606", "HsapDv:0000258"),  # human
        ("NCBITaxon:10090", "MmusDv:0000036"),  # mouse
        ("NCBITaxon:7955", "ZFS:0000048"),  # zebrafish
        ("NCBITaxon:7227", "FBdv:00000054"),  # fly
        ("NCBITaxon:6239", "WBls:0000669"),  # worm
        ("NCBITaxon:7955", "UBERON:0000068"),  # UBERON fallback always allowed
        ("NCBITaxon:7955", "unknown"),  # sentinels are not terms
    ],
)
def test_matching_stage_ontology_for_organism_is_silent(taxid, stage_id):
    assert validate_experimental_metadata_v0_2(_organism_stage(taxid, stage_id)) == []


def test_stage_ontology_rule_is_silent_for_multiple_organisms():
    """`organism` and `development_stage` are independent lists.

    With two organisms there is no positional correspondence to check against, and
    unioning their ontologies would accept a stage term belonging to whichever
    organism happened to be co-listed. Silent rather than falsely reassuring.
    """
    obj = _base_object()
    obj["biosample"]["organism"] = [
        {"ontology_term_id": "NCBITaxon:9606", "label": "Homo sapiens"},
        {"ontology_term_id": "NCBITaxon:10090", "label": "Mus musculus"},
    ]
    obj["biosample"]["development_stage"] = [
        {"ontology_term_id": "HsapDv:0000258", "label": "adult stage"}
    ]
    assert validate_experimental_metadata_v0_2(obj) == []


def test_stage_ontology_rule_is_silent_for_an_unlisted_organism():
    """Cross-modality names a stage ontology for five organisms only.

    For anything else the rule cannot know what to expect, so it stays quiet rather
    than guessing.
    """
    obj = _organism_stage("NCBITaxon:9598", "HsapDv:0000258")  # Pan troglodytes
    assert validate_experimental_metadata_v0_2(obj) == []


def test_missing_cell_line_name_error_is_not_masked_by_the_should_pass():
    """Ordering guard: the MUST validator is declared first and must still fire.

    Pydantic stops at the first raising `mode="after"` validator, so a SHOULD
    accumulator declared ahead of the MUST rule would swallow the error.
    """
    obj = _cell_line_object()
    del obj["biosample"]["cell_line_name"]
    obj["biosample"]["cell_line_id"] = "CVCL:0030"  # also divergent
    issues = validate_experimental_metadata_v0_2(obj)
    assert any(
        i.severity is Severity.ERROR and "cell_line_name" in i.message for i in issues
    )
