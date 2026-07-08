"""JSON Schema oracle for the experimental-metadata conditional rules.

These tests exercise the test-only oracle (``tests.validation.json_schema_oracle``),
which validates against the generated JSON Schema — the independent cross-check that
the runtime Pydantic rules match the rules ``gen-json-schema`` derives from the SoT.
They target the conditional cross-field rules, since gen-pydantic does not emit them.
Each rule is exercised on the smallest class that carries it, asserting on the
specific required-property message rather than on a fully-valid object (the
classes have their own required floors that are out of scope here).

Skipped when ``jsonschema`` is not installed — it is a dev/test-only dependency, not
on the runtime path (which enforces the generated Pydantic models).
"""

from __future__ import annotations

import importlib.util

import pytest

from dca_helpers.validation.result import Severity
from tests.validation.json_schema_oracle import (
    ROOT_CLASS,
    experimental_block_keys,
    validate_dca_experimental_metadata,
    validate_experimental_metadata,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jsonschema") is None,
    reason="jsonschema not installed (dev/test-only dependency)",
)


def _messages(obj, target_class):
    return [
        i.message
        for i in validate_experimental_metadata(obj, target_class=target_class)
    ]


def _requires(obj, target_class, field):
    """True if validating `obj` reports `field` as a required property."""
    needle = f"'{field}' is a required property"
    return needle in _messages(obj, target_class)


# --- Perturbation: modality -> reagent block ------------------------------


def test_compound_modality_requires_chemical_block():
    assert _requires({"modality": "compound"}, "Perturbation", "chemical")
    # Supplying the block clears that specific violation.
    with_block = {"modality": "compound", "chemical": {"name": "staurosporine"}}
    assert not _requires(with_block, "Perturbation", "chemical")


def test_genetic_modality_requires_genetic_block():
    for modality in ("crispr", "orf", "shrna", "mirna"):
        assert _requires({"modality": modality}, "Perturbation", "genetic"), modality


def test_physical_modality_requires_neither_reagent_block():
    msgs = _messages({"modality": "physical"}, "Perturbation")
    assert "'chemical' is a required property" not in msgs
    assert "'genetic' is a required property" not in msgs


# --- Biosample: cell line -> cell_line_name -------------------------------


def test_cell_line_tissue_type_requires_cell_line_name():
    assert _requires({"tissue_type": "cell line"}, "Biosample", "cell_line_name")
    assert not _requires(
        {"tissue_type": "cell line", "cell_line_name": "HeLa"},
        "Biosample",
        "cell_line_name",
    )


def test_non_cell_line_does_not_require_cell_line_name():
    assert not _requires({"tissue_type": "tissue"}, "Biosample", "cell_line_name")


# --- WellPerturbation: non-empty well -> perturbation ---------------------


def test_treated_well_requires_perturbation():
    for role in ("treatment", "negcon", "poscon"):
        assert _requires(
            {"control_class": role}, "WellPerturbation", "perturbation"
        ), role


def test_empty_well_is_exempt_from_perturbation():
    assert not _requires({"control_class": "empty"}, "WellPerturbation", "perturbation")


# --- Provenance: agent -> citation ----------------------------------------
# The grounded -> citation rule was dropped on the base PR (df32979) as
# unexpressible; only the source_kind: agent -> citation rule remains.


def test_agent_source_requires_citation():
    assert _requires({"source_kind": "agent"}, "Provenance", "citation")


def test_grounded_alone_does_not_require_citation():
    # grounded is no longer a precondition; a grounded human/original_data value
    # without a citation must validate (source_kind drives the requirement).
    assert not _requires(
        {"source_kind": "human", "grounded": True}, "Provenance", "citation"
    )


# --- Study: JUMP-CP dataset -> cp_version ---------------------------------


def test_jump_cp_dataset_requires_cp_version():
    assert _requires({"dataset_name": "JUMP-CP_2023"}, "Study", "cp_version")


def test_non_jump_dataset_does_not_require_cp_version():
    assert not _requires({"dataset_name": "some-other-study"}, "Study", "cp_version")


# --- Plumbing -------------------------------------------------------------


def test_root_requires_core_blocks():
    # df32979 tightened the container: study / biosample / specimen /
    # study_component / provenance are now required, so an empty root fails.
    issues = validate_experimental_metadata({}, target_class=ROOT_CLASS)
    assert issues
    assert all(i.severity is Severity.ERROR for i in issues)
    msgs = " ".join(i.message for i in issues)
    for block in ("study", "biosample", "specimen", "study_component", "provenance"):
        assert block in msgs


def test_required_floor_enforced_on_study():
    issues = validate_experimental_metadata({}, target_class="Study")
    assert issues
    assert all(i.severity is Severity.ERROR for i in issues)
    msgs = " ".join(i.message for i in issues)
    for required_field in ("source", "description", "experimenter", "license"):
        assert required_field in msgs


def test_biosample_requires_label_fields():
    # df32979 made the human-readable labels required alongside the term ids.
    issues = validate_experimental_metadata(
        {"tissue_type": "tissue"}, target_class="Biosample"
    )
    msgs = " ".join(i.message for i in issues)
    for label in ("organism", "tissue", "disease", "development_stage"):
        assert label in msgs


def test_provenance_map_rejects_bare_string_shorthand():
    # The bare-enum shorthand ({field_path: "agent"}) bypasses agent -> citation;
    # the validator closes that hole by requiring the FieldProvenance object form.
    issues = validate_experimental_metadata(
        {"provenance": {"study.source": "agent"}}, target_class=ROOT_CLASS
    )
    assert any(i.loc == ("provenance", "study.source") for i in issues)


def test_provenance_agent_entry_requires_citation():
    # With the object form enforced, an agent entry without a citation fails.
    bad = {"provenance": {"study.source": {"source_kind": "agent"}}}
    bad_paths = {
        i.loc for i in validate_experimental_metadata(bad, target_class=ROOT_CLASS)
    }
    assert ("provenance", "study.source") in bad_paths


def test_issue_loc_points_at_offending_path():
    issues = validate_experimental_metadata(
        {"modality": "compound"}, target_class="Perturbation"
    )
    # required-property errors are reported at the object root (empty path)
    assert any(i.loc == () for i in issues)


# --- dca-block extraction (store-flow entry point) ------------------------


def test_experimental_block_keys_are_the_top_level_blocks():
    keys = experimental_block_keys()
    assert {
        "study",
        "biosample",
        "specimen",
        "study_component",
        "provenance",
    } <= keys
    # channel-metadata keys are NOT experimental blocks
    assert "channels" not in keys
    assert "normalization_statistics" not in keys


def test_dca_channel_only_block_fails_required_floor():
    # A dca block with only channel metadata is missing the mirrored floor.
    dca = {
        "version": "0.2",
        "url": "https://example.com",
        "channels": [],
        "normalization_statistics": {},
    }
    issues = validate_dca_experimental_metadata(dca)
    assert issues
    msgs = " ".join(i.message for i in issues)
    for block in ("study", "biosample", "specimen", "study_component", "provenance"):
        assert block in msgs
    # locs are prefixed with "dca" so they point back into the block
    assert all(i.loc[0] == "dca" for i in issues)


def test_dca_block_rule_violation_is_reported_under_dca():
    # tissue_type "cell line" without cell_line_name, embedded in the dca block.
    dca = {
        "version": "0.2",
        "channels": [],
        "biosample": {"tissue_type": "cell line"},
    }
    issues = validate_dca_experimental_metadata(dca)
    assert any(
        i.loc[:2] == ("dca", "biosample") and "cell_line_name" in i.message
        for i in issues
    )


def test_dca_channel_keys_do_not_leak_into_experimental_object():
    # version/url/channels must be ignored, not treated as unexpected properties.
    dca = {"version": "0.2", "channels": [{"name": "DAPI"}]}
    issues = validate_dca_experimental_metadata(dca)
    msgs = " ".join(i.message for i in issues)
    assert "channels" not in msgs
    assert "Additional properties" not in msgs
