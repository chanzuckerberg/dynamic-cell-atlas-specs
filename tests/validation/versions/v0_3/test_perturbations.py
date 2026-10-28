"""
v0.3 perturbation rules (versions/v0_3/experimental_metadata.py): the CELLxGENE-
aligned modality vocabulary, the required ``perturbation`` block and its
``{"modality": "no perturbations"}`` form. Every case is checked against both the
Pydantic rules and the generated JSON Schema, which must agree.
"""

import json
from pathlib import Path

import pytest

from dca_helpers.validation.validator import _WELL_PERTURBATION_VALIDATORS
from dca_helpers.validation.versions.v0_3.experimental_metadata import (
    GENETIC_MODALITIES,
    validate_experimental_metadata_v0_3,
    validate_perturbation_v0_3,
    validate_well_perturbation_v0_3,
)

jsonschema = pytest.importorskip("jsonschema")

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEMA = json.loads(
    (
        REPO_ROOT / "schema/v0.3/generated/dca_experimental_metadata.schema.json"
    ).read_text()
)


def _schema_accepts(def_name: str, obj: dict) -> bool:
    validator = jsonschema.Draft202012Validator(
        {**SCHEMA["$defs"][def_name], "$defs": SCHEMA["$defs"]}
    )
    return validator.is_valid(obj)


def _treatment(modality: str, **blocks) -> dict:
    return {
        "control_class": "treatment",
        "definition": {"perturbation_id": "p1", "modality": modality, **blocks},
    }


GENETIC = {"genetic": {"gene_symbol": "TP53"}}
CHEMICAL = {"chemical": {"pubchem_cid": "PubChem:2244"}}

VALID = {
    "no perturbations": {"modality": "no perturbations"},
    "empty well in a screen": {"control_class": "empty"},
    "compound": _treatment("compound", **CHEMICAL),
    **{f"genetic: {m}": _treatment(m, **GENETIC) for m in sorted(GENETIC_MODALITIES)},
    **{
        f"no typed block: {m}": _treatment(m)
        for m in (
            "diet",
            "protein",
            "temperature",
            "physical",
            "biological",
            "not reported",
            "other",
        )
    },
    "negative control": {
        "control_class": "negcon",
        "definition": {"perturbation_id": "ntc", "modality": "CRISPR knockout screen"}
        | GENETIC,
    },
}

INVALID = {
    "empty block": {},
    "no perturbations with a control class": {
        "modality": "no perturbations",
        "control_class": "negcon",
    },
    "no perturbations with a definition": {
        "modality": "no perturbations",
        "definition": {"perturbation_id": "p1", "modality": "compound"} | CHEMICAL,
    },
    "no perturbations with dosing": {
        "modality": "no perturbations",
        "duration_h": 24.0,
    },
    "treatment without definition": {"control_class": "treatment"},
    "removed value crispr": _treatment("crispr", **GENETIC),
    "removed value unknown": _treatment("unknown"),
    "no perturbations as a definition modality": _treatment("no perturbations"),
    "CRISPR without the genetic block": _treatment("CRISPR interference screen"),
    "compound without the chemical block": _treatment("compound"),
}


def _model_accepts(block: dict) -> bool:
    return (
        validate_well_perturbation_v0_3({"plate": "P1", "well": "A01", **block}) == []
    )


@pytest.mark.parametrize("name", list(VALID))
def test_valid_blocks_pass_model_and_schema(name):
    block = VALID[name]
    assert _model_accepts(block), name
    assert _schema_accepts("PerturbationAssignment", block), name


@pytest.mark.parametrize("name", list(INVALID))
def test_invalid_blocks_fail_model_and_schema(name):
    block = INVALID[name]
    assert not _model_accepts(block), name
    # Typed-block requirements on the nested definition are LinkML rules the JSON
    # Schema renders too, so every case here is rejected by both.
    assert not _schema_accepts("PerturbationAssignment", block), name


def _experimental(perturbation=None) -> dict:
    fixture = (
        REPO_ROOT
        / "tests/fixtures/v0_2/valid_plate_with_provenance.ome.zarr/A/1/0/zarr.json"
    )
    dca = json.loads(fixture.read_text())["attributes"]["dca"]
    keys = (
        "study",
        "biosample",
        "specimen",
        "study_component",
        "acquisition",
        "provenance",
    )
    obj = {k: dca[k] for k in keys if k in dca}
    if perturbation is not None:
        obj["perturbation"] = perturbation
    return obj


def _errors(issues, loc=("perturbation",)):
    return [
        i
        for i in issues
        if i.severity.value == "error" and tuple(i.loc[: len(loc)]) == loc
    ]


def test_perturbation_is_required_on_every_image_group():
    errors = _errors(validate_experimental_metadata_v0_3(_experimental()))
    assert any("Field required" in i.message for i in errors)


def test_no_perturbations_block_satisfies_the_requirement():
    issues = validate_experimental_metadata_v0_3(
        _experimental({"modality": "no perturbations"})
    )
    assert _errors(issues) == []


def test_no_perturbations_message_names_the_extra_fields():
    issues = validate_experimental_metadata_v0_3(
        _experimental({"modality": "no perturbations", "control_class": "negcon"})
    )
    assert any(
        "stands alone" in i.message and "control_class" in i.message
        for i in _errors(issues)
    )


def test_bare_definition_uses_the_v0_3_vocabulary():
    assert (
        validate_perturbation_v0_3(
            {"perturbation_id": "g", "modality": "CRISPR activation screen"} | GENETIC
        )
        == []
    )
    assert validate_perturbation_v0_3(
        {"perturbation_id": "g", "modality": "crispr"} | GENETIC
    )


def test_plate_walk_checks_well_rows_with_the_version_s_rules():
    assert _WELL_PERTURBATION_VALIDATORS["0.3"] is validate_well_perturbation_v0_3
    assert _WELL_PERTURBATION_VALIDATORS["0.2"] is not validate_well_perturbation_v0_3
