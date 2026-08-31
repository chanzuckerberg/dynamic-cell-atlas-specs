"""
Full v0.2 ``ExperimentalMetadata`` validation: Pydantic-enforced rules + provenance coverage.

Three things here are not obvious from the code:

  * Why the conditional rules are re-stated as Pydantic validators rather than
    delegated to the generated JSON Schema: ``gen-pydantic`` records the LinkML
    ``if/then`` rules (e.g. ``source_kind: agent ⇒ citation``) only as inert
    ``linkml_meta`` and does not enforce them — historically this class loaded the
    JSON Schema to get that enforcement. Instead, each rule reachable from the root
    object is now enforced directly on a subclass of the generated model: the
    nested ``study`` / ``biosample`` / ``provenance`` definitions are overridden with
    subclasses that carry a ``model_validator`` for their rule. The generated base
    already enforces required-ness, types, enums, and CURIE patterns, so only the
    conditional rules need restating here.

    Scope note: five of the rules are reachable from ``ExperimentalMetadata`` and
    are restated on overridden nested fields — ``Study`` (JUMP-CP ⇒ ``cp_version``,
    and ``study_type`` "Other" ⇒ ``study_type_other``), ``Biosample`` ("cell line" ⇒
    ``cell_line_name``), ``Provenance``/``FieldProvenance`` (``agent`` ⇒
    ``citation``), and — since the ``perturbation`` block was added to the root
    container — ``PerturbationAssignment`` (non-``empty`` ⇒ ``definition``) together
    with the ``Perturbation`` modality rules on its nested ``definition`` /
    ``cotreatment``. That is the point of nesting the assignment: the perturbation
    rules are now enforced by the same ``validate_dca_experimental_metadata_v0_2``
    pass that checks every other block, instead of only by a caller who knows to
    invoke a standalone entry point.

    The ``study_type`` rule is the one exception to "restated from LinkML": unlike
    the other four, it never had a LinkML ``if/then`` form to restate at all —
    ``gen-json-schema`` cannot express an "array contains a value" precondition, so
    this rule has existed only as validator-enforced prose in the schema docstring
    until now.

    ``WellPerturbation`` remains a root object that this container never nests (it is
    the well-level *placement* of the same assignment), so its rules stay reachable
    via the standalone ``validate_well_perturbation_v0_2`` — which the store walk
    still does not call (see validator.rst). ``validate_perturbation_v0_2`` likewise
    stays for validating a bare codebook entry.

  * Why ``provenance`` map values need ``field_path`` injected before validation:
    ``provenance`` is a map keyed by each value's dotted ``field_path`` (LinkML
    ``inlined`` + key), so the key carries the path and the map value does not repeat
    it. But ``gen-pydantic`` renders ``FieldProvenance.field_path`` as required, so the
    same valid map fails Pydantic. The before-validator copies each map key into its
    value's ``field_path`` so the generated model validates the map form.

  * Why provenance coverage lives here at all: it is a cross-element constraint (the
    set of ``provenance`` keys must line up with the populated fields), and LinkML
    rules are per-slot predicates with no set-coverage operator, so it cannot be
    expressed in the schema.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from pydantic import Field, model_validator

from dca_helpers.validation.versions.v0_2._ontology_terms import (
    SENTINELS,
    Term,
    iter_ontology_terms,
)

if TYPE_CHECKING:
    from dca_helpers.validation.result import Issue

# Blocks carrying curated values that need provenance. A block's own nested
# `provenance` attribute (e.g. AcquisitionMetadata.provenance) is the parallel
# layer, not a curated value, so it is excluded during the walk below.
_METADATA_BLOCKS: tuple[str, ...] = (
    "study",
    "biosample",
    "specimen",
    "study_component",
    "acquisition",
    "image_data",
    "perturbation",
)

# Source of the rules below: schema/v0.2/dca_experimental_metadata.yaml. Kept in sync
# with the YAML by hand; the generated JSON Schema is the cross-check (see
# tests/validation/test_experimental_metadata.py for exhaustive per-rule coverage) —
# except the study_type "Other" ⇒ study_type_other rule, which has no JSON Schema
# equivalent to cross-check against and so is exercised only by the Pydantic-level
# test in tests/validation/versions/v0_2/test_experimental_metadata.py.
_JUMP_CP_DATASET = re.compile(r".*[Jj][Uu][Mm][Pp][ _-]?[Cc][Pp].*")

# Genetic perturbation modalities: each requires the ``genetic`` block (the
# ``compound`` modality requires ``chemical`` instead). Mirrors the
# Perturbation ``modality`` if/then rules in the YAML.
_GENETIC_MODALITIES: frozenset[str] = frozenset(
    {"crispr", "orf", "shrna", "mirna", "sirna"}
)

# Cross-modality v1.1.0 keys the development-stage ontology to the organism. It names a
# stage ontology for exactly these five; every other organism falls back to a UBERON
# life-cycle term. The schema pattern is a flat union (a regex cannot reach across
# slots), so the pairing is checked here instead. SHOULD, not MUST: the pattern already
# rejects unknown prefixes, and this only catches a *mismatched* one.
_ORGANISM_STAGE_ONTOLOGY: dict[str, str] = {
    "NCBITaxon:9606": "HsapDv",  # Homo sapiens
    "NCBITaxon:10090": "MmusDv",  # Mus musculus
    "NCBITaxon:7955": "ZFS",  # Danio rerio
    "NCBITaxon:7227": "FBdv",  # Drosophila melanogaster
    "NCBITaxon:6239": "WBls",  # Caenorhabditis elegans
}

# Always acceptable on development_stage regardless of organism: the cross-organism
# fallback ontology. The sentinels (which stand in for a CURIE, not for a term) are
# also always acceptable; they live in _ontology_terms.SENTINELS, shared with the
# lookup guard so the two passes cannot disagree about what a sentinel is.
_STAGE_FALLBACK_PREFIX = "UBERON"


def _generated_module():
    # gen-pydantic output, committed inside the package (Makefile PYDANTIC_OUT). It
    # is a normal importable module, so the validator works on any install layout
    # (editable, wheel, zipapp) without locating a file outside the package tree.
    from dca_helpers.validation.versions.v0_2 import _generated_models

    return _generated_models


def _base_experimental_metadata() -> type:
    return _generated_module().ExperimentalMetadata


_ExperimentalMetadataBase = _base_experimental_metadata()
_StudyBase = _generated_module().Study
_BiosampleBase = _generated_module().Biosample
_FieldProvenanceBase = _generated_module().FieldProvenance
_PerturbationBase = _generated_module().Perturbation
_PerturbationAssignmentBase = _generated_module().PerturbationAssignment
_WellPerturbationBase = _generated_module().WellPerturbation


def experimental_block_keys() -> frozenset[str]:
    """The top-level ``ExperimentalMetadata`` block names (``study``, ``biosample``,
    ``specimen``, ``study_component``, ``acquisition``, ``image_data``,
    ``perturbation``, ``provenance``).

    Derived from the generated model's fields so it tracks the source of truth, and so
    the runtime path does not need to read the JSON Schema file (which is a test-only
    cross-check, not shipped in the package).
    """
    return frozenset(_ExperimentalMetadataBase.model_fields)


def _assert_enum_values(enum_name: str, *values: str) -> None:
    """Trip at import if a rule literal is no longer a member of its generated enum.

    The rules below compare against bare strings (``_enum_value(...)`` yields a plain
    ``.value``), so a member renamed in a schema regen would silently stop matching
    and quietly disable the rule. Asserting the literals against the enum surfaces
    that drift at import instead.
    """
    members = {m.value for m in getattr(_generated_module(), enum_name)}
    missing = set(values) - members
    if missing:
        raise AssertionError(
            f"{enum_name} is missing rule value(s) {sorted(missing)} (has "
            f"{sorted(members)}); regenerate the model or update the rules in "
            "this module to match the schema."
        )


_assert_enum_values("ProvenanceSourceKind", "agent")
_assert_enum_values("TissueType", "cell line")
_assert_enum_values("PerturbationModality", "compound", *_GENETIC_MODALITIES)
_assert_enum_values("PerturbationControlClass", "empty")
_assert_enum_values("StudyType", "Other")


def _enum_value(value: object) -> object:
    """Return an Enum member's ``.value``, or the value itself when not an Enum."""
    return getattr(value, "value", value)


def _is_populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, str)) and len(value) == 0:
        return False
    return True


def _term_ids(value: object) -> list[str]:
    """Pull ``ontology_term_id`` strings out of a multivalued ontology-term slot.

    Tolerates both the constructed model objects and raw dicts, and skips anything
    without a string id, so a malformed entry is left to the pattern check rather
    than crashing the cross-slot rules.
    """
    out: list[str] = []
    for item in value or ():
        tid = (
            item.get("ontology_term_id")
            if isinstance(item, dict)
            else getattr(item, "ontology_term_id", None)
        )
        if isinstance(tid, str) and tid:
            out.append(tid)
    return out


# --- LinkML if/then rules restated as Pydantic validators -------------------
# Each subclass overrides nothing structurally; it only carries the rule its
# generated base records as inert metadata. The rule message is a MUST (no
# "[SHOULD]" prefix), so _convert_errors maps it to Severity.ERROR.


class _StudyV0_2(_StudyBase):  # type: ignore[misc, valid-type]
    """``Study`` with two MUST rules enforced: the ``dataset_name`` JUMP-CP ⇒
    ``cp_version`` rule, and the ``study_type`` "Other" ⇒ ``study_type_other`` rule.

    Accumulated into one raise rather than two separate validators: pydantic runs
    ``mode="after"`` model validators in definition order and stops at the first
    raise, so two independent MUST checks on the same class would leave a Study
    violating both with the second error masked until the first is fixed and the
    object re-validated. Mirrors the accumulate-then-raise-once pattern
    ``_BiosampleV0_2._cross_slot_should_warnings`` uses for its SHOULDs.
    """

    @model_validator(mode="after")
    def _require_conditional_study_fields(self):
        errors: list[str] = []

        name = getattr(self, "dataset_name", None)
        if isinstance(name, str) and _JUMP_CP_DATASET.match(name):
            if not _is_populated(getattr(self, "cp_version", None)):
                errors.append(
                    "cp_version is required when dataset_name identifies a JUMP-CP "
                    f"dataset (got dataset_name='{name}')"
                )

        types = {_enum_value(v) for v in getattr(self, "study_type", None) or ()}
        if "Other" in types:
            if not _is_populated(getattr(self, "study_type_other", None)):
                errors.append(
                    "study_type_other is required when study_type includes 'Other' "
                    "(the OME Experiment.Type vocabulary did not adequately describe "
                    "the study)"
                )

        if errors:
            raise ValueError("\n".join(errors))
        return self


class _BiosampleV0_2(_BiosampleBase):  # type: ignore[misc, valid-type]
    """``Biosample`` with the ``tissue_type`` "cell line" ⇒ ``cell_line_name`` rule."""

    @model_validator(mode="after")
    def _require_cell_line_name(self):
        if _enum_value(getattr(self, "tissue_type", None)) == "cell line":
            if not _is_populated(getattr(self, "cell_line_name", None)):
                raise ValueError(
                    "cell_line_name is required when tissue_type is 'cell line'"
                )
        return self

    # Declared after the MUST rule above: pydantic runs `mode="after"` model
    # validators in definition order and stops at the first raise, so putting the
    # SHOULD accumulator last keeps it from masking the error.
    @model_validator(mode="after")
    def _cross_slot_should_warnings(self):
        """Cross-slot SHOULDs the LinkML schema cannot express (per-slot predicates).

        Accumulated into one raise: ``_convert_errors`` splits a newline-separated
        message into one WARNING per line, so both findings surface in a single pass.

        Caveat worth knowing before reusing this class: these are the first SHOULDs
        raised from model *construction* rather than from ``_coverage_messages``,
        which runs on a plain dict. So ``_BiosampleV0_2(**data)`` now raises for
        metadata that ``validate_experimental_metadata_v0_2`` reports as merely
        warned about. The reporting path is unaffected, and nothing here is exported
        (``validation/__init__`` publishes only ``validate`` / ``validate_zarr_node``
        and the result types), but a consumer constructing the model directly to
        *read* metadata would see an exception where the report says warning.
        """
        warnings: list[str] = []
        warnings += self._stage_ontology_matches_organism()
        warnings += self._cell_line_identity_is_consistent()
        if warnings:
            raise ValueError("\n".join(warnings))
        return self

    def _stage_ontology_matches_organism(self) -> list[str]:
        organisms = [
            tid
            for tid in _term_ids(getattr(self, "organism", None))
            if tid not in SENTINELS
        ]
        # Single organism only. `organism` and `development_stage` are independent
        # lists with no positional correspondence, so with two organisms the check
        # could only union their ontologies -- and a union accepts a mouse stage term
        # on a human/mouse biosample that lists no mouse stage at all, which reads as
        # checked while checking nothing. Unlisted organism: cross-modality names a
        # stage ontology for five, everything else falls back to UBERON, so there is
        # nothing to compare against. Both cases stay quiet rather than guess.
        if len(organisms) != 1 or organisms[0] not in _ORGANISM_STAGE_ONTOLOGY:
            return []
        allowed = {_ORGANISM_STAGE_ONTOLOGY[organisms[0]], _STAGE_FALLBACK_PREFIX}
        out = []
        for tid in _term_ids(getattr(self, "development_stage", None)):
            if tid in SENTINELS:
                continue
            if Term.parse(tid).prefix not in allowed:
                out.append(
                    f"[SHOULD] development_stage '{tid}' does not use the stage "
                    f"ontology for organism {organisms[0]} "
                    f"(expected {' or '.join(sorted(allowed))}, or a sentinel)"
                )
        return out

    def _cell_line_identity_is_consistent(self) -> list[str]:
        """Two SHOULDs about cell-line identity, in order of how often they go wrong.

        The `tissue` slot is authoritative for a cell-line sample and must carry the
        Cellosaurus term; `cell_line_id` only mirrors it. Checking the authority
        first matters: a UBERON or CL term in `tissue` is the common real-world error
        and is wrong whether or not the mirror is populated, whereas two CVCL
        accessions disagreeing is comparatively rare.
        """
        if _enum_value(getattr(self, "tissue_type", None)) != "cell line":
            return []
        term_ids = _term_ids(getattr(self, "tissue", None))
        accessions = [tid for tid in term_ids if Term.parse(tid).prefix == "CVCL"]
        if not accessions:
            # Sentinels stand in for a CURIE rather than naming a term, so a wholly
            # sentinel `tissue` is the documented escape hatch, not a wrong ontology.
            wrong = [tid for tid in term_ids if tid not in SENTINELS]
            if not wrong:
                return []
            return [
                "[SHOULD] tissue_type is 'cell line' but tissue holds no CVCL: "
                f"accession (got {'/'.join(wrong)}); tissue is the authoritative "
                "slot for cell-line identity"
            ]
        cell_line_id = getattr(self, "cell_line_id", None)
        if not _is_populated(cell_line_id) or cell_line_id in accessions:
            return []
        return [
            f"[SHOULD] cell_line_id '{cell_line_id}' does not match the authoritative "
            f"tissue term {'/'.join(accessions)}; tissue wins on conflict"
        ]


class _FieldProvenanceV0_2(_FieldProvenanceBase):  # type: ignore[misc, valid-type]
    """``FieldProvenance`` with the ``source_kind: agent`` ⇒ ``citation`` rule."""

    @model_validator(mode="after")
    def _require_citation_for_agent(self):
        if _enum_value(getattr(self, "source_kind", None)) == "agent":
            if not _is_populated(getattr(self, "citation", None)):
                raise ValueError(
                    "citation is required when source_kind is 'agent' "
                    "(agent-extracted values must carry their citation)"
                )
        return self


class _PerturbationV0_2(_PerturbationBase):  # type: ignore[misc, valid-type]
    """``Perturbation`` with the ``modality`` ⇒ typed-block rules enforced.

    ``modality: compound`` requires the ``chemical`` block; a genetic modality
    (``crispr`` / ``orf`` / ``shrna`` / ``mirna`` / ``sirna``) requires the
    ``genetic`` block. The remaining modalities (``physical`` / ``biological`` /
    ``other`` / ``unknown``) require neither.
    """

    @model_validator(mode="after")
    def _require_modality_block(self):
        modality = _enum_value(getattr(self, "modality", None))
        if modality == "compound" and not _is_populated(
            getattr(self, "chemical", None)
        ):
            raise ValueError(
                "chemical is required when modality is 'compound' "
                "(a compound perturbation must carry the chemical identifier block)"
            )
        if modality in _GENETIC_MODALITIES and not _is_populated(
            getattr(self, "genetic", None)
        ):
            raise ValueError(
                "genetic is required when modality is a genetic kind "
                f"(crispr / orf / shrna / mirna / sirna; got modality='{modality}')"
            )
        return self


class _PerturbationAssignmentV0_2(_PerturbationAssignmentBase):  # type: ignore[misc, valid-type]
    """``PerturbationAssignment`` with the non-``empty`` ⇒ ``definition`` rule.

    Treated or control material (``treatment`` / ``negcon`` / ``poscon``) must name a
    perturbation; ``empty`` has none to record. The nested perturbation fields use
    ``_PerturbationV0_2`` so the Perturbation modality rules are enforced on them too.

    This is the body of both placements: the ``perturbation`` block of
    ``ExperimentalMetadata`` (image level) and, via ``_WellPerturbationV0_2``, the
    well-level row. Declaring the rule once here is what keeps the two placements
    from drifting.
    """

    definition: Optional[_PerturbationV0_2] = Field(default=None)
    cotreatment: Optional[_PerturbationV0_2] = Field(default=None)

    @model_validator(mode="after")
    def _require_definition_for_nonempty(self):
        if _enum_value(getattr(self, "control_class", None)) != "empty":
            if not _is_populated(getattr(self, "definition", None)):
                raise ValueError(
                    "definition is required when control_class is not 'empty' "
                    "(treated and control material must name a perturbation)"
                )
        return self


class _WellPerturbationV0_2(  # type: ignore[misc, valid-type]
    _WellPerturbationBase, _PerturbationAssignmentV0_2
):
    """``WellPerturbation`` — a ``PerturbationAssignment`` plus the ``(plate, well)`` key.

    The non-``empty`` ⇒ ``definition`` rule is inherited from
    ``_PerturbationAssignmentV0_2``, mirroring the LinkML ``is_a`` in the schema (where
    the rule is likewise declared once, on the parent). A ``model_validator`` inherits
    cleanly through the MRO.

    The nested field overrides do NOT, and must be restated. ``model_fields`` is
    resolved in MRO order, and the generated ``WellPerturbation`` — which precedes
    ``_PerturbationAssignmentV0_2`` here — redeclares ``definition`` /
    ``cotreatment`` with the *generated* ``Perturbation`` range. Inheriting
    alone therefore silently reverts them to the un-ruled class and the
    ``modality`` ⇒ typed-block rules stop firing for well-level rows (caught by review
    on #140; ``test_well_perturbation_validator_still_enforces_nested_modality_rules``
    is the guard).

    Reversing the base order is the other way to reach the ruled ranges, and it does
    work: ``PerturbationAssignment`` declares neither ``plate`` nor ``well``, so the
    generated class is their only declaration in the MRO and no ordering can dislodge
    them. Restating the overrides is preferred because it pins the ruled ranges at the
    point of use rather than making them a silent consequence of base order — the same
    property that made the original regression invisible.
    """

    definition: Optional[_PerturbationV0_2] = Field(default=None)
    cotreatment: Optional[_PerturbationV0_2] = Field(default=None)


def _coverage_messages(data: dict) -> tuple[list[str], list[str]]:
    """Provenance-coverage check, returning ``(must_errors, should_warnings)``.

    Coverage is checked at top-level-attribute granularity, matching the curated
    ``field_path`` convention (``study.year_imaged``, ``study.experimenter``), with
    ``additional_metadata`` lists expanded per item to
    ``<block>.additional_metadata.<name>``.

    Missing coverage is a SHOULD, not an error: provenance is required only for
    ``agent`` values and optional for ``human`` / ``original_data``, but the
    ``source_kind`` that decides this lives in the (absent) record, so a missing
    record cannot be tested for agent-ness. The ``agent ⇒ citation`` rule on records
    that *do* exist is enforced by ``_FieldProvenanceV0_2`` above.

    ``provenance`` is a map keyed by ``field_path``, so the provided paths are the
    map keys (duplicates are structurally impossible).
    """
    prov = data.get("provenance")
    # When provenance is malformed the field validation already reports it; walking
    # a non-map here would emit a flood of misleading "missing" warnings.
    if not isinstance(prov, dict):
        return [], []

    expected: set[str] = set()
    for block_name in _METADATA_BLOCKS:
        block = data.get(block_name)
        if not isinstance(block, dict):
            continue
        for attr_name, value in block.items():
            if attr_name == "provenance" or not _is_populated(value):
                continue
            if attr_name == "additional_metadata" and isinstance(value, list):
                for item in value:
                    name = item.get("name") if isinstance(item, dict) else None
                    if name:
                        expected.add(f"{block_name}.additional_metadata.{name}")
                continue
            expected.add(f"{block_name}.{attr_name}")

    provided_set = set(prov.keys())

    must_errors: list[str] = []
    should_warnings: list[str] = []

    for path in sorted(provided_set - expected):
        must_errors.append(
            f"provenance field_path '{path}' does not match any populated "
            "field in the experimental metadata"
        )
    for path in sorted(expected - provided_set):
        should_warnings.append(
            f"[SHOULD] field '{path}' has no provenance record; every curated "
            "value should carry its provenance (required for agent-curated values)"
        )
    return must_errors, should_warnings


class DCAExperimentalMetadataV0_2(_ExperimentalMetadataBase):  # type: ignore[misc, valid-type]
    """v0.2 ``ExperimentalMetadata`` validated by Pydantic rules + coverage.

    The nested ``study`` / ``biosample`` / ``perturbation`` / ``provenance``
    definitions are overridden
    with subclasses that enforce their LinkML if/then rule. Validate with
    ``validate_experimental_metadata_v0_2``, which runs ``model_validate`` (required
    floor, types, enums, CURIE patterns, and the if/then rules) *and* the cross-field
    provenance-coverage check, merging both into one issue list.

    Coverage is deliberately **not** enforced inside a ``model_validator``: a
    ``mode="before"`` validator that raised would abort field validation, masking
    every required / type / enum / if-then error whenever coverage produced any
    message (the common case, since most objects have incomplete provenance). The
    only ``mode="before"`` step kept here is the provenance ``field_path`` injection,
    which never raises.
    """

    study: _StudyV0_2 = Field(default=...)
    biosample: _BiosampleV0_2 = Field(default=...)
    perturbation: Optional[_PerturbationAssignmentV0_2] = Field(default=None)
    provenance: dict[str, _FieldProvenanceV0_2] = Field(default=...)

    @model_validator(mode="before")
    @classmethod
    def _inject_provenance_field_path(cls, data: object) -> object:
        # Map form: copy each provenance key into its value's field_path so the
        # generated FieldProvenance (which requires field_path) validates a map
        # whose values carry only source_kind / citation / grounded. Never raises,
        # so the field validation that enforces the required floor and if/then rules
        # always runs (coverage is checked separately, see _coverage_issues).
        if not isinstance(data, dict):
            return data
        prov = data.get("provenance")
        if isinstance(prov, dict):
            data = {
                **data,
                "provenance": {
                    key: (
                        {**value, "field_path": key}
                        if isinstance(value, dict)
                        else value
                    )
                    for key, value in prov.items()
                },
            }
        return data


def _coverage_issues(obj: dict) -> list[Issue]:
    """Turn the provenance-coverage check into ``Issue`` objects.

    Orphan field_paths are MUST errors; uncovered fields are SHOULD warnings. Runs
    on the raw dict (not the constructed model) so coverage is reported even when the
    object also has field-validation errors — both layers surface in one pass.
    """
    from dca_helpers.validation.result import Issue, Severity

    if not isinstance(obj, dict):
        return []
    must_errors, should_warnings = _coverage_messages(obj)
    issues = [
        Issue(loc=("provenance",), message=msg, severity=Severity.ERROR)
        for msg in must_errors
    ]
    issues += [
        Issue(
            loc=("provenance",),
            message=(
                msg[len("[SHOULD]") :].strip() if msg.startswith("[SHOULD]") else msg
            ),
            severity=Severity.WARNING,
        )
        for msg in should_warnings
    ]
    return issues


def _sentinel_label_issues(obj: dict) -> list[Issue]:
    """The half of the id/label agreement rule that needs no ontology service.

    ``OntologyTerm`` states three cases. A real CURIE's label must be the ontology's
    label, which requires resolving the term and so belongs to
    ``ontology_lookup.check_ontology_terms``. The other two are pure same-object
    comparisons and are checked here, on the always-on path: the lookup guard runs
    only under ``--check-ontology`` because it needs the network, and it skips
    sentinels outright, so nothing would otherwise catch these. Both find their
    terms with the shared ``_ontology_terms.iter_ontology_terms`` walk.

    SHOULD rather than MUST so the rule can ship in a patch release.
    """
    from dca_helpers.validation.result import Issue, Severity

    if not isinstance(obj, dict):
        return []
    issues: list[Issue] = []
    for loc, term_id, label in iter_ontology_terms(obj):
        if term_id in ("unknown", "na") and label != term_id:
            issues.append(
                Issue(
                    loc=loc,
                    message=(
                        f"ontology_term_id is '{term_id}' but label is '{label}'; "
                        f"the label should repeat the sentinel. A label that names a "
                        f"real thing next to a '{term_id}' id is what consumers "
                        f"display, so it reads as a value the record does not have"
                    ),
                    severity=Severity.WARNING,
                )
            )
        elif term_id == "unavailable" and label in SENTINELS:
            issues.append(
                Issue(
                    loc=loc,
                    message=(
                        f"ontology_term_id is 'unavailable' but label is '{label}'; "
                        "'unavailable' means the value is known and has no ontology "
                        "term, so the label should carry that free-text value"
                    ),
                    severity=Severity.WARNING,
                )
            )
    return issues


def validate_experimental_metadata_v0_2(obj: dict) -> list[Issue]:
    """Validate one experimental-metadata object, returning ``Issue`` objects.

    Runs the layers and merges them: ``model_validate`` (required floor, types,
    enums, CURIE patterns, and the if/then rules), the provenance-coverage check,
    and the sentinel half of id/label agreement. Empty list when the object fully
    conforms.
    """
    import pydantic

    from dca_helpers.validation.validator import _convert_errors

    issues: list[Issue] = []
    try:
        DCAExperimentalMetadataV0_2.model_validate(obj)
    except pydantic.ValidationError as exc:
        issues.extend(_convert_errors(exc))
    issues.extend(_coverage_issues(obj))
    issues.extend(_sentinel_label_issues(obj))
    return issues


def validate_dca_experimental_metadata_v0_2(
    dca: dict, check_ontology: bool = False
) -> list[Issue]:
    """Validate the experimental metadata embedded in a store's ``dca`` block.

    The experimental blocks sit directly in the ``dca`` object alongside the channel
    metadata (``version`` / ``url`` / ``channels`` / ``normalization_statistics``).
    Pull just the experimental blocks and validate them as one ``ExperimentalMetadata``
    — enforcing the if/then rules and the provenance-coverage check. Each issue's
    ``loc`` is prefixed with ``"dca"`` so it points back into the block. Block names
    are derived from the generated model (the source of truth).

    When ``check_ontology`` is true, additionally run the opt-in OLS consistency
    guard (``ontology_lookup.check_ontology_terms``): every ``ontology_term_id`` is
    resolved against the EBI Ontology Lookup Service and its ``label`` compared to
    the ontology's canonical label. This makes network requests, so it is off by
    default and enabled by ``dca-validate --check-ontology``.
    """
    from dca_helpers.validation.result import Issue

    obj = {k: v for k, v in dca.items() if k in experimental_block_keys()}
    issues = validate_experimental_metadata_v0_2(obj)
    if check_ontology:
        from dca_helpers.validation.ontology_lookup import check_ontology_terms

        issues = issues + check_ontology_terms(obj)
    return [
        Issue(loc=("dca", *issue.loc), message=issue.message, severity=issue.severity)
        for issue in issues
    ]


def validate_perturbation_v0_2(obj: dict) -> list[Issue]:
    """Validate one ``Perturbation`` definition, returning ``Issue`` objects.

    Enforces the ``modality`` ⇒ typed-block rules (``compound`` ⇒ ``chemical``,
    genetic ⇒ ``genetic``) on top of the generated model's type / enum / required
    checks. Empty list when the object fully conforms. For a bare codebook entry:
    a ``Perturbation`` reached through ``ExperimentalMetadata.perturbation`` is
    already checked by ``validate_experimental_metadata_v0_2``.
    """
    import pydantic

    from dca_helpers.validation.validator import _convert_errors

    try:
        _PerturbationV0_2.model_validate(obj)
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)
    return []


def validate_well_perturbation_v0_2(obj: dict) -> list[Issue]:
    """Validate one ``WellPerturbation`` row, returning ``Issue`` objects.

    Enforces the non-``empty`` ⇒ ``definition`` rule, plus the ``Perturbation``
    modality rules on the nested ``definition`` / ``cotreatment``.
    Empty list when the object fully conforms. Standalone because
    ``WellPerturbation`` is a root object, never nested under ``ExperimentalMetadata``.
    """
    import pydantic

    from dca_helpers.validation.validator import _convert_errors

    try:
        _WellPerturbationV0_2.model_validate(obj)
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)
    return []
