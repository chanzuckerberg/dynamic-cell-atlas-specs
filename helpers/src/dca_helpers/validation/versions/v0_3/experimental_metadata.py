"""
v0.3 ``ExperimentalMetadata`` validation.

The v0.3 LinkML schema (schema/v0.3/dca_experimental_metadata.yaml) differs from
v0.2 in two places:

* ``ImageData.processing_variant`` is an ordered list of processing steps instead of
  a single string, with a recommended ``ProcessingStep`` vocabulary.
* Perturbations: the ``PerturbationModality`` vocabulary is aligned with CELLxGENE
  7.1 (four CRISPR values replace ``crispr``; ``diet`` / ``protein`` / ``temperature``
  / ``not reported`` are added; ``unknown`` is removed), and ``perturbation`` is
  required on every image group, as ``{"modality": "no perturbations"}`` when
  nothing was applied.

Every other block is unchanged, so this module reuses the v0.2 rule classes and
overrides only ``image_data`` and the perturbation classes. If a later edit to the
v0.3 schema touches another block, override that block here too; the drift test in
tests/validation/versions/v0_3/test_experimental_metadata.py fails if the generated
v0.3 models stop matching the v0.2 ones anywhere else.

Rules added on top of the generated model:

* ``"raw"`` means "no processing" and MUST be the only item when present.
* Items SHOULD come from the ``ProcessingStep`` vocabulary.
* ``compound`` requires the ``chemical`` block and a genetic modality the
  ``genetic`` block (as in v0.2, with the v0.3 genetic values).
* ``perturbation.modality`` (``no perturbations``) stands alone; without it,
  ``control_class`` is required, and a non-``empty`` one requires ``definition``.

The projection-specific agreement (an image lists the projection step of each method
it uses exactly when it carries ``dca.projection``) is not experimental metadata and is
checked with the other projection rules in ``versions/v0_3/models.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import Field, model_validator

from dca_helpers.validation.versions.v0_2.experimental_metadata import (
    DCAExperimentalMetadataV0_2,
    _coverage_issues,
    _enum_value,
    _is_populated,
    _sentinel_label_issues,
    experimental_block_keys,
)
from dca_helpers.validation.versions.v0_3 import _generated_models

if TYPE_CHECKING:
    from dca_helpers.validation.result import Issue

PROCESSING_STEPS: frozenset[str] = frozenset(
    step.value for step in _generated_models.ProcessingStep
)
RAW_STEP = _generated_models.ProcessingStep.raw.value
MIP_STEP = _generated_models.ProcessingStep.maximum_intensity_projection.value

# projections.rst § Methods: each dca.projection method and the processing step a
# projection using it lists. Values come from the generated vocabulary, so a renamed
# step fails at import rather than drifting.
PROJECTION_STEP_BY_METHOD: dict[str, str] = {
    "max": MIP_STEP,
    "mean": _generated_models.ProcessingStep.mean_intensity_projection.value,
    "sum": _generated_models.ProcessingStep.sum_intensity_projection.value,
    "slice": _generated_models.ProcessingStep.z_slice.value,
}
PROJECTION_STEPS: frozenset[str] = frozenset(PROJECTION_STEP_BY_METHOD.values())


# Genetic modalities: each requires the ``genetic`` block (``compound`` requires
# ``chemical``). Mirrors the Perturbation ``modality`` if/then rules in the YAML.
GENETIC_MODALITIES: frozenset[str] = frozenset(
    {
        "CRISPR activation screen",
        "CRISPR interference screen",
        "CRISPR knockout mutant",
        "CRISPR knockout screen",
        "orf",
        "shrna",
        "mirna",
        "sirna",
    }
)
NO_PERTURBATIONS = _generated_models.NoPerturbations.no_perturbations.value
# The assignment fields that "no perturbations" excludes: everything but modality and
# the block's own provenance.
_ASSIGNMENT_FIELDS: tuple[str, ...] = (
    "control_class",
    "definition",
    "cotreatment",
    "concentration",
    "solvent",
    "duration_h",
    "timepoint_primary_h",
    "timepoint_secondary_h",
)

# A literal renamed in a schema regen would silently disable its rule; fail at import.
_MODALITIES = {m.value for m in _generated_models.PerturbationModality}
if not ({"compound"} | GENETIC_MODALITIES) <= _MODALITIES:
    raise AssertionError(
        "PerturbationModality is missing rule value(s) "
        f"{sorted(({'compound'} | GENETIC_MODALITIES) - _MODALITIES)}; regenerate the "
        "model or update GENETIC_MODALITIES."
    )


class _PerturbationV0_3(_generated_models.Perturbation):
    """``Perturbation`` with the ``modality`` ⇒ typed-block rules, v0.3 vocabulary."""

    @model_validator(mode="after")
    def _require_modality_block(self):
        modality = _enum_value(self.modality)
        if modality == "compound" and not _is_populated(self.chemical):
            raise ValueError(
                "chemical is required when modality is 'compound' "
                "(a compound perturbation must carry the chemical identifier block)"
            )
        if modality in GENETIC_MODALITIES and not _is_populated(self.genetic):
            raise ValueError(
                "genetic is required when modality is a genetic kind "
                f"({' / '.join(sorted(GENETIC_MODALITIES))}; got modality='{modality}')"
            )
        return self


class _PerturbationAssignmentV0_3(_generated_models.PerturbationAssignment):
    """``PerturbationAssignment`` with its two forms enforced.

    ``modality: "no perturbations"`` stands alone. Without it, ``control_class`` is
    required, and anything but ``empty`` must name a ``definition``. The nested
    definitions use ``_PerturbationV0_3`` so the modality rules apply to them too.
    """

    definition: Optional[_PerturbationV0_3] = Field(default=None)
    cotreatment: Optional[_PerturbationV0_3] = Field(default=None)

    @model_validator(mode="after")
    def _check_form(self):
        if _enum_value(self.modality) == NO_PERTURBATIONS:
            extra = [f for f in _ASSIGNMENT_FIELDS if _is_populated(getattr(self, f))]
            if extra:
                raise ValueError(
                    f"modality 'no perturbations' stands alone, but the block also "
                    f"sets {extra}; when nothing was applied, record nothing else"
                )
            return self
        control_class = _enum_value(self.control_class)
        if control_class is None:
            raise ValueError(
                "control_class is required unless modality is 'no perturbations' "
                "(record {'modality': 'no perturbations'} when nothing was applied)"
            )
        if control_class != "empty" and not _is_populated(self.definition):
            raise ValueError(
                "definition is required when control_class is not 'empty' "
                "(treated and control material must name a perturbation)"
            )
        return self


class _WellPerturbationV0_3(  # type: ignore[misc]
    _generated_models.WellPerturbation, _PerturbationAssignmentV0_3
):
    """``WellPerturbation`` — a v0.3 ``PerturbationAssignment`` plus ``(plate, well)``.

    The nested overrides are restated because the generated ``WellPerturbation``
    precedes the ruled class in the MRO and redeclares them with the generated range
    (see ``_WellPerturbationV0_2``).
    """

    definition: Optional[_PerturbationV0_3] = Field(default=None)
    cotreatment: Optional[_PerturbationV0_3] = Field(default=None)


class DCAExperimentalMetadataV0_3(DCAExperimentalMetadataV0_2):
    """v0.3 ``ExperimentalMetadata``: the v0.2 rules with the v0.3 ``ImageData`` and
    perturbation classes, and ``perturbation`` required."""

    image_data: Optional[_generated_models.ImageData] = Field(default=None)
    perturbation: _PerturbationAssignmentV0_3 = Field(default=...)


def _processing_variant_issues(obj: dict) -> list[Issue]:
    """The two list rules the generated model cannot express."""
    from dca_helpers.validation.result import Issue, Severity

    image_data = obj.get("image_data") if isinstance(obj, dict) else None
    steps = (
        image_data.get("processing_variant") if isinstance(image_data, dict) else None
    )
    # A non-list value is a type error the model already reports.
    if not isinstance(steps, list):
        return []
    loc = ("image_data", "processing_variant")
    issues: list[Issue] = []
    if RAW_STEP in steps and len(steps) > 1:
        issues.append(
            Issue(
                loc=loc,
                message=(
                    f"processing_variant lists '{RAW_STEP}' together with other steps "
                    f"{steps}; '{RAW_STEP}' means no processing and MUST be the only "
                    "item when present"
                ),
                severity=Severity.ERROR,
            )
        )
    unknown = [s for s in steps if isinstance(s, str) and s not in PROCESSING_STEPS]
    if unknown:
        issues.append(
            Issue(
                loc=loc,
                message=(
                    f"processing_variant items {unknown} are not in the ProcessingStep "
                    f"vocabulary {sorted(PROCESSING_STEPS)}; items should be short step "
                    "names, with any qualifying detail in image_data.processing_description"
                ),
                severity=Severity.WARNING,
            )
        )
    return issues


def validate_experimental_metadata_v0_3(obj: dict) -> list[Issue]:
    """Validate one v0.3 experimental-metadata object, returning ``Issue`` objects.

    The v0.2 layers (model validation, provenance coverage, sentinel labels) against
    the v0.3 model, plus the ``processing_variant`` list rules.
    """
    import pydantic

    from dca_helpers.validation.validator import _convert_errors

    issues: list[Issue] = []
    try:
        DCAExperimentalMetadataV0_3.model_validate(obj)
    except pydantic.ValidationError as exc:
        issues.extend(_convert_errors(exc))
    issues.extend(_coverage_issues(obj))
    issues.extend(_sentinel_label_issues(obj))
    issues.extend(_processing_variant_issues(obj))
    return issues


def validate_dca_experimental_metadata_v0_3(
    dca: dict, check_ontology: bool = False
) -> list[Issue]:
    """v0.3 counterpart of ``validate_dca_experimental_metadata_v0_2``: pull the
    experimental blocks out of a ``dca`` object and validate them, with each issue's
    ``loc`` prefixed by ``"dca"``."""
    from dca_helpers.validation.result import Issue

    obj = {k: v for k, v in dca.items() if k in experimental_block_keys()}
    issues = validate_experimental_metadata_v0_3(obj)
    if check_ontology:
        from dca_helpers.validation.ontology_lookup import check_ontology_terms

        issues = issues + check_ontology_terms(obj)
    return [
        Issue(loc=("dca", *issue.loc), message=issue.message, severity=issue.severity)
        for issue in issues
    ]


def validate_perturbation_v0_3(obj: dict) -> list[Issue]:
    """Validate one v0.3 ``Perturbation`` definition (a bare codebook entry)."""
    import pydantic

    from dca_helpers.validation.validator import _convert_errors

    try:
        _PerturbationV0_3.model_validate(obj)
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)
    return []


def validate_well_perturbation_v0_3(obj: dict) -> list[Issue]:
    """Validate one v0.3 ``WellPerturbation`` row, returning ``Issue`` objects."""
    import pydantic

    from dca_helpers.validation.validator import _convert_errors

    try:
        _WellPerturbationV0_3.model_validate(obj)
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)
    return []
