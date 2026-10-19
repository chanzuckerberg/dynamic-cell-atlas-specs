"""
v0.3 ``ExperimentalMetadata`` validation.

The v0.3 LinkML schema (schema/v0.3/dca_experimental_metadata.yaml) differs from
v0.2 in one slot: ``ImageData.processing_variant`` is an ordered list of processing
steps instead of a single string, with a recommended ``ProcessingStep`` vocabulary.
Every other block is unchanged, so this module reuses the v0.2 rule classes and
overrides only ``image_data``. If a later edit to the v0.3 schema touches another
block, override that block here too; the drift test in
tests/validation/versions/v0_3/test_experimental_metadata.py fails if the generated
v0.3 models stop matching the v0.2 ones anywhere else.

Rules added on top of the generated model:

* ``"raw"`` means "no processing" and MUST be the only item when present.
* Items SHOULD come from the ``ProcessingStep`` vocabulary.

The projection-specific agreement (an image lists "maximum intensity projection"
exactly when it carries ``dca.projection``) is not experimental metadata and is
checked with the other projection rules in ``versions/v0_3/models.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import Field

from dca_helpers.validation.versions.v0_2.experimental_metadata import (
    DCAExperimentalMetadataV0_2,
    _coverage_issues,
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


class DCAExperimentalMetadataV0_3(DCAExperimentalMetadataV0_2):
    """v0.3 ``ExperimentalMetadata``: the v0.2 rules with the v0.3 ``ImageData``."""

    image_data: Optional[_generated_models.ImageData] = Field(default=None)


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
