"""
Drift-detection tests: assert structural consistency between
docs/v0.2/schema.json (documentation artifact) and the Pydantic models in
helpers/src/dca_helpers/validation/versions/v0_2/models.py (runtime enforcement).

Catches:
  - A required field added to schema.json but missing from the Pydantic model
    (or vice versa).
  - An enum value added to a frozenset in models.py but not to schema.json
    (or vice versa).
  - A minimum bound in schema.json not reflected as Field(ge=...) in the model.
  - A const in schema.json not reflected as a Literal[...] type in the model.

These tests read schema.json at test time and introspect model field metadata
via annotated_types.Ge and typing.get_args(Literal[...]).  No real Zarr stores
or S3 access required.
"""

import json
import typing

import annotated_types
import pytest

from dca_helpers.validation.versions.v0_2.models import (
    CHANNEL_TYPES,
    CPG_LABELED_MOLECULES,
    CPG_LABELED_STRUCTURES,
    MARKER_TYPES,
    DCAChannelMetadataV0_2,
    DCAChannelNormalization,
    DCAGroupMetadataV0_2,
    DCAIntensityStatistics,
    DCANormalizationMetadata,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def schema(root_dir) -> dict:
    schema_path = root_dir / "docs" / "v0.2" / "schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _required_model_fields(model) -> set:
    return {name for name, field in model.model_fields.items() if field.is_required()}


def _get_ge(model, field_name) -> float | None:
    for m in model.model_fields[field_name].metadata:
        if isinstance(m, annotated_types.Ge):
            return m.ge
    return None


def _get_literal_values(model, field_name) -> set | None:
    ann = model.model_fields[field_name].annotation
    if typing.get_origin(ann) is typing.Literal:
        return set(typing.get_args(ann))
    return None


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------


def test_dca_required_fields(schema):
    schema_required = set(schema["properties"]["dca"]["required"])
    model_required = _required_model_fields(DCAGroupMetadataV0_2)
    assert model_required == schema_required, (
        f"DCAGroupMetadataV0_2 required fields {model_required!r} "
        f"do not match schema.json dca.required {schema_required!r}"
    )


def test_channel_metadata_required_fields(schema):
    schema_required = set(schema["$defs"]["ChannelMetadata"]["required"])
    model_required = _required_model_fields(DCAChannelMetadataV0_2)
    assert model_required == schema_required, (
        f"DCAChannelMetadataV0_2 required fields {model_required!r} "
        f"do not match schema.json ChannelMetadata.required {schema_required!r}"
    )


def test_intensity_statistics_required_fields(schema):
    schema_required = set(schema["$defs"]["IntensityStatistics"]["required"])
    model_required = _required_model_fields(DCAIntensityStatistics)
    assert model_required == schema_required, (
        f"DCAIntensityStatistics required fields {model_required!r} "
        f"do not match schema.json IntensityStatistics.required {schema_required!r}"
    )


def test_channel_normalization_required_fields(schema):
    schema_required = set(schema["$defs"]["ChannelNormalization"]["required"])
    model_required = _required_model_fields(DCAChannelNormalization)
    assert model_required == schema_required, (
        f"DCAChannelNormalization required fields {model_required!r} "
        f"do not match schema.json ChannelNormalization.required {schema_required!r}"
    )


# ---------------------------------------------------------------------------
# Enum vocabularies
# ---------------------------------------------------------------------------


def test_enum_vocabularies(schema):
    channel_meta = schema["$defs"]["ChannelMetadata"]["properties"]
    bio_annotation = schema["$defs"]["BiologicalAnnotation"]["properties"]

    checks = [
        (
            CHANNEL_TYPES,
            channel_meta["channel_type"]["enum"],
            "CHANNEL_TYPES / ChannelMetadata.channel_type",
        ),
        (
            MARKER_TYPES,
            bio_annotation["marker_type"]["enum"],
            "MARKER_TYPES / BiologicalAnnotation.marker_type",
        ),
        (
            CPG_LABELED_STRUCTURES,
            bio_annotation["cpg_labeled_structure"]["enum"],
            "CPG_LABELED_STRUCTURES / BiologicalAnnotation.cpg_labeled_structure",
        ),
        (
            CPG_LABELED_MOLECULES,
            bio_annotation["cpg_labeled_molecule"]["enum"],
            "CPG_LABELED_MOLECULES / BiologicalAnnotation.cpg_labeled_molecule",
        ),
    ]
    for model_set, schema_enum, label in checks:
        assert model_set == set(schema_enum), (
            f"{label}: model frozenset {sorted(model_set)!r} "
            f"!= schema.json enum {sorted(schema_enum)!r}"
        )


# ---------------------------------------------------------------------------
# Numeric minimum bounds
# ---------------------------------------------------------------------------


def test_numeric_minimum_bounds(schema):
    intensity_props = schema["$defs"]["IntensityStatistics"]["properties"]
    norm_meta_props = schema["$defs"]["NormalizationMetadata"]["properties"]
    channel_meta_props = schema["$defs"]["ChannelMetadata"]["properties"]

    checks = [
        (DCAIntensityStatistics, "p95_p5", intensity_props["p95_p5"]["minimum"]),
        (DCAIntensityStatistics, "p99_p1", intensity_props["p99_p1"]["minimum"]),
        (DCAIntensityStatistics, "iqr", intensity_props["iqr"]["minimum"]),
        (DCANormalizationMetadata, "grid", norm_meta_props["grid"]["minimum"]),
        (DCAChannelMetadataV0_2, "index", channel_meta_props["index"]["minimum"]),
    ]
    for model, field_name, expected_min in checks:
        actual_ge = _get_ge(model, field_name)
        assert actual_ge == expected_min, (
            f"{model.__name__}.{field_name}: Field ge={actual_ge!r} "
            f"!= schema.json minimum={expected_min!r}"
        )


# ---------------------------------------------------------------------------
# Const / Literal
# ---------------------------------------------------------------------------


def test_version_const(schema):
    schema_const = schema["properties"]["dca"]["properties"]["version"]["const"]
    model_literals = _get_literal_values(DCAGroupMetadataV0_2, "version")
    assert model_literals == {schema_const}, (
        f"DCAGroupMetadataV0_2.version Literal values {model_literals!r} "
        f"do not match schema.json version const {schema_const!r}"
    )
