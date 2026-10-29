"""
Drift-detection tests for v0.3: docs/v0.3/schema.json (documentation artifact) against
the Pydantic models in helpers/src/dca_helpers/validation/versions/v0_3/models.py
(runtime enforcement), for what v0.3 adds (``dca.projection``).

Two layers:

* Structural, like the v0.2 drift tests: required fields, enums and the version const.
* Behavioural: a set of ``dca.projection`` blocks is validated by both the JSON
  Schema and ``DCAProjectionV0_3``, and they must agree. Rules a JSON Schema cannot
  express (orderings, per-timepoint list lengths, bounds that depend on another field)
  are listed in ``MODEL_ONLY``, where the model must reject and the schema accepts.
"""

import json
import typing

import pydantic
import pytest

from dca_helpers.validation.versions.v0_3.models import (
    DCAGroupMetadataV0_3,
    DCAProjectionV0_3,
    ProjectionChannelV0_3,
    ProjectionFocusV0_3,
)

jsonschema = pytest.importorskip("jsonschema")

pytestmark = pytest.mark.unit


@pytest.fixture(scope="session")
def schema(root_dir) -> dict:
    return json.loads((root_dir / "docs" / "v0.3" / "schema.json").read_text("utf-8"))


@pytest.fixture(scope="session")
def projection_validator(schema):
    return jsonschema.Draft202012Validator(
        {**schema["$defs"]["Projection"], "$defs": schema["$defs"]}
    )


def _required_model_fields(model) -> set:
    return {name for name, field in model.model_fields.items() if field.is_required()}


def _literal_values(model, field_name) -> set:
    ann = model.model_fields[field_name].annotation
    for arg in (ann, *typing.get_args(ann)):
        if typing.get_origin(arg) is typing.Literal:
            return set(typing.get_args(arg))
    raise AssertionError(f"{model.__name__}.{field_name} has no Literal")


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,def_name",
    [
        (DCAProjectionV0_3, "Projection"),
        (ProjectionChannelV0_3, "ProjectionChannel"),
        (ProjectionFocusV0_3, "ProjectionFocus"),
    ],
)
def test_required_fields(schema, model, def_name):
    assert _required_model_fields(model) == set(
        schema["$defs"][def_name].get("required", [])
    )


def test_method_enums(schema):
    d = schema["$defs"]
    assert _literal_values(DCAProjectionV0_3, "method") == set(
        d["Projection"]["properties"]["method"]["enum"]
    )
    assert _literal_values(ProjectionChannelV0_3, "method") == set(
        d["ProjectionChannel"]["properties"]["method"]["enum"]
    )


def test_focus_per_enum(schema):
    assert _literal_values(ProjectionFocusV0_3, "per") == set(
        schema["$defs"]["ProjectionFocus"]["properties"]["per"]["enum"]
    )


def test_version_const(schema):
    assert _literal_values(DCAGroupMetadataV0_3, "version") == {
        schema["properties"]["dca"]["properties"]["version"]["const"]
    }


# ---------------------------------------------------------------------------
# Behavioural: schema and model agree on each block
# ---------------------------------------------------------------------------

_UNIFORM = {
    "source_store": "plate.ome.zarr",
    "method": "max",
    "z_start": 0,
    "z_end": 16,
    "z_dim_source": 16,
}
_CHANNELS = {
    "membrane": {"method": "max", "z_start": 0, "z_end": 86},
    "nuclei": {"method": "mean", "z_start": [0, 0, 0], "z_end": [59, 58, 60]},
    "mCherry": {"method": "slice", "z_start": 42, "z_end": 43},
}
_PER_CHANNEL = {
    "source_store": "plate.ome.zarr",
    "z_dim_source": 86,
    "channels": _CHANNELS,
}

AGREE = {
    # valid
    "uniform full": (_UNIFORM, True),
    "uniform minimal": ({"method": "max"}, True),
    "uniform with step": ({**_UNIFORM, "z_step": 2}, True),
    "uniform slice": ({**_UNIFORM, "method": "slice", "z_start": 3, "z_end": 4}, True),
    "per-channel": (_PER_CHANNEL, True),
    "per-channel, no source (before deposit)": (
        {
            "z_dim_source": 7,
            "channels": {"GFP": {"method": "max", "z_start": 0, "z_end": 7}},
        },
        True,
    ),
    "per-channel without planes": ({"channels": {"GFP": {"method": "max"}}}, True),
    "focus per timepoint with nulls": (
        {
            **_PER_CHANNEL,
            "focus": {"channel": "P", "per": "timepoint", "z_per_t": [19, None, 20]},
        },
        True,
    ),
    "focus per fov not found": (
        {**_PER_CHANNEL, "focus": {"channel": "P", "per": "fov", "z": None}},
        True,
    ),
    # invalid
    "neither form": ({"source_store": "p"}, False),
    "both forms": ({**_PER_CHANNEL, "method": "max"}, False),
    "unknown method": ({**_UNIFORM, "method": "min"}, False),
    "uniform fields in per-channel form": ({**_PER_CHANNEL, "z_fraction": 1.0}, False),
    "uniform range without z_dim_source": (
        {"method": "max", "z_start": 0, "z_end": 4},
        False,
    ),
    "channel range without z_dim_source": (
        {"channels": {"a": {"method": "max", "z_start": 0, "z_end": 4}}},
        False,
    ),
    "channel z_start without z_end": (
        {"z_dim_source": 9, "channels": {"a": {"method": "max", "z_start": 0}}},
        False,
    ),
    "z_step without range": (
        {"channels": {"a": {"method": "max", "z_step": 2}}},
        False,
    ),
    "slice without planes": ({"channels": {"a": {"method": "slice"}}}, False),
    "focus per fov without z": (
        {**_PER_CHANNEL, "focus": {"channel": "P", "per": "fov"}},
        False,
    ),
    "focus per timepoint without z_per_t": (
        {**_PER_CHANNEL, "focus": {"channel": "P", "per": "timepoint"}},
        False,
    ),
}

# The model rejects these; a JSON Schema cannot express the rule, so it accepts them.
MODEL_ONLY = {
    "z_start >= z_end": {**_UNIFORM, "z_start": 5, "z_end": 5},
    "z_end beyond z_dim_source": {**_UNIFORM, "z_end": 20},
    "uniform slice of several planes": {**_UNIFORM, "method": "slice"},
    "per-timepoint lists of different lengths": {
        "z_dim_source": 86,
        "channels": {"a": {"method": "mean", "z_start": [0, 0], "z_end": [5, 6, 7]}},
    },
    "int z_start with list z_end": {
        "z_dim_source": 86,
        "channels": {"a": {"method": "mean", "z_start": 0, "z_end": [5, 6]}},
    },
    "focus plane outside the source": {
        **_PER_CHANNEL,
        "focus": {"channel": "P", "per": "fov", "z": 90},
    },
}


def _model_accepts(block) -> bool:
    try:
        DCAProjectionV0_3.model_validate(block)
        return True
    except pydantic.ValidationError:
        return False


@pytest.mark.parametrize("name", list(AGREE))
def test_schema_and_model_agree(projection_validator, name):
    block, expected = AGREE[name]
    assert projection_validator.is_valid(block) is expected, f"schema: {name}"
    assert _model_accepts(block) is expected, f"model: {name}"


@pytest.mark.parametrize("name", list(MODEL_ONLY))
def test_model_only_rules(projection_validator, name):
    block = MODEL_ONLY[name]
    assert projection_validator.is_valid(
        block
    ), f"schema now rejects {name}: move it to AGREE"
    assert not _model_accepts(block), f"model accepts {name}"
