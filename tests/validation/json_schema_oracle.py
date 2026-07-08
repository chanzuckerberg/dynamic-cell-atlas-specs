"""
Test-only oracle: enforce the v0.2 experimental-metadata schema via JSON Schema.

This is **not** on the runtime path. The validator enforces the generated Pydantic
models (``validation/versions/v0_2/experimental_metadata.py``); this module loads the
committed JSON Schema and validates against it independently, so the test suite can
cross-check that the hand-written Pydantic ``if/then`` rules still agree with the
rules ``gen-json-schema`` derives straight from the LinkML source of truth. It lives
under ``tests/`` (imported as ``tests.validation.json_schema_oracle``) to make that
test-only role explicit and to keep ``jsonschema`` a dev/test dependency.

The LinkML schema (``schema/v0.2/dca_experimental_metadata.yaml``) is the single
source of truth. ``make schema`` generates the committed JSON Schema
(``schema/v0.2/generated/dca_experimental_metadata.schema.json``); this module loads
that JSON Schema and validates an experimental-metadata object against it.

Why JSON Schema makes a good oracle and the Pydantic models don't enforce it on their
own: the schema's conditional cross-field constraints are authored as LinkML ``rules``
(e.g. ``modality: compound`` requires the ``chemical`` block; ``tissue_type: "cell
line"`` requires ``cell_line_name``). ``gen-json-schema`` emits these as enforceable
``if/then`` blocks, but ``gen-pydantic`` records them only as inert metadata. So the
JSON Schema is the artifact that carries the rules straight from the SoT, and
``jsonschema`` is the engine that enforces them — exactly what an oracle needs.

Every JSON Schema violation is a MUST (the schema encodes only required-ness,
types, enums, CURIE-shape patterns, and the if/then rules), so all findings are
``Severity.ERROR``. The deferred ontology-aware layer — term existence,
term-id/label agreement, grounding re-verification — is not checked here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from dca_helpers.validation.result import Issue, Severity

# The root class curated per Biosample.group_id; also the default validation
# target for a full experimental-metadata object.
ROOT_CLASS = "ExperimentalMetadata"

_SCHEMA_RELPATH = Path("schema/v0.2/generated/dca_experimental_metadata.schema.json")


class SchemaNotFoundError(FileNotFoundError):
    """The committed JSON Schema artifact could not be located."""


def _find_schema_path() -> Path:
    """Locate the committed JSON Schema by walking up from this file.

    dca_helpers is installed editable / from source, so the generated artifact
    sits at ``<repo>/schema/v0.2/generated/...`` above the package. Search upward for
    it rather than hard-coding a parent depth.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _SCHEMA_RELPATH
        if candidate.is_file():
            return candidate
    raise SchemaNotFoundError(
        f"Could not find {_SCHEMA_RELPATH} above {__file__}. "
        "Run `make schema` to regenerate it from the LinkML source."
    )


def _close_provenance_shorthand(schema: dict) -> None:
    """Drop the bare-enum shorthand branch from the ``provenance`` map, in place.

    For an inlined multivalued slot whose range has a key, LinkML emits a
    "compact" shorthand: the ``provenance`` map's value may be either the full
    ``FieldProvenance`` object *or* a bare ``ProvenanceSourceKind`` string (the
    key slot alone). The bare-string branch is a hole — ``{"study.source":
    "agent"}`` validates without a citation, bypassing the ``agent ⇒ citation``
    rule that lives on the object form. We require the object form so that rule
    (and ``source_kind``-required) is always reached. This is a validator-layer
    fix, not a schema change: the LinkML SoT keeps the shorthand it generates.
    """
    try:
        prov = schema["$defs"]["ExperimentalMetadata"]["properties"]["provenance"]
    except KeyError:
        return
    additional = prov.get("additionalProperties")
    if not isinstance(additional, dict):
        return
    branches = additional.get("anyOf")
    if not branches:
        return
    object_branches = [
        b
        for b in branches
        if isinstance(b, dict) and "FieldProvenance" in b.get("$ref", "")
    ]
    if object_branches:
        prov["additionalProperties"] = object_branches[0]


@lru_cache(maxsize=1)
def _load_schema() -> dict:
    schema = json.loads(_find_schema_path().read_text())
    _close_provenance_shorthand(schema)
    return schema


@lru_cache(maxsize=None)
def _validator_for(target_class: str):
    """Return a cached jsonschema validator for ``$defs/<target_class>``.

    Only the ``$defs`` block is kept (so internal ``$ref``s resolve), with a
    top-level ``$ref`` pointing at the requested class. The rest of the root
    schema is intentionally dropped: with ``tree_root`` the root carries
    ``ExperimentalMetadata``'s own ``required`` / ``properties``, and spreading
    those here would impose the root's required blocks on every sub-class check
    (validating a ``Citation`` would otherwise wrongly demand ``study`` /
    ``biosample`` / ...).

    ``format`` keywords (``date`` / ``date-time`` / ``uri``) are enforced via the
    format checker; without it they are annotation-only and accept any string.
    """
    from jsonschema import Draft202012Validator

    schema = _load_schema()
    defs_key = "$defs" if "$defs" in schema else "definitions"
    if target_class not in schema.get(defs_key, {}):
        raise KeyError(
            f"Class '{target_class}' is not defined in the generated JSON Schema "
            f"(available under '{defs_key}')."
        )
    rooted = {defs_key: schema[defs_key], "$ref": f"#/{defs_key}/{target_class}"}
    return Draft202012Validator(
        rooted, format_checker=Draft202012Validator.FORMAT_CHECKER
    )


def validate_experimental_metadata(
    obj: dict,
    *,
    target_class: str = ROOT_CLASS,
) -> list[Issue]:
    """Validate one experimental-metadata object against the generated schema.

    Parameters
    ----------
    obj
        A decoded experimental-metadata object — the full ``ExperimentalMetadata``
        (a manifest Parquet row / the curated record) or a nested sub-object when
        ``target_class`` names one (e.g. ``Perturbation``, ``Biosample``).
    target_class
        The ``$defs`` class to validate against. Defaults to the root.

    Returns
    -------
    A list of :class:`Issue`, one per schema violation, all ``Severity.ERROR``.
    Empty when the object conforms.
    """
    validator = _validator_for(target_class)
    issues: list[Issue] = []
    for err in sorted(validator.iter_errors(obj), key=lambda e: list(e.absolute_path)):
        issues.append(
            Issue(
                loc=tuple(err.absolute_path),
                message=err.message,
                severity=Severity.ERROR,
            )
        )
    return issues


@lru_cache(maxsize=1)
def experimental_block_keys() -> frozenset[str]:
    """The top-level ``ExperimentalMetadata`` block names.

    Derived from the schema (``study``, ``biosample``, ``specimen``,
    ``study_component``, ``acquisition``, ``image_data``, ``provenance``) so it
    tracks the source of truth rather than being hard-coded.
    """
    defs_key = "$defs" if "$defs" in _load_schema() else "definitions"
    props = _load_schema()[defs_key][ROOT_CLASS].get("properties", {})
    return frozenset(props)


def validate_dca_experimental_metadata(dca: dict) -> list[Issue]:
    """Validate the experimental metadata embedded in a store's ``dca`` block.

    The experimental blocks sit directly in the ``dca`` object alongside the
    channel metadata (``version`` / ``url`` / ``channels`` /
    ``normalization_statistics``). Pull just the experimental blocks and validate
    them as one ``ExperimentalMetadata`` object — this enforces the required
    floor mirrored into ``zarr.json`` and the conditional rules. Each issue's
    ``loc`` is prefixed with ``"dca"`` so it points back into the block.
    """
    obj = {k: v for k, v in dca.items() if k in experimental_block_keys()}
    return [
        Issue(loc=("dca", *issue.loc), message=issue.message, severity=issue.severity)
        for issue in validate_experimental_metadata(obj, target_class=ROOT_CLASS)
    ]
