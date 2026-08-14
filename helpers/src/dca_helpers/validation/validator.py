"""
Generic, version-agnostic validator.

Flow for one Zarr store:
  1. zarr.open_group()            — open store, read raw ome and dca attrs
  1a. GEFF skip                   — if raw attrs contain "geff" key, return a
                                    skipped result immediately (GEFF stores are not
                                    OME-Zarr and are not validated)
  1b. HCS fast path               — if raw attrs contain "plate", bypass open_ome_zarr
                                    (which eagerly opens all wells+fields) and call
                                    _validate_hcs_plate directly from raw attrs
  2. open_ome_zarr()              — OME NGFF structural validation (non-HCS stores only)
  3. classify_group()             — determine ZarrNodeType, enforce mutual exclusivity
  4. Dispatch by node type:
       IMAGE        → _validate_image()       (DCAStoreSpec model)
       IMAGE_LABEL  → _validate_image_label() (DCALabelStoreSpec model
                                              — not yet registered)
       LABELS_LIST  → _validate_labels_list() (no arrays — pass immediately)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import obstore.exceptions as _obs_exc
import pydantic
from ome_zarr_models import open_ome_zarr
from ome_zarr_models.v05.image import Image

from dca_helpers import CURRENT_SPEC_VERSION
from dca_helpers.validation.discovery import _open_zarr_group
from dca_helpers.validation.registry import (
    UnsupportedSpecVersionError,
    get_label_model,
    get_model,
)
from dca_helpers.validation.result import Issue, Severity, ZarrNodeValidationResult
from dca_helpers.validation.versions.v0_2.experimental_metadata import (
    validate_dca_experimental_metadata_v0_2,
    validate_well_perturbation_v0_2,
)
from dca_helpers.validation.zarr_node import ZarrNodeType, classify_group

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic_zarr.v3 import ArraySpec


# ---------------------------------------------------------------------------
# Store metadata extraction
# ---------------------------------------------------------------------------


def _get_compression(codecs: tuple | list) -> tuple[str | None, int | None]:
    """
    Walk a codec chain (list/tuple of dicts) and return (codec_id, level) for
    the first compression codec found.  Recurses into sharding_indexed inner codecs.

    Each codec dict has the shape: {'name': str, 'configuration': dict}
    where 'configuration' is optional (e.g. bytes codec has no configuration).
    """
    for codec in codecs:
        name = codec.get("name", "")
        cfg = codec.get("configuration") or {}

        if name == "sharding_indexed":
            return _get_compression(cfg.get("codecs", ()))

        if name == "blosc":
            return "blosc", cfg.get("clevel")

        if name == "zstd":
            return "zstd", cfg.get("level")

        if name == "lz4":
            return "lz4", cfg.get("acceleration")

        if name == "gzip":
            return "gzip", cfg.get("level")

    return None, None


def _extract_level(level_index: int, path: str, spec: "ArraySpec") -> dict:
    """
    Extract DCA-relevant metadata from one pydantic_zarr ArraySpec.

    In zarr v3 with sharding:
      - spec.chunk_grid['configuration']['chunk_shape']  → outer shard (write unit)
      - codecs[0]['configuration']['chunk_shape']        → inner chunk (read unit)
      - compression lives inside the sharding codec's inner codecs

    Without sharding:
      - spec.chunk_grid['configuration']['chunk_shape']  → chunk shape
      - compression is in codecs directly
    """
    codecs = spec.codecs  # tuple of dicts
    shard_shape = tuple(spec.chunk_grid["configuration"]["chunk_shape"])

    if codecs and codecs[0].get("name") == "sharding_indexed":
        cfg = codecs[0]["configuration"]
        chunk_shape = tuple(cfg["chunk_shape"])
        inner_codecs = cfg.get("codecs", ())
    else:
        chunk_shape = shard_shape
        shard_shape = None
        inner_codecs = codecs

    codec_id, codec_level = _get_compression(inner_codecs)
    dim_names = spec.dimension_names
    return {
        "level_index": level_index,
        "path": path,
        "dtype": str(spec.data_type),
        "shape": list(spec.shape),
        "chunk_shape": list(chunk_shape),
        "shard_shape": list(shard_shape) if shard_shape else None,
        "codec_id": codec_id or "none",
        "codec_level": codec_level,
        "dimension_names": list(dim_names) if dim_names is not None else None,
    }


def _build_node_dict(img: Image) -> dict:
    """
    Build the dict that DCAStoreSpec.model_validate() expects from an
    ome_zarr_models Image (or compatible multiscale group).
    """
    ms = img.ome_attributes.multiscales[0]
    axes = [ax.name for ax in ms.axes]
    axis_units = {
        ax.name: (
            ax.unit if isinstance(ax.unit, str) or ax.unit is None else str(ax.unit)
        )
        for ax in ms.axes
    }
    levels = []
    shapes = []
    for i, ds in enumerate(ms.datasets):
        spec = img.members[ds.path]
        levels.append(_extract_level(i, ds.path, spec))
        shapes.append(list(spec.shape))
    return {
        "axes": axes,
        "axis_units": axis_units,
        "multiscale_level_count": len(ms.datasets),
        "levels": levels,
        "array_shapes": shapes,
    }


# ---------------------------------------------------------------------------
# HCS plate traversal
# ---------------------------------------------------------------------------

_ALPHANUMERIC = re.compile(r"^[a-zA-Z0-9]+$")
# OME-NGFF v0.5 well groups may contain a 'labels' sub-group alongside field
# images. Exclude it (and any future spec-defined non-field keys) from the
# orphan check so 'labels' is never reported as an undeclared field.
_WELL_NON_FIELD_KEYS: frozenset[str] = frozenset({"labels"})


def _check_plate_schema(raw_plate_attrs: dict) -> list[Issue]:
    """
    Validate plate metadata schema constraints previously enforced by
    ome-zarr-models PlateBase Pydantic validators (bypassed by the HCS fast path).

    Checks (errors raised when condition is violated)
    --------------------------------------------------
    MUST  — row/column names must be alphanumeric
    MUST  — row names must be unique within plate.rows
    MUST  — column names must be unique within plate.columns
    MUST  — acquisition id must be a non-negative integer
    MUST  — well path must contain exactly one '/' (<row>/<column>)
    MUST  — well path row part must reference a declared row
    MUST  — well path column part must reference a declared column
    """
    issues: list[Issue] = []

    # rows — alphanumeric and unique
    row_names: list[str] = []
    for row in raw_plate_attrs.get("rows", []):
        name = str(row.get("name", ""))
        row_names.append(name)
        if not _ALPHANUMERIC.match(name):
            issues.append(
                Issue(
                    loc=("plate", "rows", name),
                    message=f"Row name {name!r} must be alphanumeric ([a-zA-Z0-9])",
                    severity=Severity.ERROR,
                )
            )
    seen: set[str] = set()
    for name in row_names:
        if name in seen:
            issues.append(
                Issue(
                    loc=("plate", "rows"),
                    message=f"Duplicate row name {name!r} in plate.rows",
                    severity=Severity.ERROR,
                )
            )
        seen.add(name)

    # columns — alphanumeric and unique
    col_names: list[str] = []
    for col in raw_plate_attrs.get("columns", []):
        name = str(col.get("name", ""))
        col_names.append(name)
        if not _ALPHANUMERIC.match(name):
            issues.append(
                Issue(
                    loc=("plate", "columns", name),
                    message=f"Column name {name!r} must be alphanumeric ([a-zA-Z0-9])",
                    severity=Severity.ERROR,
                )
            )
    seen = set()
    for name in col_names:
        if name in seen:
            issues.append(
                Issue(
                    loc=("plate", "columns"),
                    message=f"Duplicate column name {name!r} in plate.columns",
                    severity=Severity.ERROR,
                )
            )
        seen.add(name)

    # fieldCount — must be a positive integer if present
    # OME-NGFF v0.5 zarr.json uses camelCase "fieldCount"
    field_count = raw_plate_attrs.get("fieldCount")
    if field_count is not None and (
        not isinstance(field_count, int) or field_count < 1
    ):
        issues.append(
            Issue(
                loc=("plate", "fieldCount"),
                message=(
                    f"plate.fieldCount must be a positive integer, got {field_count!r}"
                ),
                severity=Severity.ERROR,
            )
        )

    # acquisitions — id is required, must be a non-negative integer, and must be unique
    seen_aq_ids: set[int] = set()
    for aq_idx, aq in enumerate(raw_plate_attrs.get("acquisitions", [])):
        if "id" not in aq:
            issues.append(
                Issue(
                    loc=("plate", "acquisitions", str(aq_idx)),
                    message="Acquisition entry is missing required 'id' field",
                    severity=Severity.ERROR,
                )
            )
        else:
            aq_id = aq["id"]
            if aq_id is None or not isinstance(aq_id, int) or aq_id < 0:
                issues.append(
                    Issue(
                        loc=("plate", "acquisitions", str(aq_idx), "id"),
                        message=(
                            f"Acquisition id must be a non-negative integer, got {aq_id!r}"
                        ),
                        severity=Severity.ERROR,
                    )
                )
            else:
                if aq_id in seen_aq_ids:
                    issues.append(
                        Issue(
                            loc=("plate", "acquisitions", str(aq_idx), "id"),
                            message=f"Duplicate acquisition id {aq_id!r} in plate.acquisitions",
                            severity=Severity.ERROR,
                        )
                    )
                seen_aq_ids.add(aq_id)

    # well paths — uniqueness, syntax, and row/column cross-references
    row_name_set = set(row_names)
    col_name_set = set(col_names)
    seen_well_paths: set[str] = set()
    for well in raw_plate_attrs.get("wells", []):
        well_path = well.get("path", "")

        if well_path in seen_well_paths:
            issues.append(
                Issue(
                    loc=("plate", "wells", well_path),
                    message=f"Duplicate well path {well_path!r} in plate.wells",
                    severity=Severity.ERROR,
                )
            )
            continue
        seen_well_paths.add(well_path)

        parts = well_path.split("/")
        if len(parts) != 2:
            issues.append(
                Issue(
                    loc=("plate", "wells", well_path),
                    message=(
                        f"Well path {well_path!r} must contain exactly one '/' "
                        f"(expected <row>/<column>)"
                    ),
                    severity=Severity.ERROR,
                )
            )
            continue
        row_part, col_part = parts
        if row_name_set and row_part not in row_name_set:
            issues.append(
                Issue(
                    loc=("plate", "wells", well_path),
                    message=(
                        f"Well path row {row_part!r} is not declared in "
                        f"plate.rows: {sorted(row_name_set)}"
                    ),
                    severity=Severity.ERROR,
                )
            )
        if col_name_set and col_part not in col_name_set:
            issues.append(
                Issue(
                    loc=("plate", "wells", well_path),
                    message=(
                        f"Well path column {col_part!r} is not declared in "
                        f"plate.columns: {sorted(col_name_set)}"
                    ),
                    severity=Severity.ERROR,
                )
            )

    return issues


def _gather_plate_fields(
    node_path: str,
    plate_wells: list[dict],
    plate_acquisitions: "list[dict] | None" = None,
    *,
    spec_version: str,
) -> "tuple[list[Issue], list[str], dict[str, object]]":
    """
    Open each declared well zarr.json once and return:
      - issues for wells / fields that fail per-well traversal checks
      - field paths for all declared fields that exist on disk
      - a well_path -> well_perturbation mapping (raw attrs value, or None
        when absent) for every well opened, so the caller can cross-check
        each well's row against its field-of-view consensus after the field
        loop runs, without reopening well groups (see #154)

    Handles only checks that require opening well groups on disk.
    Plate-level and well-path-level schema checks (row/column names,
    well path syntax, acquisition id format, fieldCount) are the
    responsibility of _check_plate_schema, which must be called first.

    Uses raw zarr attrs rather than ome-zarr-models Well objects to avoid
    Well.from_zarr(), which eagerly opens all Image sub-groups via
    get_optional_group_paths() — O(N_fields) sequential S3 calls.

    Checks (errors raised when condition is violated)
    --------------------------------------------------
    MUST  — every declared well must have a group on disk
    MUST  — every declared field must have a group on disk
    MUST  — field image path must be alphanumeric ([a-zA-Z0-9])
    MUST  — field image paths must be unique within a well
    MUST  — field acquisition ID must be in plate.acquisitions (if acquisitions
            are declared); only checked for fields that exist on disk
    MUST  — no orphan field groups on disk (sub-groups of a well not declared
            in ome.well.images); "labels" is excluded as it is the only
            OME-NGFF-specified non-field sub-group allowed in a well
    MUST  — a well group's dca.well_perturbation, when present, satisfies
            WellPerturbation's rules (dispatched to
            validate_well_perturbation_v0_2 rather than the image-group entry
            point, which would filter the block out and misreport the
            image-only required floor as missing — see validator.rst).
            Absence is never an error: "this well has a perturbation" is a
            fact about the experiment, not the store. Gated the same way as
            the image-level check, on spec_version being an experimental one.
    """
    issues: list[Issue] = []
    field_paths: list[str] = []
    well_perturbation_by_well: dict[str, object] = {}
    check_well_perturbation = spec_version in _EXPERIMENTAL_SPEC_VERSIONS

    valid_aq_ids: set[int] | None = None
    if plate_acquisitions is not None:
        valid_aq_ids = {
            aq["id"]
            for aq in plate_acquisitions
            if isinstance(aq.get("id"), int) and aq["id"] >= 0
        }

    for well_meta in plate_wells:
        well_path = well_meta["path"]
        try:
            well_group = _open_zarr_group(f"{node_path}/{well_path}")
        except Exception:
            issues.append(
                Issue(
                    loc=("plate", "wells", well_path),
                    message=(
                        f"Well '{well_path}' is declared in "
                        f"ome.plate.wells but has no group on disk"
                    ),
                    severity=Severity.ERROR,
                )
            )
            continue

        well_attrs = dict(well_group.attrs)
        well_ome = well_attrs.get("ome", {})
        well_images = well_ome.get("well", {}).get("images", [])
        try:
            well_keys = set(well_group.keys())
        except Exception:
            well_keys = set()

        well_dca = well_attrs.get("dca")
        well_perturbation = (
            well_dca.get("well_perturbation") if isinstance(well_dca, dict) else None
        )
        well_perturbation_by_well[well_path] = well_perturbation
        if check_well_perturbation and well_perturbation is not None:
            for wp_issue in validate_well_perturbation_v0_2(well_perturbation):
                issues.append(
                    Issue(
                        loc=(
                            "plate",
                            "wells",
                            well_path,
                            "dca",
                            "well_perturbation",
                            *wp_issue.loc,
                        ),
                        message=wp_issue.message,
                        severity=wp_issue.severity,
                    )
                )

        seen_image_paths: set[str] = set()
        for img_meta in well_images:
            img_path = img_meta.get("path", "")
            if not img_path:
                continue

            if not _ALPHANUMERIC.match(img_path):
                issues.append(
                    Issue(
                        loc=("plate", "wells", well_path, img_path),
                        message=(
                            f"Field image path {img_path!r} in well '{well_path}' "
                            f"must be alphanumeric ([a-zA-Z0-9])"
                        ),
                        severity=Severity.ERROR,
                    )
                )

            if img_path in seen_image_paths:
                issues.append(
                    Issue(
                        loc=("plate", "wells", well_path, img_path),
                        message=(
                            f"Duplicate field image path {img_path!r} declared "
                            f"in well '{well_path}'"
                        ),
                        severity=Severity.ERROR,
                    )
                )
                continue
            seen_image_paths.add(img_path)

            if img_path not in well_keys:
                issues.append(
                    Issue(
                        loc=("plate", "wells", well_path, img_path),
                        message=(
                            f"Field '{well_path}/{img_path}' is declared in "
                            f"ome.well.images but has no group on disk"
                        ),
                        severity=Severity.ERROR,
                    )
                )
            else:
                field_paths.append(f"{well_path}/{img_path}")
                aq_id = img_meta.get("acquisition")
                if (
                    valid_aq_ids is not None
                    and aq_id is not None
                    and aq_id not in valid_aq_ids
                ):
                    issues.append(
                        Issue(
                            loc=("plate", "wells", well_path, img_path, "acquisition"),
                            message=(
                                f"Acquisition ID {aq_id!r} for field "
                                f"'{well_path}/{img_path}' is not in "
                                f"ome.plate.acquisitions: {sorted(valid_aq_ids)}"
                            ),
                            severity=Severity.ERROR,
                        )
                    )

        for key in sorted(well_keys - seen_image_paths - _WELL_NON_FIELD_KEYS):
            issues.append(
                Issue(
                    loc=("plate", "wells", well_path, key),
                    message=(
                        f"Field group '{well_path}/{key}' exists on disk "
                        f"but is not declared in ome.well.images"
                    ),
                    severity=Severity.ERROR,
                )
            )

    return issues, field_paths, well_perturbation_by_well


# ---------------------------------------------------------------------------
# Pydantic error conversion
# ---------------------------------------------------------------------------

_SHOULD_MARKER = "[SHOULD]"

# Spec versions whose dca block mirrors the experimental-metadata required floor
# (validated against the generated JSON Schema). v0.1 predates experimental
# metadata, so it is excluded.
_EXPERIMENTAL_SPEC_VERSIONS: frozenset[str] = frozenset({"0.2"})


def _convert_errors(exc: pydantic.ValidationError) -> list[Issue]:
    """
    Convert a pydantic.ValidationError into a list of Issue objects.

    Validators in models.py prefix SHOULD-level messages with "[SHOULD]".
    Everything else is treated as a MUST violation (ERROR severity).

    A single ValueError can contain multiple newline-separated messages (used
    when a model validator accumulates several SHOULD warnings).  Each line
    becomes its own Issue.
    """
    issues: list[Issue] = []

    for error in exc.errors(include_url=False):
        loc = tuple(error.get("loc", ()))
        raw_msg: str = error.get("msg", "")

        # Pydantic prefixes user messages with "Value error, " — strip it.
        if raw_msg.startswith("Value error, "):
            raw_msg = raw_msg[len("Value error, ") :]

        for line in raw_msg.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(_SHOULD_MARKER):
                issues.append(
                    Issue(
                        loc=loc,
                        message=line[len(_SHOULD_MARKER) :].strip(),
                        severity=Severity.WARNING,
                    )
                )
            else:
                issues.append(Issue(loc=loc, message=line, severity=Severity.ERROR))

    return issues


# ---------------------------------------------------------------------------
# Per-node-type validation helpers
# ---------------------------------------------------------------------------


def _validate_image(
    node_path: str,
    ome_obj: Image,
    ngff_version: str | None,
    spec_version: str,
    ModelClass: "type[BaseModel]",
    *,
    dca: object,
    parent_path: str | None = None,
    check_ontology: bool = False,
) -> list[ZarrNodeValidationResult]:
    """
    Run DCA spec validation for one image or label image multiscale group.

    Used directly for IMAGE nodes and called per-field by _validate_hcs_plate.
    The caller supplies the appropriate ModelClass (image or label) and the
    raw dca attrs dict to inject (None for label groups).
    """
    try:
        node_dict = _build_node_dict(ome_obj)
    except Exception as exc:
        return [
            ZarrNodeValidationResult(
                node_path=node_path,
                parent_path=parent_path,
                spec_version=spec_version,
                passed=False,
                ngff_version=ngff_version,
                issues=[
                    Issue(
                        loc=("metadata_extraction",),
                        message=str(exc),
                        severity=Severity.ERROR,
                    )
                ],
            )
        ]

    level_count = node_dict["multiscale_level_count"]
    node_dict["spec_version"] = spec_version
    node_dict["dca"] = dca

    issues: list[Issue] = []
    try:
        ModelClass.model_validate(node_dict)
    except pydantic.ValidationError as exc:
        issues.extend(_convert_errors(exc))

    if isinstance(dca, dict) and spec_version in _EXPERIMENTAL_SPEC_VERSIONS:
        issues.extend(
            validate_dca_experimental_metadata_v0_2(dca, check_ontology=check_ontology)
        )

    errors = [i for i in issues if i.severity == Severity.ERROR]
    return [
        ZarrNodeValidationResult(
            node_path=node_path,
            parent_path=parent_path,
            spec_version=spec_version,
            passed=len(errors) == 0,
            ngff_version=ngff_version,
            multiscale_level_count=level_count,
            issues=issues,
        )
    ]


def _validate_field(
    node_path: str,
    field_path: str,
    spec_version: str,
    ngff_version: str | None,
    ModelClass: "type[BaseModel]",
    *,
    check_ontology: bool = False,
) -> "tuple[ZarrNodeValidationResult, tuple | None, tuple | None, object]":
    """
    Open and validate one HCS field image.

    Returns (result, axes, level-0 chunk_shape, perturbation). axes and
    chunk_shape are None when the field could not be opened, so the caller
    can skip them in uniformity accumulation. perturbation is this field's
    raw dca.perturbation value (or None if absent or unopenable), returned
    so the caller can cross-check it against sibling fields and the parent
    well's well_perturbation row (see #154) without reopening this field.
    """
    full_path = f"{node_path}/{field_path}"
    try:
        field_zarr_group = _open_zarr_group(full_path)
        dca = dict(field_zarr_group.attrs).get("dca")
        field_img = open_ome_zarr(field_zarr_group)
        node_dict = _build_node_dict(field_img)
    except _obs_exc.BaseError as exc:
        return (
            ZarrNodeValidationResult(
                node_path=full_path,
                parent_path=node_path,
                spec_version=spec_version,
                passed=False,
                ngff_version=ngff_version,
                issues=[
                    Issue(
                        loc=("s3_io",),
                        message=f"S3 access error (network/throttling/auth): {exc}",
                        severity=Severity.ERROR,
                    )
                ],
            ),
            None,
            None,
            None,
        )
    except Exception as exc:
        return (
            ZarrNodeValidationResult(
                node_path=full_path,
                parent_path=node_path,
                spec_version=spec_version,
                passed=False,
                ngff_version=ngff_version,
                issues=[
                    Issue(
                        loc=("plate", field_path, "metadata_extraction"),
                        message=str(exc),
                        severity=Severity.ERROR,
                    )
                ],
            ),
            None,
            None,
            None,
        )

    axes = tuple(node_dict["axes"])
    chunk_shape = (
        tuple(node_dict["levels"][0]["chunk_shape"]) if node_dict["levels"] else None
    )
    level_count = node_dict["multiscale_level_count"]
    node_dict["spec_version"] = spec_version
    node_dict["dca"] = dca

    issues: list[Issue] = []
    try:
        ModelClass.model_validate(node_dict)
    except pydantic.ValidationError as exc:
        issues.extend(_convert_errors(exc))

    if isinstance(dca, dict) and spec_version in _EXPERIMENTAL_SPEC_VERSIONS:
        issues.extend(
            validate_dca_experimental_metadata_v0_2(dca, check_ontology=check_ontology)
        )

    errors = [i for i in issues if i.severity == Severity.ERROR]
    perturbation = dca.get("perturbation") if isinstance(dca, dict) else None
    return (
        ZarrNodeValidationResult(
            node_path=full_path,
            parent_path=node_path,
            spec_version=spec_version,
            passed=len(errors) == 0,
            ngff_version=ngff_version,
            multiscale_level_count=level_count,
            issues=issues,
        ),
        axes,
        chunk_shape,
        perturbation,
    )


# WellPerturbation's own attributes on top of the PerturbationAssignment it
# extends (schema/v0.2/dca_experimental_metadata.yaml WellPerturbation) — the
# (plate, well) key that resolves a row to a well group. Stripping these is
# what makes a well row comparable to an image group's placement-agnostic
# `perturbation` block.
_WELL_PERTURBATION_KEY_FIELDS: frozenset[str] = frozenset({"plate", "well"})


def _well_assignment_fields(well_perturbation: object) -> object:
    """
    Reduce a dca.well_perturbation value to the PerturbationAssignment shape
    shared with an image group's dca.perturbation block, by dropping the
    (plate, well) key. Returns the value unchanged if it isn't a dict — a
    malformed well_perturbation is already reported by
    validate_well_perturbation_v0_2, so this only needs to avoid crashing,
    not produce a second diagnosis of the same defect.
    """
    if not isinstance(well_perturbation, dict):
        return well_perturbation
    return {
        k: v
        for k, v in well_perturbation.items()
        if k not in _WELL_PERTURBATION_KEY_FIELDS
    }


def _canonicalize_perturbation(value: object) -> object:
    """
    Normalize a perturbation-like value for equality comparison by dropping
    None-valued keys from dicts, recursively.

    The perturbation schemas (PerturbationAssignment / WellPerturbation) make
    most properties optional, and two producers can express "no value for
    this optional field" either by omitting the key or by writing it
    explicitly as `null`. Those are semantically identical, but raw `!=`
    comparison would treat them as a disagreement. Canonicalizing both sides
    before comparing collapses that distinction without affecting any
    genuine difference in the data actually present.

    Only dict keys are dropped when their value is None — list entries are
    left untouched (list position is meaningful, and no field in these
    schemas is a list of nullable placeholders), and non-dict/non-list
    values, including a top-level None, pass through unchanged so a missing
    or malformed block still compares the way the rest of this function
    expects.
    """
    if isinstance(value, dict):
        return {
            k: _canonicalize_perturbation(v) for k, v in value.items() if v is not None
        }
    if isinstance(value, list):
        return [_canonicalize_perturbation(v) for v in value]
    return value


def _check_well_perturbation_agreement(
    field_paths: list[str],
    perturbation_by_field: dict[str, object],
    well_perturbation_by_well: dict[str, object],
) -> list[Issue]:
    """
    Cross-check each well's dca.well_perturbation row against the dca.perturbation
    of its own field-of-view image groups (#154).

    Per well:
      1. Every field of view under it must carry an identical `perturbation`
         value. Checked unconditionally, whether or not the well has a
         well_perturbation row at all — fields disagreeing with each other is
         always a real error (PerturbationAssignment is placement-agnostic, so
         there is no legitimate per-field variation, not even a before/after
         timepoint, which lives inside one static assignment's timepoint_*
         fields rather than as separate assignments).
      2. Only once the fields agree is there a consensus to compare the row
         against. A well with no well_perturbation row passes unconditionally
         at this step — the row is an optional, additive index over the
         image-group placement, not a second independently-required copy.

    Wells with no field of view in field_paths are not checked — there is no
    image-group data to derive a consensus from.
    """
    issues: list[Issue] = []

    fields_by_well: dict[str, list[str]] = {}
    for fp in field_paths:
        well_path = fp.rsplit("/", 1)[0]
        fields_by_well.setdefault(well_path, []).append(fp)

    for well_path, fps in fields_by_well.items():
        reference_fp = fps[0]
        consensus = perturbation_by_field.get(reference_fp)
        canonical_consensus = _canonicalize_perturbation(consensus)
        disagreeing = [
            fp
            for fp in fps[1:]
            if _canonicalize_perturbation(perturbation_by_field.get(fp))
            != canonical_consensus
        ]
        if disagreeing:
            example_fp = disagreeing[0]
            issues.append(
                Issue(
                    loc=("plate", "wells", well_path, "dca", "perturbation"),
                    message=(
                        f"dca.perturbation is not identical across all field(s) of "
                        f"view in well '{well_path}' ({len(disagreeing)} of "
                        f"{len(fps) - 1} disagree with '{reference_fp}'): "
                        f"'{reference_fp}' has {consensus!r}, '{example_fp}' has "
                        f"{perturbation_by_field.get(example_fp)!r}. All fields of "
                        f"view under one well must carry the identical perturbation "
                        f"assignment."
                    ),
                    severity=Severity.ERROR,
                )
            )
            continue

        well_perturbation = well_perturbation_by_well.get(well_path)
        if well_perturbation is None:
            continue

        well_assignment = _well_assignment_fields(well_perturbation)
        if _canonicalize_perturbation(well_assignment) != canonical_consensus:
            issues.append(
                Issue(
                    loc=("plate", "wells", well_path, "dca", "well_perturbation"),
                    message=(
                        f"dca.well_perturbation disagrees with the dca.perturbation "
                        f"consensus derived from the field(s) of view in well "
                        f"'{well_path}': well_perturbation has {well_assignment!r}, "
                        f"image groups have {consensus!r}. The image-group block is "
                        f"authoritative; re-derive this well row from it, never the "
                        f"reverse."
                    ),
                    severity=Severity.ERROR,
                )
            )

    return issues


def _validate_hcs_plate(
    node_path: str,
    raw_plate_attrs: dict,
    ngff_version: str | None,
    spec_version: str,
    ModelClass: "type[BaseModel]",
    field_progress_hook: "Callable[[list[str]], dict[str, Callable[[], None]]] | None" = None,
    check_ontology: bool = False,
) -> list[ZarrNodeValidationResult]:
    """
    Run plate-level structural checks and per-field DCA validation.

    Returns one result for the plate root (structural issues) plus one
    result per field image. Fields are validated concurrently via a thread
    pool — each field's S3 I/O is independent.

    Takes raw plate OME attrs (the dict under ome.plate) rather than an
    ome-zarr-models HCS object to avoid HCS.from_zarr(), which recursively
    opens all well and field sub-groups at construction time.

    field_progress_hook: called with the list of field paths once they are
    known (after _gather_plate_fields). Returns a dict mapping each field
    path to an advance callable invoked when that field completes. All
    callables must be thread-safe.
    """
    schema_issues = _check_plate_schema(raw_plate_attrs)
    gather_issues, field_paths, well_perturbation_by_well = _gather_plate_fields(
        node_path,
        raw_plate_attrs.get("wells", []),
        raw_plate_attrs.get("acquisitions"),
        spec_version=spec_version,
    )
    plate_issues = schema_issues + gather_issues

    field_advance_fns: dict[str, Callable[[], None]] = {}
    if field_progress_hook is not None:
        field_advance_fns = field_progress_hook(field_paths)

    seen_axes: list[tuple] = []
    seen_chunks: list[tuple] = []
    field_results: list[ZarrNodeValidationResult] = []
    perturbation_by_field: dict[str, object] = {}

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                _validate_field,
                node_path,
                fp,
                spec_version,
                ngff_version,
                ModelClass,
                check_ontology=check_ontology,
            ): fp
            for fp in field_paths
        }
        for future in as_completed(futures):
            fp = futures[future]
            result, axes, chunk_shape, perturbation = future.result()
            field_results.append(result)
            perturbation_by_field[fp] = perturbation
            if axes is not None:
                seen_axes.append(axes)
            if chunk_shape is not None:
                seen_chunks.append(chunk_shape)
            if fp in field_advance_fns:
                field_advance_fns[fp]()

    if spec_version in _EXPERIMENTAL_SPEC_VERSIONS:
        plate_issues.extend(
            _check_well_perturbation_agreement(
                field_paths, perturbation_by_field, well_perturbation_by_well
            )
        )

    if len(set(seen_axes)) > 1:
        plate_issues.append(
            Issue(
                loc=("plate", "axes"),
                message=(
                    f"Axes are not uniform across all plate fields: "
                    f"{set(seen_axes)}"
                ),
                severity=Severity.WARNING,
            )
        )

    if len(set(seen_chunks)) > 1:
        plate_issues.append(
            Issue(
                loc=("plate", "chunk_shape"),
                message=(
                    f"Level-0 chunk shapes are not uniform across all "
                    f"plate fields: {set(seen_chunks)}"
                ),
                severity=Severity.WARNING,
            )
        )

    errors = [i for i in plate_issues if i.severity == Severity.ERROR]
    plate_result = ZarrNodeValidationResult(
        node_path=node_path,
        spec_version=spec_version,
        passed=len(errors) == 0,
        ngff_version=ngff_version,
        issues=plate_issues,
    )

    return [plate_result] + field_results


def _validate_labels_list(
    node_path: str,
    ngff_version: str | None,
    spec_version: str,
) -> list[ZarrNodeValidationResult]:
    """LABELS_LIST groups carry no arrays — pass immediately."""
    return [
        ZarrNodeValidationResult(
            node_path=node_path,
            spec_version=spec_version,
            passed=True,
            ngff_version=ngff_version,
        )
    ]


# ---------------------------------------------------------------------------
# Core per-store validation — dispatcher
# ---------------------------------------------------------------------------


def validate_zarr_node(
    node_path: str,
    spec_version: str = CURRENT_SPEC_VERSION,
    ModelClass: "type[BaseModel] | None" = None,
    field_progress_hook: "Callable[[list[str]], dict[str, Callable[[], None]]] | None" = None,
    check_ontology: bool = False,
) -> list[ZarrNodeValidationResult]:
    """
    Run full validation (NGFF structural + DCA spec) for one store.

    Returns one result for standalone image stores, or one result per field
    image for HCS plates.

    Parameters
    ----------
    node_path:   local path or s3:// URL to the Zarr store root
    spec_version: DCA spec version string, e.g. "0.1"
    ModelClass:   the DCAStoreSpec class for this version; resolved from the
                  registry if not provided
    """
    if ModelClass is None:
        ModelClass = get_model(spec_version)

    # Step 1: open group and read raw ome attrs before ome-zarr-models dispatch.
    # This is required because ome-zarr-models uses extra="allow" on BaseOMEAttrs
    # and would silently accept a group with both 'plate' and 'multiscales'.
    try:
        zarr_group = _open_zarr_group(node_path)
        raw_attrs = dict(zarr_group.attrs)
        raw_ome_attrs = raw_attrs.get("ome", {})
        raw_dca_attrs = raw_attrs.get("dca")
    except _obs_exc.BaseError as exc:
        return [
            ZarrNodeValidationResult(
                node_path=node_path,
                spec_version=spec_version,
                passed=False,
                issues=[
                    Issue(
                        loc=("s3_io",),
                        message=f"S3 access error (network/throttling/auth): {exc}",
                        severity=Severity.ERROR,
                    )
                ],
            )
        ]
    except Exception as exc:
        return [
            ZarrNodeValidationResult(
                node_path=node_path,
                spec_version=spec_version,
                passed=False,
                issues=[
                    Issue(
                        loc=("store_open",), message=str(exc), severity=Severity.ERROR
                    )
                ],
            )
        ]

    # Step 1a: GEFF skip — GEFF tracking stores are not OME-Zarr.
    # Detect by the presence of a top-level "geff" key in root attrs.
    if "geff" in raw_attrs:
        return [
            ZarrNodeValidationResult(
                node_path=node_path,
                spec_version=spec_version,
                passed=True,
                skipped=True,
                node_type=ZarrNodeType.GEFF,
            )
        ]

    # Step 1b: HCS fast path — bypass open_ome_zarr for plates.
    # open_ome_zarr on an HCS root calls HCS.from_zarr(), which recursively opens
    # every well via Well.from_zarr(), which in turn opens every field image via
    # Image.from_zarr() — O(N_fields) sequential S3 calls before any validation.
    # Detect HCS from raw attrs and validate directly from the plate metadata dict.
    if "plate" in raw_ome_attrs and "multiscales" not in raw_ome_attrs:
        ngff_version = str(raw_ome_attrs.get("version", "0.5"))
        results = _validate_hcs_plate(
            node_path,
            raw_ome_attrs.get("plate", {}),
            ngff_version,
            spec_version,
            ModelClass,
            field_progress_hook=field_progress_hook,
            check_ontology=check_ontology,
        )
        for r in results:
            if r.node_type is None:
                r.node_type = ZarrNodeType.HCS_PLATE
        return results

    # Step 2: OME NGFF structural validation (non-HCS stores).
    try:
        ome_obj = open_ome_zarr(zarr_group)
    except Exception as exc:
        return [
            ZarrNodeValidationResult(
                node_path=node_path,
                spec_version=spec_version,
                passed=False,
                issues=[
                    Issue(
                        loc=("ngff_validation",),
                        message=str(exc),
                        severity=Severity.ERROR,
                    )
                ],
            )
        ]

    ngff_version = str(ome_obj.ome_zarr_version)

    # Step 3: classify node type, catching ambiguous ome attrs.
    try:
        node_type = classify_group(node_path, ome_obj, raw_ome_attrs)
    except ValueError as exc:
        return [
            ZarrNodeValidationResult(
                node_path=node_path,
                spec_version=spec_version,
                passed=False,
                ngff_version=ngff_version,
                issues=[
                    Issue(
                        loc=("node_classification",),
                        message=str(exc),
                        severity=Severity.ERROR,
                    )
                ],
            )
        ]

    # Step 4: dispatch by node type.
    if node_type == ZarrNodeType.LABELS_LIST:
        results = _validate_labels_list(node_path, ngff_version, spec_version)
        for r in results:
            if r.node_type is None:
                r.node_type = node_type
        return results

    if node_type == ZarrNodeType.IMAGE_LABEL:
        try:
            LabelModelClass = get_label_model(spec_version)
        except UnsupportedSpecVersionError:
            # DCALabelStoreSpec not yet registered for this version — skip with warning.
            results = [
                ZarrNodeValidationResult(
                    node_path=node_path,
                    spec_version=spec_version,
                    passed=True,
                    ngff_version=ngff_version,
                    issues=[
                        Issue(
                            loc=("label_validation",),
                            message=(
                                f"DCA label group validation is not yet supported for "
                                f"spec version '{spec_version}'. "
                                f"Label arrays were not checked."
                            ),
                            severity=Severity.WARNING,
                        )
                    ],
                )
            ]
        else:
            results = _validate_image(
                node_path,
                ome_obj,
                ngff_version,
                spec_version,
                LabelModelClass,
                dca=None,
                check_ontology=check_ontology,
            )
        for r in results:
            if r.node_type is None:
                r.node_type = node_type
        return results

    # IMAGE
    results = _validate_image(
        node_path,
        ome_obj,
        ngff_version,
        spec_version,
        ModelClass,
        dca=raw_dca_attrs,
        check_ontology=check_ontology,
    )
    for r in results:
        if r.node_type is None:
            r.node_type = node_type
    return results
