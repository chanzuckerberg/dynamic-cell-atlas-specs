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
from dca_helpers.validation.discovery import open_zarr_group, well_of_field
from dca_helpers.validation.registry import (
    UnsupportedSpecVersionError,
    get_label_model,
    get_model,
)
from dca_helpers.validation.result import (
    Issue,
    PlateSubsetInfo,
    Severity,
    ZarrNodeValidationResult,
)
from dca_helpers.validation.versions.v0_2.experimental_metadata import (
    validate_dca_experimental_metadata_v0_2,
    validate_well_perturbation_v0_2,
)
from dca_helpers.validation.zarr_node import ZarrNodeType, classify_group

if TYPE_CHECKING:
    import zarr
    from pydantic import BaseModel
    from pydantic_zarr.v3 import ArraySpec


# ---------------------------------------------------------------------------
# OME/DCA attrs coercion and extraction helpers
#
# These interpret the *contents* of OME/DCA attrs — a validation-domain
# concern, distinct from the store-location/opening mechanics discovery.py
# owns.
# ---------------------------------------------------------------------------


def _as_dict(value: object) -> dict:
    """Coerce a malformed zarr attrs value to a dict — absent, null, or
    non-dict values all become {}."""
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list:
    """Coerce a malformed zarr attrs value to a list — absent, null, or
    non-list values all become []."""
    return value if isinstance(value, list) else []


def _entry_path(entry: object) -> str:
    """Extract an entry's declared "path" as a string — used for both
    plate.wells[] and well.images[] entries. Non-dict, missing, or
    non-string "path" becomes "" rather than raising."""
    if not isinstance(entry, dict):
        return ""
    path = entry.get("path", "")
    return path if isinstance(path, str) else ""


def is_hcs_plate(raw_ome_attrs: dict) -> bool:
    """
    True if attrs declare an HCS plate: a dict "plate" key with no sibling
    "multiscales" (ambiguous otherwise).

    The isinstance check matters, not just presence — a malformed non-dict
    "plate" must not take the HCS fast path, or it would pass silently
    instead of failing structural validation.
    """
    return isinstance(raw_ome_attrs.get("plate"), dict) and (
        "multiscales" not in raw_ome_attrs
    )


def declared_well_paths(raw_ome_attrs: dict) -> "frozenset[str]":
    """Every plate.wells[] entry's declared "path"; entries with no usable
    path are skipped (see _entry_path)."""
    return frozenset(
        path
        for w in _as_list(_as_dict(raw_ome_attrs.get("plate")).get("wells"))
        if (path := _entry_path(w))
    )


def well_images(well_ome: dict) -> list:
    """A well's declared field-image list (ome.well.images) from its "ome"
    attrs dict; a missing/malformed value becomes []."""
    images = _as_dict(well_ome.get("well")).get("images", [])
    return images if isinstance(images, list) else []


def ome_attrs_of(zarr_group: "zarr.Group") -> dict:
    """The "ome" sub-dict of an already-opened zarr group's attrs, coerced
    defensively (see _as_dict)."""
    return _as_dict(dict(zarr_group.attrs).get("ome"))


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
        "axis_scales": _level0_axis_scales(ms),
        "multiscale_level_count": len(ms.datasets),
        "levels": levels,
        "array_shapes": shapes,
    }


def _level0_axis_scales(ms: object) -> dict[str, float | None]:
    """Level 0's ``scale`` transform, keyed by axis name.

    Store parsing only — no rule lives here. A version's model decides what, if
    anything, a scale means; ``v0_1`` ignores the key entirely (every model is
    ``extra="allow"``). Any axis whose scale cannot be read is reported as ``None``
    rather than defaulted, so a model can tell "absent" from "1.0".
    """
    axis_names = [ax.name for ax in ms.axes]  # type: ignore[attr-defined]
    scales: dict[str, float | None] = dict.fromkeys(axis_names)
    datasets = ms.datasets  # type: ignore[attr-defined]
    if not datasets:
        return scales
    for transform in getattr(datasets[0], "coordinateTransformations", ()) or ():
        values = getattr(transform, "scale", None)
        if values is None or len(values) != len(axis_names):
            continue
        for name, value in zip(axis_names, values):
            try:
                scales[name] = float(value)
            except (TypeError, ValueError):
                scales[name] = None
    return scales


# ---------------------------------------------------------------------------
# HCS plate traversal
# ---------------------------------------------------------------------------

_ALPHANUMERIC = re.compile(r"^[a-zA-Z0-9]+$")
# OME-NGFF v0.5 well groups may contain a 'labels' sub-group alongside field
# images. Exclude it (and any future spec-defined non-field keys) from the
# orphan check so 'labels' is never reported as an undeclared field.
_WELL_NON_FIELD_KEYS: frozenset[str] = frozenset({"labels"})


def _validate_names(
    entries: list, plate_key: str, kind: str
) -> "tuple[list[str], list[Issue]]":
    """
    Validate a plate.rows/columns entry list for alphanumeric, unique names;
    returns (names, issues). Shared by _check_plate_schema for both, with
    plate_key/kind distinguishing them in loc/message text.
    """
    issues: list[Issue] = []
    names: list[str] = []
    for entry in entries:
        name = str(entry.get("name", "")) if isinstance(entry, dict) else ""
        names.append(name)
        if not _ALPHANUMERIC.match(name):
            issues.append(
                Issue(
                    loc=("plate", plate_key, name),
                    message=f"{kind} name {name!r} must be alphanumeric ([a-zA-Z0-9])",
                    severity=Severity.ERROR,
                )
            )
    seen: set[str] = set()
    for name in names:
        if name in seen:
            issues.append(
                Issue(
                    loc=("plate", plate_key),
                    message=f"Duplicate {kind.lower()} name {name!r} in plate.{plate_key}",
                    severity=Severity.ERROR,
                )
            )
        seen.add(name)
    return names, issues


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
    # rows/columns — alphanumeric and unique
    row_names, row_issues = _validate_names(
        _as_list(raw_plate_attrs.get("rows")), "rows", "Row"
    )
    col_names, col_issues = _validate_names(
        _as_list(raw_plate_attrs.get("columns")), "columns", "Column"
    )
    issues: list[Issue] = row_issues + col_issues

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
    for aq_idx, aq in enumerate(_as_list(raw_plate_attrs.get("acquisitions"))):
        if not isinstance(aq, dict) or "id" not in aq:
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
    for well in _as_list(raw_plate_attrs.get("wells")):
        well_path = _entry_path(well)

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


_WELL_OPEN_FAILED = object()  # sentinel: distinguishes "well couldn't be
# opened" from a legitimate well_perturbation of None


def _well_s3_issue(well_path: str, action: str, exc: Exception) -> Issue:
    return Issue(
        loc=("s3_io",),
        message=(
            f"S3 access error (network/throttling/auth) {action} "
            f"well '{well_path}': {exc}"
        ),
        severity=Severity.ERROR,
    )


def _process_one_well(
    node_path: str,
    well_path: str,
    valid_aq_ids: "set[int] | None",
    check_well_perturbation: bool,
    field_offset: int,
    field_slice_end: "int | None",
) -> "tuple[str, list[Issue], list[str], list[str], object]":
    """
    Open one well and run the per-well MUST checks documented on
    _gather_plate_fields. Returns (well_path, issues, well_field_paths,
    selected_field_paths, well_perturbation_or_sentinel) — the last element
    is _WELL_OPEN_FAILED when the well group couldn't be opened or its
    children couldn't be listed over S3, so the caller adds no
    well_perturbation_by_well entry for it.

    Pure function of its arguments (no shared mutable state) so it can run
    concurrently across wells via a thread pool — see _gather_plate_fields.
    """
    try:
        well_group = open_zarr_group(f"{node_path}/{well_path}")
    except _obs_exc.BaseError as exc:
        return (
            well_path,
            [_well_s3_issue(well_path, "opening", exc)],
            [],
            [],
            _WELL_OPEN_FAILED,
        )
    except Exception:
        return (
            well_path,
            [
                Issue(
                    loc=("plate", "wells", well_path),
                    message=(
                        f"Well '{well_path}' is declared in "
                        f"ome.plate.wells but has no group on disk"
                    ),
                    severity=Severity.ERROR,
                )
            ],
            [],
            [],
            _WELL_OPEN_FAILED,
        )

    issues: list[Issue] = []
    well_attrs = dict(well_group.attrs)
    well_ome = _as_dict(well_attrs.get("ome"))
    field_images = well_images(well_ome)
    try:
        well_keys = set(well_group.keys())
    except _obs_exc.BaseError as exc:
        return (
            well_path,
            [_well_s3_issue(well_path, "listing", exc)],
            [],
            [],
            _WELL_OPEN_FAILED,
        )
    except Exception:
        well_keys = set()

    well_dca = well_attrs.get("dca")
    well_perturbation = (
        well_dca.get("well_perturbation") if isinstance(well_dca, dict) else None
    )
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
    well_field_paths: list[str] = []
    for img_meta in field_images:
        img_path = _entry_path(img_meta)
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
            well_field_paths.append(f"{well_path}/{img_path}")
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

    selected_field_paths = well_field_paths[field_offset:field_slice_end]
    return well_path, issues, well_field_paths, selected_field_paths, well_perturbation


def _gather_plate_fields(
    node_path: str,
    plate_wells: list[dict],
    plate_acquisitions: "list[dict] | None" = None,
    *,
    spec_version: str,
    well_filter: "frozenset[str] | None" = None,
    field_offset: int = 0,
    field_limit: "int | None" = None,
) -> "tuple[list[Issue], list[str], list[str], dict[str, object]]":
    """
    Open each in-scope declared well zarr.json once and return:
      - issues for wells / fields that fail per-well traversal checks
      - field paths for all declared fields that exist on disk, across every
        opened well (independent of field_offset/field_limit)
      - a per-well offset/limit slice of the above (see field_offset/field_limit
        below); identical to the previous return value when both are default
      - a well_path -> well_perturbation mapping (raw attrs value, or None
        when absent) for every well opened, so the caller can cross-check
        each well's row against its field-of-view consensus after the field
        loop runs, without reopening well groups.

    Handles only checks that require opening well groups on disk.
    Plate-level and well-path-level schema checks (row/column names,
    well path syntax, acquisition id format, fieldCount) are the
    responsibility of _check_plate_schema, which must be called first.

    Uses raw zarr attrs rather than ome-zarr-models Well objects to avoid
    Well.from_zarr(), which eagerly opens all Image sub-groups via
    get_optional_group_paths() — O(N_fields) sequential S3 calls.

    Wells are opened concurrently via a thread pool (each well's own S3
    round trip is independent) — for a plate with many wells this is the
    dominant cost of even a structural-only pass, distinct from the
    per-well field-count problem.

    well_filter: when given, any well whose path is not in this
        set is skipped entirely — never opened, no issues raised for it, not
        counted as missing. All checks below still run in full for every well
        that *is* opened (selected wells, or all wells when well_filter is
        None) — this parameter only controls which wells are opened, not how
        thoroughly an opened well is checked.
    field_offset, field_limit: after building one opened well's
        on-disk field list in the order fields are declared in that well's
        ome.well.images, apply the ordinary Python slice
        ``declared[field_offset : field_offset + field_limit]`` (or
        ``declared[field_offset:]`` when field_limit is None) independently
        per well — not globally across wells. The slice is only used to
        build the 3rd return value (selected_field_paths); the 2nd return
        value (field_paths) always contains every on-disk field regardless
        of these params, so orphan/duplicate/missing-group/acquisition-id
        checks below are never affected by slicing. An out-of-range offset or
        an empty ome.well.images list simply yields an empty slice for that
        well — not an error.

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
    selected_field_paths: list[str] = []
    well_perturbation_by_well: dict[str, object] = {}
    check_well_perturbation = spec_version in _EXPERIMENTAL_SPEC_VERSIONS
    field_slice_end = None if field_limit is None else field_offset + field_limit

    valid_aq_ids: set[int] | None = None
    if plate_acquisitions is not None:
        valid_aq_ids = {
            aq["id"]
            for aq in plate_acquisitions
            if isinstance(aq, dict) and isinstance(aq.get("id"), int) and aq["id"] >= 0
        }

    well_paths_to_process: list[str] = []
    for well_meta in plate_wells:
        well_path = _entry_path(well_meta)
        if not well_path:
            # Malformed entry (missing/empty path) — _check_plate_schema
            # already reports this via its own well-path syntax check;
            # nothing to open here.
            continue
        if well_filter is not None and well_path not in well_filter:
            continue
        well_paths_to_process.append(well_path)

    # executor.map (not submit + as_completed) so results merge back in
    # well-declaration order regardless of which well's S3 round trip
    # finishes first — callers rely on field_paths/selected_field_paths
    # being in a deterministic order.
    with ThreadPoolExecutor() as executor:
        for (
            well_path,
            well_issues,
            well_field_paths,
            well_selected,
            perturbation,
        ) in executor.map(
            lambda wp: _process_one_well(
                node_path,
                wp,
                valid_aq_ids,
                check_well_perturbation,
                field_offset,
                field_slice_end,
            ),
            well_paths_to_process,
        ):
            issues.extend(well_issues)
            field_paths.extend(well_field_paths)
            selected_field_paths.extend(well_selected)
            if perturbation is not _WELL_OPEN_FAILED:
                well_perturbation_by_well[well_path] = perturbation

    return issues, field_paths, selected_field_paths, well_perturbation_by_well


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


def _single_error_result(
    node_path: str,
    spec_version: str,
    loc: tuple,
    message: str,
    *,
    ngff_version: str | None = None,
    parent_path: str | None = None,
    node_type: "ZarrNodeType | None" = None,
) -> ZarrNodeValidationResult:
    """Build a failed ZarrNodeValidationResult with one ERROR issue — the
    shape behind most early-exit returns below."""
    return ZarrNodeValidationResult(
        node_path=node_path,
        parent_path=parent_path,
        spec_version=spec_version,
        passed=False,
        ngff_version=ngff_version,
        issues=[Issue(loc=loc, message=message, severity=Severity.ERROR)],
        node_type=node_type,
    )


def _run_dca_validation(
    node_dict: dict,
    dca: object,
    spec_version: str,
    ModelClass: "type[BaseModel]",
    check_ontology: bool,
) -> list[Issue]:
    """Validate node_dict against ModelClass, plus dca's experimental-metadata
    block for experimental spec versions. Shared by _validate_image and
    _validate_field, which only differ in how node_dict is built."""
    issues: list[Issue] = []
    try:
        ModelClass.model_validate(node_dict)
    except pydantic.ValidationError as exc:
        issues.extend(_convert_errors(exc))

    if isinstance(dca, dict) and spec_version in _EXPERIMENTAL_SPEC_VERSIONS:
        issues.extend(
            validate_dca_experimental_metadata_v0_2(dca, check_ontology=check_ontology)
        )

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
            _single_error_result(
                node_path,
                spec_version,
                ("metadata_extraction",),
                str(exc),
                ngff_version=ngff_version,
                parent_path=parent_path,
            )
        ]

    level_count = node_dict["multiscale_level_count"]
    node_dict["spec_version"] = spec_version
    node_dict["dca"] = dca

    issues = _run_dca_validation(
        node_dict, dca, spec_version, ModelClass, check_ontology
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
    well's well_perturbation row without reopening this field.
    """
    full_path = f"{node_path}/{field_path}"
    try:
        field_zarr_group = open_zarr_group(full_path)
        dca = dict(field_zarr_group.attrs).get("dca")
        field_img = open_ome_zarr(field_zarr_group)
        node_dict = _build_node_dict(field_img)
    except _obs_exc.BaseError as exc:
        return (
            _single_error_result(
                full_path,
                spec_version,
                ("s3_io",),
                f"S3 access error (network/throttling/auth): {exc}",
                ngff_version=ngff_version,
                parent_path=node_path,
                node_type=ZarrNodeType.IMAGE,
            ),
            None,
            None,
            None,
        )
    except Exception as exc:
        return (
            _single_error_result(
                full_path,
                spec_version,
                ("plate", field_path, "metadata_extraction"),
                str(exc),
                ngff_version=ngff_version,
                parent_path=node_path,
                node_type=ZarrNodeType.IMAGE,
            ),
            None,
            None,
            None,
        )

    axes = tuple(ax.lower() for ax in node_dict["axes"])
    chunk_shape = (
        tuple(node_dict["levels"][0]["chunk_shape"]) if node_dict["levels"] else None
    )
    level_count = node_dict["multiscale_level_count"]
    node_dict["spec_version"] = spec_version
    node_dict["dca"] = dca

    issues = _run_dca_validation(
        node_dict, dca, spec_version, ModelClass, check_ontology
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
            node_type=ZarrNodeType.IMAGE,
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
    of its own field-of-view image groups.

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
        fields_by_well.setdefault(well_of_field(fp), []).append(fp)

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
    well_filter: "frozenset[str] | None" = None,
    field_offset: int = 0,
    field_limit: "int | None" = None,
) -> list[ZarrNodeValidationResult]:
    """
    Run plate-level structural checks and per-field DCA validation.

    Returns one result for the plate root (structural issues) plus one
    result per selected field image. Fields are validated concurrently via a
    thread pool — each field's S3 I/O is independent.

    Takes raw plate OME attrs (the dict under ome.plate) rather than an
    ome-zarr-models HCS object to avoid HCS.from_zarr(), which recursively
    opens all well and field sub-groups at construction time.

    field_progress_hook: called with the list of *selected* field paths
    (post well_filter/field_offset/field_limit) once they are known (after
    _gather_plate_fields) — this may be a strict subset of the plate's full
    declared field list when subsetting is in effect. Returns a dict mapping
    each field path to an advance callable invoked when that field
    completes. All callables must be thread-safe.

    well_filter, field_offset, field_limit: forwarded to _gather_plate_fields
    to scope this run to a subset of the plate's wells/fields.
    Only the selected fields go through _validate_field's expensive S3 I/O;
    every opened well's full declared field list still gets the cheap
    metadata-only structural checks inside _gather_plate_fields regardless of
    slicing. When any of the three is non-default, the returned plate-root
    result carries a PlateSubsetInfo recording wells/fields declared vs.
    selected, so a partial run is never mistaken for full coverage.
    """
    schema_issues = _check_plate_schema(raw_plate_attrs)
    declared_wells = _as_list(raw_plate_attrs.get("wells"))
    gather_issues, field_paths, selected_field_paths, well_perturbation_by_well = (
        _gather_plate_fields(
            node_path,
            declared_wells,
            raw_plate_attrs.get("acquisitions"),
            spec_version=spec_version,
            well_filter=well_filter,
            field_offset=field_offset,
            field_limit=field_limit,
        )
    )
    plate_issues = schema_issues + gather_issues

    field_advance_fns: dict[str, Callable[[], None]] = {}
    if field_progress_hook is not None:
        field_advance_fns = field_progress_hook(selected_field_paths)

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
            for fp in selected_field_paths
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
                selected_field_paths, perturbation_by_field, well_perturbation_by_well
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

    subset_coverage = None
    if well_filter is not None or field_offset or field_limit is not None:
        # Excludes malformed entries (missing/blank "path") — they were never
        # openable to begin with, so counting them here would make
        # wells_selected < wells_total (and thus PARTIAL) even when every
        # valid well was selected.
        wells_total = sum(1 for w in declared_wells if _entry_path(w))
        # wells actually opened (well_perturbation_by_well is only populated
        # after a successful open — see _gather_plate_fields), not merely
        # requested, so a well_filter entry that fails to open isn't
        # double-counted as "selected".
        wells_selected = len(well_perturbation_by_well)
        fields_total = len(field_paths)
        fields_selected = len(selected_field_paths)
        # Only a genuine narrowing counts as partial — e.g. --wells naming
        # every well the plate has, or a --field-limit no field count
        # reaches, selects everything and isn't actually a subset.
        if wells_selected < wells_total or fields_selected < fields_total:
            subset_coverage = PlateSubsetInfo(
                wells_total=wells_total,
                wells_selected=wells_selected,
                fields_total=fields_total,
                fields_selected=fields_selected,
            )

    errors = [i for i in plate_issues if i.severity == Severity.ERROR]
    plate_result = ZarrNodeValidationResult(
        node_path=node_path,
        spec_version=spec_version,
        passed=len(errors) == 0,
        ngff_version=ngff_version,
        issues=plate_issues,
        subset_coverage=subset_coverage,
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


def _tag_node_type(
    results: list[ZarrNodeValidationResult], node_type: ZarrNodeType
) -> list[ZarrNodeValidationResult]:
    """Set node_type on every result that doesn't already carry one, then
    return results — the common tail of each dispatch branch below."""
    for r in results:
        if r.node_type is None:
            r.node_type = node_type
    return results


def validate_zarr_node(
    node_path: str,
    spec_version: str = CURRENT_SPEC_VERSION,
    ModelClass: "type[BaseModel] | None" = None,
    field_progress_hook: "Callable[[list[str]], dict[str, Callable[[], None]]] | None" = None,
    check_ontology: bool = False,
    wells: "frozenset[str] | None" = None,
    field_offset: int = 0,
    field_limit: "int | None" = None,
) -> list[ZarrNodeValidationResult]:
    """
    Run full validation (NGFF structural + DCA spec) for one store.

    Returns one result for standalone image stores, or one result per
    selected field image for HCS plates.

    Parameters
    ----------
    node_path:   local path or s3:// URL to the Zarr store root
    spec_version: DCA spec version string, e.g. "0.1"
    ModelClass:   the DCAStoreSpec class for this version; resolved from the
                  registry if not provided
    wells, field_offset, field_limit: well/field-subset selection for HCS
                  plates, forwarded to _validate_hcs_plate. Any of
                  these being non-default on a non-HCS-plate node, or `wells`
                  naming a well the plate doesn't declare, is reported as a
                  single ERROR result rather than silently ignored.
    """
    if ModelClass is None:
        ModelClass = get_model(spec_version)

    # Step 1: open group and read raw ome attrs before ome-zarr-models dispatch.
    # This is required because ome-zarr-models uses extra="allow" on BaseOMEAttrs
    # and would silently accept a group with both 'plate' and 'multiscales'.
    try:
        zarr_group = open_zarr_group(node_path)
        raw_attrs = dict(zarr_group.attrs)
        raw_ome_attrs = _as_dict(raw_attrs.get("ome"))
        raw_dca_attrs = raw_attrs.get("dca")
    except _obs_exc.BaseError as exc:
        return [
            _single_error_result(
                node_path,
                spec_version,
                ("s3_io",),
                f"S3 access error (network/throttling/auth): {exc}",
            )
        ]
    except Exception as exc:
        return [
            _single_error_result(node_path, spec_version, ("store_open",), str(exc))
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

    # Step 1a-bis: well/field subset flags only apply to HCS plates.
    # Checked here — before step 1b — so a subset request against a non-plate
    # node, or naming an undeclared well, is reported as a clean ERROR result
    # instead of silently validating the whole store or crashing later.
    hcs_plate = is_hcs_plate(raw_ome_attrs)
    if wells is not None or field_offset or field_limit is not None:
        if field_offset < 0 or (field_limit is not None and field_limit < 0):
            return [
                _single_error_result(
                    node_path,
                    spec_version,
                    ("subset",),
                    "field_offset and field_limit must be >= 0",
                )
            ]
        if not hcs_plate:
            return [
                _single_error_result(
                    node_path,
                    spec_version,
                    ("subset",),
                    "well/field subset options (--wells, --field-offset, "
                    "--field-limit) only apply to HCS plates",
                )
            ]
        if wells is not None:
            # A malformed well entry (missing/blank "path") contributes
            # nothing here — it's not a real well name to compare --wells
            # against, and including "" would make the "declared wells"
            # list in the error message below confusing.
            declared = declared_well_paths(raw_ome_attrs)
            unknown = sorted(wells - declared)
            if unknown:
                return [
                    _single_error_result(
                        node_path,
                        spec_version,
                        ("subset", "wells"),
                        f"Unknown well(s) in --wells: {unknown}; "
                        f"declared wells: {sorted(declared)}",
                    )
                ]

    # Step 1b: HCS fast path — bypass open_ome_zarr for plates.
    # open_ome_zarr on an HCS root calls HCS.from_zarr(), which recursively opens
    # every well via Well.from_zarr(), which in turn opens every field image via
    # Image.from_zarr() — O(N_fields) sequential S3 calls before any validation.
    # Detect HCS from raw attrs and validate directly from the plate metadata dict.
    if hcs_plate:
        ngff_version = str(raw_ome_attrs.get("version", "0.5"))
        results = _validate_hcs_plate(
            node_path,
            _as_dict(raw_ome_attrs.get("plate")),
            ngff_version,
            spec_version,
            ModelClass,
            field_progress_hook=field_progress_hook,
            check_ontology=check_ontology,
            well_filter=wells,
            field_offset=field_offset,
            field_limit=field_limit,
        )
        return _tag_node_type(results, ZarrNodeType.HCS_PLATE)

    # Step 2: OME NGFF structural validation (non-HCS stores).
    try:
        ome_obj = open_ome_zarr(zarr_group)
    except Exception as exc:
        return [
            _single_error_result(
                node_path, spec_version, ("ngff_validation",), str(exc)
            )
        ]

    ngff_version = str(ome_obj.ome_zarr_version)

    # Step 3: classify node type, catching ambiguous ome attrs.
    try:
        node_type = classify_group(node_path, ome_obj, raw_ome_attrs)
    except ValueError as exc:
        return [
            _single_error_result(
                node_path,
                spec_version,
                ("node_classification",),
                str(exc),
                ngff_version=ngff_version,
            )
        ]

    # Step 4: dispatch by node type.
    if node_type == ZarrNodeType.LABELS_LIST:
        results = _validate_labels_list(node_path, ngff_version, spec_version)
        return _tag_node_type(results, node_type)

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
        return _tag_node_type(results, node_type)

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
    return _tag_node_type(results, node_type)
