"""
Top-level runner: discover stores, run validation, return results.

Usage
-----
from dca_helpers.validation.runner import validate

# Validate a single store directly (fast — no discovery):
results = validate("s3://my-bucket/datasets/plate.ome.zarr")

# Validate all stores found under a path (recursive discovery):
results = validate("s3://my-bucket/datasets/")

for r in results:
    print(r.node_path, "PASS" if r.passed else "FAIL")
    for issue in r.issues:
        print(f"  [{issue.severity.value.upper()}] {issue.message}")
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Literal

from rich.live import Live

from dca_helpers import CURRENT_SPEC_VERSION, __version__
from dca_helpers.validation.discovery import (
    discover_zarr_stores,
    is_zarr_store,
    open_zarr_group,
)
from dca_helpers.validation.progress_tree import _TreeDisplay
from dca_helpers.validation.registry import get_model
from dca_helpers.validation.result import (
    ValidationRun,
    ValidationSummary,
    ZarrNodeValidationResult,
)
from dca_helpers.validation.validator import (
    declared_well_paths,
    is_hcs_plate,
    ome_attrs_of,
    validate_zarr_node,
    well_images,
)

_DEFAULT_SPEC_VERSION = CURRENT_SPEC_VERSION
logger = logging.getLogger(__name__)


def _parse_wells(wells: str) -> "frozenset[str]":
    """Parse a --wells CLI/API value into a well-path set.

    Comma-split, strip whitespace, drop empty tokens, dedupe (duplicate
    entries are harmless/idempotent). Raises ValueError if nothing survives
    (e.g. an empty or all-whitespace string) — an empty selection is always
    a usage error, never "validate zero wells".
    """
    parsed = frozenset(w.strip() for w in wells.split(",") if w.strip())
    if not parsed:
        raise ValueError("wells must name at least one well path")
    return parsed


def validate(
    path: str,
    spec_version: str = _DEFAULT_SPEC_VERSION,
    strategy: Literal["glob", "walk"] = "glob",
    show_progress: bool = True,
    check_ontology: bool = False,
    wells: "str | None" = None,
    field_offset: int = 0,
    field_limit: "int | None" = None,
) -> ValidationRun:
    """
    Validate Zarr stores at or under `path`.

    - If `path` is itself a Zarr store root (contains zarr.json), validates it
      directly without any recursive discovery.
    - Otherwise, discovers all Zarr store roots under `path` and validates each.

    Parameters
    ----------
    path         : local path or s3://bucket/prefix — any level of the hierarchy
    spec_version : DCA spec version to validate against.  Defaults to the
        current spec version (package version truncated to major.minor).
    check_ontology : when True, additionally resolve every ``ontology_term_id``
        in the experimental metadata against the EBI Ontology Lookup Service and
        check its label. Makes network requests; off by default.
    wells, field_offset, field_limit : well/field-subset selection for a
        single HCS plate. `wells` is a comma-separated list of
        exact `plate.wells[].path` strings (e.g. "A/1,B/2"). Any of these
        being non-default requires `path` to resolve directly to one store
        (raises ValueError otherwise — subsetting a multi-store discovery
        root is ambiguous) and is forwarded to `validate_zarr_node`, which
        errors if the resolved store isn't an HCS plate.

    Returns
    -------
    ValidationRun — iterable collection of ZarrNodeValidationResult with summary metrics
    """
    ModelClass = get_model(spec_version)
    path_is_store = is_zarr_store(path)

    has_subset_flags = wells is not None or field_offset or field_limit is not None
    if field_offset < 0 or (field_limit is not None and field_limit < 0):
        raise ValueError("field_offset and field_limit must be >= 0")
    if has_subset_flags and not path_is_store:
        raise ValueError(
            "wells/field_offset/field_limit require `path` to point directly "
            "at a single Zarr store, not a discovery root"
        )
    parsed_wells = _parse_wells(wells) if wells is not None else None

    stores = (
        [path.rstrip("/")]
        if path_is_store
        else discover_zarr_stores(path, strategy=strategy)
    )
    stores_discovered = len(stores)

    start_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    results = (
        _validate_with_tree(
            stores,
            path,
            spec_version,
            ModelClass,
            check_ontology=check_ontology,
            wells=parsed_wells,
            field_offset=field_offset,
            field_limit=field_limit,
        )
        if show_progress
        else _validate_plain(
            stores,
            spec_version,
            ModelClass,
            check_ontology=check_ontology,
            wells=parsed_wells,
            field_offset=field_offset,
            field_limit=field_limit,
        )
    )

    duration = time.monotonic() - t0
    summary = ValidationSummary.from_results(
        results,
        duration_seconds=duration,
        stores_discovered=stores_discovered,
        start_timestamp=start_timestamp,
    )
    return ValidationRun(
        path=path,
        results=results,
        summary=summary,
        spec_version=spec_version,
        validator_version=__version__,
    )


def list_plate_wells(path: str) -> list[tuple[str, int]]:
    """
    Dry-run: list a single HCS plate's wells and their declared field counts,
    without validating anything (--list-wells).

    Reads only each well's declared `ome.well.images` length — unlike
    _gather_plate_fields, this never lists a well's on-disk children or
    cross-checks declared fields against disk, so it performs no structural
    validation and its counts are exactly what the plate's own metadata
    declares (not narrowed to fields that also exist on disk). A well whose
    group can't be opened at all, or whose metadata is missing/blank "path",
    is omitted rather than reported as an error; a real validate() run
    reports either case as an issue.

    Returned in alphabetical well-path order for readable listing output —
    unlike _gather_plate_fields, nothing downstream of this function depends
    on plate-declaration order, so executor.map below is used purely for the
    parallelism speedup, not to preserve any particular order.

    Raises ValueError if `path` doesn't resolve to a single store, or if that
    store isn't an HCS plate.
    """
    if not is_zarr_store(path):
        raise ValueError(f"{path!r} does not point directly at a single Zarr store")

    node_path = path.rstrip("/")
    try:
        zarr_group = open_zarr_group(node_path)
    except Exception as exc:
        # This function documents only ValueError; the CLI's --list-wells
        # branch only converts ValueError into a clean UsageError, so an
        # open failure (S3/network/auth error) must not leak past here as
        # some other exception type and produce a raw stack trace instead.
        raise ValueError(f"could not open {path!r}: {exc}") from exc
    raw_ome_attrs = ome_attrs_of(zarr_group)
    if not is_hcs_plate(raw_ome_attrs):
        raise ValueError(f"{path!r} is not an HCS plate")

    well_paths = declared_well_paths(raw_ome_attrs)

    def _count_one(well_path: str) -> "tuple[str, int] | None":
        try:
            well_group = open_zarr_group(f"{node_path}/{well_path}")
        except Exception as exc:
            # A dropped well silently understates the listing's total field
            # count — since --list-wells exists to plan a chunking scheme,
            # that's a real gap in coverage, not just a cosmetic omission.
            logger.warning(
                "list_plate_wells: could not open well %r under %r (%s) — "
                "omitted from the listing",
                well_path,
                node_path,
                exc,
            )
            return None
        well_ome = ome_attrs_of(well_group)
        return (well_path, len(well_images(well_ome)))

    # Opening wells is one S3 round trip each and independent — for plates
    # with hundreds of wells (as opposed to a few huge wells),
    # doing this sequentially was the dominant cost of a supposedly-cheap
    # dry-run listing.
    with ThreadPoolExecutor() as executor:
        counts = [c for c in executor.map(_count_one, well_paths) if c is not None]
    return sorted(counts)


def _validate_plain(
    stores: list[str],
    spec_version: str,
    ModelClass: object,
    check_ontology: bool = False,
    wells: "frozenset[str] | None" = None,
    field_offset: int = 0,
    field_limit: "int | None" = None,
) -> list[ZarrNodeValidationResult]:
    results: list[ZarrNodeValidationResult] = []
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                validate_zarr_node,
                node_path,
                spec_version,
                ModelClass,
                check_ontology=check_ontology,
                wells=wells,
                field_offset=field_offset,
                field_limit=field_limit,
            ): node_path
            for node_path in stores
        }
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def _validate_with_tree(
    stores: list[str],
    root_path: str,
    spec_version: str,
    ModelClass: object,
    check_ontology: bool = False,
    wells: "frozenset[str] | None" = None,
    field_offset: int = 0,
    field_limit: "int | None" = None,
) -> list[ZarrNodeValidationResult]:
    results: list[ZarrNodeValidationResult] = []
    display = _TreeDisplay(root_path, len(stores))

    def _run_store(node_path: str) -> list[ZarrNodeValidationResult]:
        # start_store fires when a worker thread picks this up — not at
        # submission time — so only actually-running stores appear in the tree.
        display.start_store(node_path)
        return validate_zarr_node(
            node_path,
            spec_version,
            ModelClass,
            display.make_field_hook(node_path),
            check_ontology=check_ontology,
            wells=wells,
            field_offset=field_offset,
            field_limit=field_limit,
        )

    with Live(display.tree, refresh_per_second=4) as live:
        display.set_live(live)
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_run_store, node_path): node_path
                for node_path in stores
            }
            for future in as_completed(futures):
                store_path = futures[future]
                store_results = future.result()
                results.extend(store_results)
                display.complete_store(store_path, store_results)

    return results
