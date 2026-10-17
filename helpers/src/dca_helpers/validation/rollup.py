"""Offline, partition-invariant consistency validation of one HCS snapshot.

Reports are evidence, not live views of a mutable store. A manifest and every
input report must refer to the same caller-managed immutable snapshot. No
source-store access (and no image pixel access) occurs during ``rollup_reports``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from dca_helpers import CURRENT_SPEC_VERSION, __version__
from dca_helpers.validation.hcs_facts import (
    HCSFieldFacts,
    HCSManifest,
    HCSPlateFacts,
    canonical_json,
)
from dca_helpers.validation.reporter import _REPORTING_FORMAT_VERSION
from dca_helpers.validation.result import (
    Issue,
    PlateSubsetInfo,
    Severity,
    ValidationRun,
    ValidationSummary,
    ZarrNodeValidationResult,
)
from dca_helpers.validation.zarr_node import ZarrNodeType

_WELL_PATH = re.compile(r"[A-Za-z0-9]+/[A-Za-z0-9]+\Z")
_FIELD_NAME = re.compile(r"[A-Za-z0-9]+\Z")


def _check_manifest(manifest: HCSManifest) -> list[str]:
    if not manifest.snapshot_id.strip():
        raise ValueError("A nonempty immutable snapshot_id is required")
    if not manifest.plate_path or manifest.plate_path.endswith("/"):
        raise ValueError(
            "Manifest plate_path must be nonempty without a trailing slash"
        )
    if manifest.validator_version != __version__:
        raise ValueError(
            "Manifest validator version differs from the running validator"
        )
    from dca_helpers.validation.registry import get_model

    get_model(manifest.spec_version)
    plate = manifest.plate
    wells = plate.declared_wells
    if (
        not plate.inventory_complete
        or len(set(wells)) != len(wells)
        or set(wells) != set(plate.wells)
    ):
        raise ValueError(
            "Manifest requires a complete, unique inventory of declared wells"
        )
    fields = []
    for well in wells:
        facts = plate.wells[well]
        if not _WELL_PATH.fullmatch(well) or not facts.inventory_complete:
            raise ValueError(f"Manifest has an incomplete or malformed well: {well!r}")
        for field in facts.expected_fields:
            if not field.startswith(well + "/") or not _FIELD_NAME.fullmatch(
                field[len(well) + 1 :]
            ):
                raise ValueError(f"Manifest has an invalid declared field: {field!r}")
        if len(set(facts.expected_fields)) != len(facts.expected_fields):
            raise ValueError(f"Manifest has duplicate declared fields in {well!r}")
        fields.extend(facts.expected_fields)
    return fields


def create_hcs_manifest(
    path: str,
    snapshot_id: str,
    spec_version: str = CURRENT_SPEC_VERSION,
) -> HCSManifest:
    """Read plate/well metadata once to record the full declared inventory.

    No fields are opened. Missing declared fields remain in the manifest,
    while unreadable wells or ambiguous declarations prevent its creation.
    Keep the source snapshot immutable throughout manifest/chunk generation.
    """
    from dca_helpers.validation.runner import validate

    run = validate(
        path.rstrip("/"),
        spec_version=spec_version,
        show_progress=False,
        field_limit=0,
        snapshot_id=snapshot_id,
    )
    if len(run.results) != 1 or not run.results[0].hcs_facts:
        raise ValueError(
            "Cannot create HCS manifest: plate/well metadata is unavailable"
        )
    try:
        plate = HCSPlateFacts.model_validate_json(run.results[0].hcs_facts)
    except ValidationError as exc:
        raise ValueError(
            "Cannot create HCS manifest from incomplete plate metadata"
        ) from exc
    manifest = HCSManifest(
        plate_path=path.rstrip("/"),
        snapshot_id=snapshot_id,
        spec_version=spec_version,
        validator_version=run.validator_version,
        plate=plate,
    )
    _check_manifest(manifest)
    return manifest


def _issue_key(issue: Issue) -> tuple:
    # Parquet's existing loc representation is list<string>. Normalize before
    # comparing, including any integer locations from an in-memory source.
    return (
        tuple(str(x) for x in issue.loc),
        issue.severity.value,
        issue.message,
        issue.code or "",
    )


def _unique_issues(issues: Iterable[Issue]) -> list[Issue]:
    by_key = {_issue_key(i): i for i in issues}
    return [by_key[key] for key in sorted(by_key)]


def _is_cross_check(issue: Issue) -> bool:
    # Schema errors can have exactly the same location as derived checks
    # (including a malformed top-level well_perturbation). Never infer origin
    # from location or message text.
    return issue.code in {
        "hcs.perturbation_agreement",
        "hcs.well_perturbation_agreement",
        "hcs.axes_uniformity",
        "hcs.chunk_uniformity",
    }


def _read_rows(report: bytes | str | Path) -> list[dict]:
    try:
        source = pa.BufferReader(report) if isinstance(report, bytes) else report
        return pq.ParquetFile(source).read().to_pylist()
    except (OSError, pa.ArrowException, TypeError) as exc:
        raise ValueError(f"Cannot read validation report: {exc}") from exc


def _result_from_row(row: dict, spec_version: str) -> ZarrNodeValidationResult:
    try:
        if not isinstance(row["passed"], bool) or row["skipped"] is not False:
            raise ValueError("HCS report must contain non-skipped validation outcomes")
        if not isinstance(row["issues"], list):
            raise ValueError("Report issues must be a list")
        issues = []
        for issue in row["issues"]:
            if not isinstance(issue["message"], str) or not isinstance(
                issue["loc"], list
            ):
                raise ValueError("Malformed report issue")
            if not all(isinstance(x, str) for x in issue["loc"]):
                raise ValueError("Malformed report issue location")
            code = issue.get("code")
            if code is not None and not isinstance(code, str):
                raise ValueError("Malformed report issue code")
            issues.append(
                Issue(
                    tuple(issue["loc"]),
                    issue["message"],
                    Severity(issue["severity"]),
                    code,
                )
            )
        if row["passed"] != (not any(i.severity == Severity.ERROR for i in issues)):
            raise ValueError("Report passed status contradicts its error findings")
        count = row["multiscale_level_count"]
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("Report has an invalid multiscale_level_count")
        return ZarrNodeValidationResult(
            node_path=row["node_path"],
            parent_path=row["parent_path"],
            spec_version=spec_version,
            passed=row["passed"],
            ngff_version=row["ngff_version"],
            multiscale_level_count=row["multiscale_level_count"],
            issues=_unique_issues(issues),
            node_type=ZarrNodeType[row["node_type"]],
            hcs_facts=row["hcs_facts"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed node in validation report") from exc


def _field_signature(result: ZarrNodeValidationResult, facts: HCSFieldFacts) -> str:
    return canonical_json(
        {
            "facts": facts.model_dump(),
            "passed": result.passed,
            "ngff_version": result.ngff_version,
            "multiscale_level_count": result.multiscale_level_count,
            "issues": [_issue_key(i) for i in result.issues],
        }
    )


def rollup_reports(
    reports: Iterable[bytes | str | Path], manifest: HCSManifest
) -> ValidationRun:
    """Combine persisted chunk reports, rerun consistency, and verify coverage.

    Incompatible, malformed, legacy or conflicting retry inputs raise ValueError;
    they are not biological findings. Missing/failed field reads produce explicit
    ``rollup.coverage`` errors and a partial result, never a complete pass.
    Original per-field and structural findings survive; chunk-local derived
    consistency findings are replaced with checks over the deduplicated union.
    """
    from dca_helpers.validation.validator import (
        _EXPERIMENTAL_SPEC_VERSIONS,
        _check_well_perturbation_agreement,
        _uniformity_issues,
    )

    t0 = time.monotonic()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Revalidate even programmatically constructed/copy-updated model instances.
    manifest = HCSManifest.model_validate_json(manifest.model_dump_json())
    expected = _check_manifest(manifest)
    expected_set = set(expected)
    fields: dict[str, tuple[ZarrNodeValidationResult, HCSFieldFacts, str]] = {}
    opened_wells: set[str] = set()
    structural_issues: list[Issue] = []
    root_signatures: dict[tuple[str, ...], str] = {}
    report_count = 0
    root_ngff: str | None = None
    for report in reports:
        report_count += 1
        rows = _read_rows(report)
        summaries = [
            r for r in rows if r.get("record_type") == "validation_run_summary"
        ]
        if len(summaries) != 1:
            raise ValueError(
                "Each report must contain exactly one validation_run_summary"
            )
        provenance = {
            "dca_spec_version": manifest.spec_version,
            "dca_validator_version": manifest.validator_version,
            "snapshot_id": manifest.snapshot_id,
            "reporting_format_version": _REPORTING_FORMAT_VERSION,
        }
        for row in rows:
            for name, value in provenance.items():
                if row.get(name) != value:
                    raise ValueError(
                        f"Incompatible {name}: expected {value!r}, got {row.get(name)!r}. "
                        "Regenerate reports with this validator and the manifest snapshot."
                    )
            if (
                not isinstance(row.get("input_path"), str)
                or row["input_path"].rstrip("/") != manifest.plate_path
            ):
                raise ValueError(
                    "Report input_path does not match manifest plate identity"
                )
            if row.get("record_type") not in ("validation_run_summary", "node"):
                raise ValueError("Unsupported report record_type")
        nodes = [
            _result_from_row(r, manifest.spec_version)
            for r in rows
            if r["record_type"] == "node"
        ]
        roots = [r for r in nodes if r.parent_path is None]
        if len(roots) != 1 or roots[0].node_path != manifest.plate_path:
            raise ValueError(
                "Each report must contain exactly one matching HCS plate root"
            )
        root = roots[0]
        if root.node_type != ZarrNodeType.HCS_PLATE or root.hcs_facts is None:
            raise ValueError(
                "Report lacks HCS plate facts; regenerate it with this validator"
            )
        plate = HCSPlateFacts.model_validate_json(root.hcs_facts)
        if (
            plate.metadata_digest != manifest.plate.metadata_digest
            or plate.declared_wells != manifest.plate.declared_wells
            or not plate.inventory_complete
        ):
            raise ValueError("Plate metadata differs from manifest snapshot")
        for well, facts in plate.wells.items():
            if well not in manifest.plate.wells or facts != manifest.plate.wells[well]:
                raise ValueError(
                    f"Well {well!r} metadata differs from manifest snapshot"
                )
        opened_wells.update(plate.wells)
        kept_issues = [i for i in root.issues if not _is_cross_check(i)]
        root_signature = canonical_json(
            {"ngff": root.ngff_version, "issues": [_issue_key(i) for i in kept_issues]}
        )
        scope = tuple(sorted(plate.wells))
        if scope in root_signatures and root_signatures[scope] != root_signature:
            raise ValueError("Conflicting plate outcomes for identical well scope")
        root_signatures[scope] = root_signature
        if report_count > 1 and root.ngff_version != root_ngff:
            raise ValueError("Incompatible plate NGFF versions")
        root_ngff = root.ngff_version
        structural_issues.extend(kept_issues)
        for result in nodes:
            if result is root:
                continue
            prefix = manifest.plate_path + "/"
            if (
                result.parent_path != manifest.plate_path
                or not isinstance(result.node_path, str)
                or not result.node_path.startswith(prefix)
                or result.node_type != ZarrNodeType.IMAGE
            ):
                raise ValueError(
                    "Report contains an unrelated or malformed field identity"
                )
            relative = result.node_path[len(prefix) :]
            if (
                relative not in expected_set
                or relative.rsplit("/", 1)[0] not in plate.wells
            ):
                raise ValueError(
                    f"Field {relative!r} is outside the declared manifest/report scope"
                )
            if result.hcs_facts is None:
                raise ValueError(
                    f"Field {relative!r} lacks HCS facts; regenerate its report"
                )
            facts = HCSFieldFacts.model_validate_json(result.hcs_facts)
            if facts.read_success and (
                facts.axes is None
                or result.multiscale_level_count is None
                or (result.multiscale_level_count > 0 and facts.chunk_shape is None)
                or (
                    result.passed
                    and (
                        not facts.axes
                        or not facts.chunk_shape
                        or result.multiscale_level_count == 0
                        or len(facts.axes) != len(facts.chunk_shape)
                        or any(size <= 0 for size in facts.chunk_shape)
                    )
                )
            ):
                raise ValueError(
                    f"Field {relative!r} has incomplete successful-read facts"
                )
            if not facts.read_success and (
                result.passed
                or facts.axes is not None
                or facts.chunk_shape is not None
                or facts.perturbation is not None
            ):
                raise ValueError(
                    f"Field {relative!r} has contradictory failed-read facts"
                )
            signature = _field_signature(result, facts)
            if relative in fields and fields[relative][2] != signature:
                raise ValueError(f"Conflicting duplicate field report: {relative!r}")
            fields[relative] = result, facts, signature
    if not report_count:
        raise ValueError("At least one validation report is required")

    observed = [
        fp
        for fp in expected
        if fp in fields
        and fields[fp][1].read_success
        and fields[fp][1].axes is not None
        and fields[fp][1].chunk_shape is not None
        and fields[fp][0].multiscale_level_count
    ]
    missing = [fp for fp in expected if fp not in fields]
    observed_set = set(observed)
    failed = [fp for fp in expected if fp in fields and fp not in observed_set]
    missing_wells = [w for w in manifest.plate.declared_wells if w not in opened_wells]
    issues = structural_issues
    if manifest.spec_version in _EXPERIMENTAL_SPEC_VERSIONS:
        issues.extend(
            _check_well_perturbation_agreement(
                observed,
                {fp: fields[fp][1].perturbation for fp in observed},
                {
                    w: f.perturbation
                    for w, f in manifest.plate.wells.items()
                    if w in opened_wells
                },
            )
        )
    issues.extend(
        _uniformity_issues(
            [tuple(fields[fp][1].axes) for fp in observed],
            [
                tuple(fields[fp][1].chunk_shape)
                for fp in observed
                if fields[fp][1].chunk_shape is not None
            ],
        )
    )
    subset = None
    if missing or failed or missing_wells:
        for label, paths in (
            ("missing fields", missing),
            ("unreadable fields", failed),
            ("unobserved wells", missing_wells),
        ):
            if paths:
                issues.append(
                    Issue(
                        ("rollup", "coverage"),
                        f"Incomplete HCS rollup: {len(paths)} {label}: "
                        f"{paths[:10]!r}"
                        + (" (first 10 shown)" if len(paths) > 10 else ""),
                        Severity.ERROR,
                    )
                )
        subset = PlateSubsetInfo(
            wells_total=len(manifest.plate.declared_wells),
            wells_selected=len(opened_wells),
            fields_total=len(expected),
            fields_selected=len(observed),
        )
    root = ZarrNodeValidationResult(
        node_path=manifest.plate_path,
        spec_version=manifest.spec_version,
        passed=not any(i.severity == Severity.ERROR for i in issues),
        ngff_version=root_ngff,
        node_type=ZarrNodeType.HCS_PLATE,
        issues=_unique_issues(issues),
        subset_coverage=subset,
        # A rolled-up result is not itself a source chunk. Avoid recursively
        # accepting it as a fresh independent observation on another rollup.
    )
    results = [root] + [fields[fp][0] for fp in expected if fp in fields]
    return ValidationRun(
        path=manifest.plate_path,
        results=results,
        summary=ValidationSummary.from_results(
            results,
            duration_seconds=time.monotonic() - t0,
            stores_discovered=1,
            start_timestamp=timestamp,
        ),
        spec_version=manifest.spec_version,
        validator_version=manifest.validator_version,
        snapshot_id=manifest.snapshot_id,
    )
