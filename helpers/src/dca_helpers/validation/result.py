from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dca_helpers.validation.zarr_node import ZarrNodeType


class Severity(str, Enum):
    ERROR = "error"  # MUST violation — always causes passed=False
    WARNING = "warning"  # SHOULD violation — informational only


@dataclass
class Issue:
    """One validation finding for a single field or rule."""

    loc: tuple[int | str, ...]  # mirrors Pydantic's error location tuple
    message: str
    severity: Severity


@dataclass
class ZarrNodeValidationResult:
    """Aggregated validation outcome for one Zarr store."""

    node_path: str
    spec_version: str | None
    passed: bool
    parent_path: str | None = (
        None  # plate root path for HCS field results; None for standalone images
    )
    ngff_version: str | None = (
        None  # highest OME-NGFF version that passed ("0.5", "0.4", or None)
    )
    multiscale_level_count: int | None = (
        None  # number of scale-level zarr.json files read
    )
    issues: list[Issue] = field(default_factory=list)
    node_type: ZarrNodeType | None = None
    skipped: bool = False  # store was intentionally not validated (e.g., GEFF)

    # For HCS plates, one result is produced per field image (parent_path = plate root).
    # For standalone images, parent_path is None.
    # Sum across results to get totals:
    #   nodes validated            = len(results)
    #   level zarr.json validated  = sum(r.multiscale_level_count or 0 for r in results)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]


@dataclass
class ValidationSummary:
    """Aggregate counts across all validated stores."""

    stores_validated: (
        int  # non-skipped stores processed; equals stores_passed + stores_failed
    )
    level_zarr_jsons_validated: (
        int  # total scale-level zarr.json files read (non-skipped only)
    )
    stores_passed: int
    stores_failed: int
    stores_skipped: int  # stores intentionally not validated (e.g., GEFF)
    stores_discovered: int  # number of distinct store paths found before validation
    duration_seconds: float  # wall-clock seconds for the validate() call
    nodes_by_type: dict[
        ZarrNodeType, int
    ]  # count of results grouped by node_type (excludes None; includes skipped types)

    @classmethod
    def from_results(
        cls,
        results: list[ZarrNodeValidationResult],
        duration_seconds: float = 0.0,
        stores_discovered: int = 0,
    ) -> "ValidationSummary":
        nodes_by_type: dict[ZarrNodeType, int] = {}
        for r in results:
            if r.node_type is not None:
                nodes_by_type[r.node_type] = nodes_by_type.get(r.node_type, 0) + 1
        non_skipped = [r for r in results if not r.skipped]
        return cls(
            stores_validated=len(non_skipped),
            level_zarr_jsons_validated=sum(
                r.multiscale_level_count or 0 for r in non_skipped
            ),
            stores_passed=sum(1 for r in non_skipped if r.passed),
            stores_failed=sum(1 for r in non_skipped if not r.passed),
            stores_skipped=sum(1 for r in results if r.skipped),
            stores_discovered=stores_discovered,
            duration_seconds=duration_seconds,
            nodes_by_type=nodes_by_type,
        )


@dataclass
class ValidationRun:
    """Results and summary for one validate() call."""

    path: str
    results: list[ZarrNodeValidationResult]
    summary: ValidationSummary

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)
