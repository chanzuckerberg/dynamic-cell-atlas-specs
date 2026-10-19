from dataclasses import dataclass, field
from enum import Enum

from dca_helpers import CURRENT_SPEC_VERSION, __version__
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
class PlateSubsetInfo:
    """Well/field subset coverage for one HCS plate validation run.

    Attached only to the plate-root ZarrNodeValidationResult (the one with
    parent_path is None) whenever --wells/--field-offset/--field-limit
    narrowed what was opened or deep-validated, so a partial run is never
    mistaken for full coverage.
    """

    wells_total: int  # wells declared in plate.wells
    wells_selected: int  # wells actually opened — can be < wells_total even
    # with --wells unset, if a well failed to open (reported separately as
    # an ERROR; this field doesn't imply the shortfall was deliberate)
    fields_total: int  # fields declared+on-disk across every opened well
    fields_selected: int  # fields actually deep-validated (post offset/limit slice)


@dataclass
class ImageFacts:
    """Level-0 facts about one validated image group, kept for cross-store checks.

    The projection pairing pass (validation/pairing.py) compares a projection with
    its source from these, so it needs no second read of either store. Not part of
    the report: reporter._NODE_RESULT_EXCLUDED_FIELDS drops it.
    """

    axes: tuple[str, ...]  # lowercased
    level0_shape: tuple[int, ...]
    level0_dtype: str
    level0_scale: dict[str, "float | None"]  # by lowercased axis name
    projection: "dict | None"  # raw dca.projection, when present


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
    subset_coverage: PlateSubsetInfo | None = None  # set on the plate-root result
    # of a partial (well/field-subset) HCS plate run; None otherwise
    image_facts: ImageFacts | None = None  # set on image results; internal, for
    # cross-store checks only (not serialized into reports)

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
    start_timestamp: str  # ISO 8601 UTC wall-clock timestamp when validate() began
    nodes_by_type: dict[
        ZarrNodeType, int
    ]  # count of results grouped by node_type (excludes None; includes skipped types)
    is_partial: bool = False  # True when any result carries a PlateSubsetInfo
    wells_total: int = 0  # summed PlateSubsetInfo.wells_total; 0 when not partial
    wells_selected: int = 0
    fields_total: int = 0
    fields_selected: int = 0

    @classmethod
    def from_results(
        cls,
        results: list[ZarrNodeValidationResult],
        duration_seconds: float = 0.0,
        stores_discovered: int = 0,
        start_timestamp: str = "",
    ) -> "ValidationSummary":
        nodes_by_type: dict[ZarrNodeType, int] = {}
        for r in results:
            if r.node_type is not None:
                nodes_by_type[r.node_type] = nodes_by_type.get(r.node_type, 0) + 1
        non_skipped = [r for r in results if not r.skipped]
        subsets = [r.subset_coverage for r in results if r.subset_coverage is not None]
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
            start_timestamp=start_timestamp,
            nodes_by_type=nodes_by_type,
            is_partial=bool(subsets),
            wells_total=sum(s.wells_total for s in subsets),
            wells_selected=sum(s.wells_selected for s in subsets),
            fields_total=sum(s.fields_total for s in subsets),
            fields_selected=sum(s.fields_selected for s in subsets),
        )


@dataclass
class ValidationRun:
    """Results and summary for one validate() call."""

    path: str
    results: list[ZarrNodeValidationResult]
    summary: ValidationSummary
    # Defaults to the currently-supported DCA spec version so direct
    # construction (tests, ad hoc scripts) doesn't need to care.
    spec_version: str = field(default_factory=lambda: CURRENT_SPEC_VERSION)
    # Defaults to the running package's own version so direct construction
    # (tests, ad hoc scripts) doesn't need to care.
    validator_version: str = field(default_factory=lambda: __version__)

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)
