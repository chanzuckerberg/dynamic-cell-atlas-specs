"""
Pydantic models for Dynamic Cell Atlas spec v0.2.

Each model encodes exactly the rules from:
  - docs/v0.2/array-standard.rst
  - docs/v0.2/ome-zarr-metadata.rst
  - docs/v0.2/channel-metadata.rst
  - docs/v0.2/normalization-statistics.rst
  - docs/v0.2/schema.json  (authoritative field names and enums)

MUST violations  → raise ValueError without a prefix
                   → classified as ERROR by the runner.
SHOULD violations → raise ValueError prefixed "[SHOULD]"
                    → classified as WARNING by the runner.

The root model DCAStoreSpecV0_2 is the single entry point:

    DCAStoreSpecV0_2.model_validate(data)

where `data` is a dict produced by
validation.validator._build_node_dict() extended with:
    spec_version — "0.2"
    dca          — dict from zarr_group.attrs.get("dca"), or None if the key is absent

Changes from v0.1:
  array-standard.rst § Multiresolution:
    - The "≥ 3 scale levels" MUST is removed.
      Replaced by: MUST have at least one level where every spatial dim ≤ 2048.
  channel-metadata.rst (entire section — new in v0.2):
    - MUST include a `dca` key in the image group zarr.json.
    - `dca` MUST contain version ("0.2"), url, channels, and normalization_statistics.
    - Each channel MUST have name, index, and description.
    - Optional channel fields: channel_type, biological_annotation.
  normalization-statistics.rst (entire section — new in v0.2):
    - normalization_statistics keyed by channel index string.
    - Each entry MUST have dataset_statistics with p1, p5, p95, p99.

All array-level rules (codec, chunk size, shard size, dtype) are unchanged from v0.1.
DCAScaleLevelSpecV0_2 inherits them directly without modification.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pydantic
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dca_helpers.validation.versions.v0_1.models import (
    AXES_REQUIRED,
    DOWNSAMPLING_FACTOR_RECOMMENDED,
    LABEL_DTYPE_RECOMMENDED,
    NON_SPATIAL_AXES,
    SPATIAL_AXES,
)
from dca_helpers.validation.versions.v0_1.models import (
    DCAScaleLevelSpec as DCAScaleLevelSpecV0_1,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# array-standard.rst § Multiresolution
COARSE_LEVEL_MAX_SPATIAL = 2048  # MUST — every spatial dim at the coarsest level

# ome-zarr-metadata.rst § Units / OME-NGFF v0.5 unit vocabularies
SPATIAL_UNIT_VOCABULARY = frozenset(
    {
        "angstrom",
        "attometer",
        "centimeter",
        "decimeter",
        "exameter",
        "femtometer",
        "foot",
        "gigameter",
        "hectometer",
        "inch",
        "kilometer",
        "megameter",
        "meter",
        "micrometer",
        "mile",
        "millimeter",
        "nanometer",
        "parsec",
        "petameter",
        "picometer",
        "terameter",
        "yard",
        "yoctometer",
        "yottameter",
        "zeptometer",
        "zettameter",
    }
)
TEMPORAL_UNIT_VOCABULARY = frozenset(
    {
        "attosecond",
        "centisecond",
        "day",
        "decisecond",
        "exasecond",
        "femtosecond",
        "gigasecond",
        "hectosecond",
        "hour",
        "kilosecond",
        "megasecond",
        "microsecond",
        "millisecond",
        "minute",
        "nanosecond",
        "petasecond",
        "picosecond",
        "second",
        "terasecond",
        "yoctosecond",
        "yottasecond",
        "zeptosecond",
        "zettasecond",
    }
)

# schema.json TimeSampling / AcquisitionMetadata.time_sampling. An absent value reads
# as "uniform", so an existing store keeps its current reading.
TIME_SAMPLING_VALUES = frozenset({"uniform", "variable"})

# channel-metadata.rst § Guidance on Channel Type / schema.json ChannelMetadata.channel_type
# Vocabulary matches schema.json enum exactly.
CHANNEL_TYPES = frozenset({"fluorescence", "chromogenic", "labelfree", "predicted"})

# channel-metadata.rst § Guidance on Marker Type Vocabulary / schema.json BiologicalAnnotation
MARKER_TYPES = frozenset({"endogenous_tag", "live_cell_dye", "fixed_dye", "antibody"})

# schema.json BiologicalAnnotation.cpg_labeled_structure enum
CPG_LABELED_STRUCTURES = frozenset(
    {
        "\u03b2-tubulin",
        "Endoplasmic Reticulum",
        "f-Actin",
        "Golgi, Plasma Membrane",
        "Golgi, Plasma Membrane, f-Actin",
        "Lysosomes",
        "Lysosomes, Endosomes",
        "Mitochondria",
        "Nucleus",
        "Nucleolus, Cytoplasmic RNA",
        "Nucleus, Nucleolus, Cytoplasmic RNA",
        "None",
        "Unknown",
    }
)

# schema.json BiologicalAnnotation.cpg_labeled_molecule enum
CPG_LABELED_MOLECULES = frozenset(
    {
        "Acidic Compartments",
        "\u03b2-tubulin",
        "DNA",
        "DNA, RNA",
        "f-Actin",
        "f-Actin, N-acetyl-D-glucosamine",
        "Glycoproteins",
        "Mitochondria",
        "N-acetyl-D-glucosamine",
        "RNA",
        "None",
        "Unknown",
    }
)


# ---------------------------------------------------------------------------
# Array-level model — all rules identical to v0.1, inherit directly
# ---------------------------------------------------------------------------


class DCAScaleLevelSpecV0_2(DCAScaleLevelSpecV0_1):
    """
    Validates one scale level (one Zarr array) inside a DCA v0.2 multiscale store.

    All array-level rules are inherited unchanged from DCAScaleLevelSpecV0_1:
      - array-standard.rst § Compression         (codec MUST / SHOULD)
      - array-standard.rst § Performance / Chunk Size      (chunk size MUST / SHOULD)
      - array-standard.rst § Performance / Shard Size      (shard size MUST / SHOULD)
      - array-standard.rst § Data Types          (dtype SHOULD)

    New in v0.2:
      - ome-zarr-metadata.rst § Dimensions: dimension_names MUST be present and
        match the axis names (ome-zarr-metadata.rst § Dimensions).
    """

    # ome-zarr-metadata.rst § Dimensions — MUST
    dimension_names: list[str] | None = Field(default=None, validate_default=True)

    @field_validator("dimension_names", mode="before")
    @classmethod
    def dimension_names_must_match_axes(cls, v: object) -> object:
        if v is None:
            raise ValueError(
                "dimension_names is missing from zarr.json array metadata. "
                "MUST be present and match axis names "
                "(ome-zarr-metadata.rst § Dimensions)."
            )
        normalized = list(v) if isinstance(v, (list, tuple)) else v
        if normalized != AXES_REQUIRED:
            raise ValueError(
                f"dimension_names {normalized} must match axes {AXES_REQUIRED} "
                "(ome-zarr-metadata.rst § Dimensions)."
            )
        return normalized


class DCALabelScaleLevelSpecV0_2(DCAScaleLevelSpecV0_2):
    """
    Validates one scale level inside a DCA v0.2 label image store.

    Inherits all array-level rules from DCAScaleLevelSpecV0_2 and overrides the
    dtype recommendation: label arrays SHOULD use uint32, not uint8/uint16.
    """

    # array-standard.rst § Data Types — SHOULD (label-specific override)
    @model_validator(mode="after")
    def check_dtype_recommendation(self) -> "DCALabelScaleLevelSpecV0_2":
        normalized = str(np.dtype(self.dtype))
        if normalized != LABEL_DTYPE_RECOMMENDED:
            raise ValueError(
                f"[SHOULD] dtype '{self.dtype}' is unusual for a label array; "
                f"label arrays should use {LABEL_DTYPE_RECOMMENDED}"
            )
        return self


# ---------------------------------------------------------------------------
# Normalization statistics models — new in v0.2
# (normalization-statistics.rst + schema.json)
# ---------------------------------------------------------------------------


class DCAIntensityStatistics(BaseModel):
    """
    Summary intensity statistics for one scope (dataset or single timepoint).
    schema.json #/$defs/IntensityStatistics
    """

    model_config = ConfigDict(extra="allow")

    # normalization-statistics.rst § Required Fields — MUST
    p1: float
    p5: float
    p95: float
    p99: float

    # normalization-statistics.rst § Optional Fields — MAY
    # schema.json IntensityStatistics: p95_p5, p99_p1, iqr have minimum: 0
    p95_p5: float | None = Field(default=None, ge=0)
    p99_p1: float | None = Field(default=None, ge=0)
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    iqr: float | None = Field(default=None, ge=0)


class DCANormalizationMetadata(BaseModel):
    """
    Optional parameters recorded when computing normalization statistics.
    schema.json #/$defs/NormalizationMetadata
    """

    model_config = ConfigDict(extra="allow")

    # schema.json NormalizationMetadata: grid has minimum: 1
    grid: int | None = Field(default=None, ge=1)


class DCAChannelNormalization(BaseModel):
    """
    Normalization statistics for one channel.
    schema.json #/$defs/ChannelNormalization
    """

    model_config = ConfigDict(extra="allow")

    # normalization-statistics.rst § Channel Normalization Fields — MUST
    dataset_statistics: DCAIntensityStatistics

    # normalization-statistics.rst § Channel Normalization Fields — MAY
    # Keys are zero-based timepoint indices as strings (e.g. "0", "1").
    timepoint_statistics: dict[str, DCAIntensityStatistics] | None = None


class DCANormalizationStatistics(BaseModel):
    """
    Per-channel normalization statistics keyed by zero-based channel index string.
    schema.json #/$defs/NormalizationStatistics

    Keys are channel index strings (e.g. "0", "1") plus an optional "metadata" key.
    No other keys are permitted (schema.json additionalProperties: false).
    """

    model_config = ConfigDict(extra="allow")

    metadata: DCANormalizationMetadata | None = None

    @model_validator(mode="before")
    @classmethod
    def check_channel_keys(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        for key in data:
            if key == "metadata":
                continue
            if not str(key).isdigit():
                raise ValueError(
                    f"Unexpected key '{key}' in normalization_statistics; "
                    "only channel index keys (e.g. '0', '1') and 'metadata' are allowed "
                    "(normalization-statistics.rst, schema.json additionalProperties: false)"
                )
        return data

    @model_validator(mode="after")
    def validate_channel_entries(self) -> "DCANormalizationStatistics":
        for key, value in (self.model_extra or {}).items():
            try:
                DCAChannelNormalization.model_validate(value)
            except pydantic.ValidationError as exc:
                raise ValueError(f"normalization_statistics['{key}']: {exc}") from exc
        return self


# ---------------------------------------------------------------------------
# Channel metadata models — new in v0.2
# ---------------------------------------------------------------------------


class DCABiologicalAnnotation(BaseModel):
    """
    Optional structured biological target description for one channel.

    channel-metadata.rst § Guidance on Channel Metadata Vocabulary /
                            Biological Annotation Entries
    schema.json #/$defs/BiologicalAnnotation
    All fields are optional. marker_type, cpg_labeled_structure, and
    cpg_labeled_molecule SHOULD use the controlled vocabularies.
    """

    model_config = ConfigDict(extra="allow")

    # schema.json BiologicalAnnotation fields
    biological_target: str | None = None
    marker_type: str | None = None
    marker: str | None = None
    cpg_labeled_structure: str | None = None
    cpg_labeled_molecule: str | None = None

    # channel-metadata.rst § Guidance on Marker Type Vocabulary — SHOULD
    @field_validator("marker_type")
    @classmethod
    def marker_type_vocabulary(cls, v: str | None) -> str | None:
        if v is not None and v not in MARKER_TYPES:
            raise ValueError(
                f"[SHOULD] marker_type '{v}' is not in the recommended vocabulary "
                f"{sorted(MARKER_TYPES)}"
            )
        return v

    # schema.json BiologicalAnnotation.cpg_labeled_structure enum — SHOULD
    @field_validator("cpg_labeled_structure")
    @classmethod
    def cpg_labeled_structure_vocabulary(cls, v: str | None) -> str | None:
        if v is not None and v not in CPG_LABELED_STRUCTURES:
            raise ValueError(
                f"[SHOULD] cpg_labeled_structure '{v}' is not in the CellPainting "
                f"Gallery vocabulary {sorted(CPG_LABELED_STRUCTURES)}"
            )
        return v

    # schema.json BiologicalAnnotation.cpg_labeled_molecule enum — SHOULD
    @field_validator("cpg_labeled_molecule")
    @classmethod
    def cpg_labeled_molecule_vocabulary(cls, v: str | None) -> str | None:
        if v is not None and v not in CPG_LABELED_MOLECULES:
            raise ValueError(
                f"[SHOULD] cpg_labeled_molecule '{v}' is not in the CellPainting "
                f"Gallery vocabulary {sorted(CPG_LABELED_MOLECULES)}"
            )
        return v


class DCAChannelMetadataV0_2(BaseModel):
    """
    Metadata for one channel in a DCA v0.2 image.

    channel-metadata.rst § Channel Fields / Required Fields
        — name, index, description MUST.
    channel-metadata.rst § Channel Fields / Optional Fields
        — channel_type, biological_annotation.
    """

    model_config = ConfigDict(extra="allow")

    # channel-metadata.rst § Channel Fields / Required Fields — MUST
    name: str
    # schema.json ChannelMetadata.index has minimum: 0
    index: int = Field(ge=0)
    description: str

    # channel-metadata.rst § Channel Fields / Optional Fields
    channel_type: str | None = None
    biological_annotation: DCABiologicalAnnotation | None = None

    # channel-metadata.rst § Guidance on Channel Metadata Vocabulary /
    #                         Guidance on Channel Type — SHOULD
    @field_validator("channel_type")
    @classmethod
    def channel_type_vocabulary(cls, v: str | None) -> str | None:
        if v is not None and v not in CHANNEL_TYPES:
            raise ValueError(
                f"[SHOULD] channel_type '{v}' is not in the recommended vocabulary "
                f"{sorted(CHANNEL_TYPES)}"
            )
        return v

    # channel-metadata.rst § Channel Fields / Optional Fields:
    # biological_annotation SHOULD be provided for fluorescence and
    # predicted channels — SHOULD
    @model_validator(mode="after")
    def check_biological_annotation_for_typed_channels(
        self,
    ) -> "DCAChannelMetadataV0_2":
        if (
            self.channel_type in ("fluorescence", "predicted")
            and self.biological_annotation is None
        ):
            raise ValueError(
                f"[SHOULD] Channel '{self.name}' has channel_type "
                f"'{self.channel_type}' but no biological_annotation. "
                f"biological_annotation SHOULD be provided "
                f"for fluorescence and predicted channels."
            )
        return self


class DCAAcquisitionViewV0_2(BaseModel):
    """
    The two ``dca.acquisition`` fields the store-level time rules read.

    A deliberately narrow *view*, not the generated ``AcquisitionMetadata``. The block
    is already validated in full by ``validate_dca_experimental_metadata_v0_2``, which
    pulls every experimental block out of ``dca`` and checks it against the generated
    model. ``validator._run_dca_validation`` runs that alongside this store-structure
    model and concatenates both sets of issues, so validating the block here as well
    would report every acquisition defect twice and would couple these structural rules
    to the full experimental model. This reads the two fields the time rules need and
    leaves ownership of the block where it already sits.

    ``time_sampling`` is typed ``str`` rather than the generated enum for a related
    reason: an unknown value has to reach the time rules as a value they can report
    against the array, not as a parse failure that aborts the surrounding checks.
    """

    model_config = ConfigDict(extra="allow")

    time_sampling: str | None = None
    timestamps: list[float] | None = None


class DCAGroupMetadataV0_2(BaseModel):
    """
    Validates the 'dca' object in the image group zarr.json.

    schema.json dca.required: version, url, channels, normalization_statistics.
    channel-metadata.rst + normalization-statistics.rst
    """

    model_config = ConfigDict(extra="allow")

    # schema.json dca.required — MUST
    version: Literal["0.2"]
    # schema.json dca.url — required (MUST be present); value is not validated
    url: str
    channels: list[DCAChannelMetadataV0_2]
    normalization_statistics: DCANormalizationStatistics

    # Non-floor, and read only by the store-level time rules. Absent on most stores.
    acquisition: DCAAcquisitionViewV0_2 | None = None

    # channel-metadata.rst § Channel Fields / Required Fields:
    # index is the zero-based channel index — duplicate indices are ambiguous — MUST
    @model_validator(mode="after")
    def check_channel_indices_unique(self) -> "DCAGroupMetadataV0_2":
        indices = [ch.index for ch in self.channels]
        seen: set[int] = set()
        duplicates = [  # type: ignore[func-returns-value]
            i for i in indices if i in seen or seen.add(i)
        ]
        if duplicates:
            raise ValueError(
                f"Channel indices must be unique; "
                f"duplicate index values: {sorted(set(duplicates))}"
            )
        return self


# ---------------------------------------------------------------------------
# Store-level model (root entry point)
# ---------------------------------------------------------------------------


class DCAStoreSpecV0_2(BaseModel):
    """
    Root DCA v0.2 model. Validates a full multiscale Zarr store.

    Call DCAStoreSpecV0_2.model_validate(data) where data is the dict
    produced by validation.validator._build_node_dict() extended with:
        spec_version — "0.2"
        dca          — dict from zarr_group.attrs.get("dca"), or None
    """

    model_config = ConfigDict(extra="allow")

    spec_version: Literal["0.2"]
    axes: list[str]
    levels: list[DCAScaleLevelSpecV0_2]

    # ome-zarr-metadata.rst § Units — populated by _build_node_dict()
    axis_units: dict[str, str | None] = Field(default_factory=dict)

    # Level 0's scale per axis — populated by _build_node_dict(). Only read for the
    # variable-time-sampling rules below; None means the scale could not be read,
    # which is not the same as 1.0.
    axis_scales: dict[str, float | None] = Field(default_factory=dict)

    # channel-metadata.rst § Location in Zarr store — validated as a nested model.
    # Pydantic coerces the raw dict to DCAGroupMetadataV0_2; None means key is absent.
    # validate_default=True ensures check_dca_present fires even when dca is absent
    # from the input dict and the field falls back to its default (None).
    dca: DCAGroupMetadataV0_2 | None = Field(default=None, validate_default=True)

    # ------------------------------------------------------------------
    # array-standard.rst § Dimensions — MUST
    # ------------------------------------------------------------------

    @field_validator("axes")
    @classmethod
    def axes_must_be_tczyx(cls, v: list[str]) -> list[str]:
        if v != AXES_REQUIRED:
            raise ValueError(
                f"Axes must be {AXES_REQUIRED}; got {v}. "
                "All datasets must be 5D with dimensions in t, c, z, y, x order."
            )
        return v

    # ------------------------------------------------------------------
    # array-standard.rst § Multiresolution — MUST
    # Replaces v0.1's "≥ 3 levels" rule with a size-based requirement:
    # at least one level must have every spatial dimension ≤ 2048.
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_coarse_level_exists(self) -> "DCAStoreSpecV0_2":
        for level in self.levels:
            _t, _c, z, y, x = level.shape
            if (
                z <= COARSE_LEVEL_MAX_SPATIAL
                and y <= COARSE_LEVEL_MAX_SPATIAL
                and x <= COARSE_LEVEL_MAX_SPATIAL
            ):
                return self
        shapes = [(lv.path, lv.shape) for lv in self.levels]
        raise ValueError(
            f"No scale level has all spatial dimensions ≤ "
            f"{COARSE_LEVEL_MAX_SPATIAL}. "
            f"MUST include a level-of-detail where every spatial dim ≤ "
            f"{COARSE_LEVEL_MAX_SPATIAL} "
            f"(array-standard.rst § Multiresolution). Level shapes: {shapes}"
        )

    # ------------------------------------------------------------------
    # array-standard.rst § Multiresolution — MUST (t/c), SHOULD (spatial)
    # Downsampling factors: t and c MUST be 1; spatial SHOULD be ≈ 2.
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_downsampling_factors(self) -> "DCAStoreSpecV0_2":
        if len(self.levels) < 2:
            return self

        must_errors: list[str] = []
        should_warnings: list[str] = []

        for i in range(len(self.levels) - 1):
            fine = self.levels[i].shape
            coarse = self.levels[i + 1].shape
            chunks = self.levels[i].chunk_shape

            for dim_idx, dim_name in enumerate(AXES_REQUIRED):
                fine_dim = fine[dim_idx]
                coarse_dim = coarse[dim_idx]
                chunk_dim = chunks[dim_idx]

                if coarse_dim == 0:
                    continue

                factor = fine_dim / coarse_dim

                if dim_name in NON_SPATIAL_AXES:
                    # array-standard.rst § Multiresolution — MUST:
                    # t and c not downsampled
                    if fine_dim != coarse_dim:
                        must_errors.append(
                            f"Level {i}→{i+1}: downsampling factor for '{dim_name}' "
                            f"must be 1 (got {fine_dim}/{coarse_dim}={factor:.2f})"
                        )
                else:
                    # array-standard.rst § Multiresolution — SHOULD: spatial factor ≈ 2.
                    # Downsampling legitimately stops once an axis reaches its chunk
                    # floor, so skip when fine_dim is already <= its chunk size.
                    if fine_dim > chunk_dim and not math.isclose(
                        factor, DOWNSAMPLING_FACTOR_RECOMMENDED, rel_tol=0.05
                    ):
                        should_warnings.append(
                            f"[SHOULD] Level {i}→{i+1}: downsampling factor for '{dim_name}' "
                            f"should be {DOWNSAMPLING_FACTOR_RECOMMENDED} "
                            f"(got {fine_dim}/{coarse_dim}≈{factor:.2f})"
                        )

        all_issues = must_errors + should_warnings
        if all_issues:
            raise ValueError("\n".join(all_issues))

        return self

    # ------------------------------------------------------------------
    # ome-zarr-metadata.rst § Units — MUST
    # Spatial (z, y, x) and temporal (t) axes must declare units from the
    # OME-NGFF v0.5 vocabulary. Channel axis (c) has no unit requirement.
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_axis_units(self) -> "DCAStoreSpecV0_2":
        errors: list[str] = []
        for ax in sorted(SPATIAL_AXES):
            unit = self.axis_units.get(ax)
            if unit is None:
                errors.append(
                    f"Spatial axis '{ax}' has no unit. "
                    "MUST use a unit from the OME-NGFF v0.5 spatial vocabulary "
                    "(ome-zarr-metadata.rst § Units)."
                )
            elif unit not in SPATIAL_UNIT_VOCABULARY:
                errors.append(
                    f"Spatial axis '{ax}' unit '{unit}' is not in the OME-NGFF v0.5 "
                    "spatial unit vocabulary "
                    "(ome-zarr-metadata.rst § Units)."
                )
        unit = self.axis_units.get("t")
        if unit is None:
            # Relaxed to SHOULD for a declared variable-rate acquisition: OME-NGFF
            # v0.5 has no 'frame' unit, so a store whose frames are unevenly spaced
            # cannot name one honestly. The declaration is not free — it pulls in the
            # timestamps MUSTs in check_variable_time_sampling below.
            if self._declares_variable_sampling():
                errors.append(
                    "[SHOULD] Temporal axis 't' has no unit, which "
                    "'acquisition.time_sampling = \"variable\"' permits. OME-NGFF "
                    "v0.5 has no 'frame' unit; the per-frame "
                    "'acquisition.timestamps' are authoritative "
                    "(ome-zarr-metadata.rst § Units)."
                )
            else:
                errors.append(
                    "Temporal axis 't' has no unit. "
                    "MUST use a unit from the OME-NGFF v0.5 temporal vocabulary, or "
                    "declare 'acquisition.time_sampling = \"variable\"' with "
                    "'acquisition.timestamps' "
                    "(ome-zarr-metadata.rst § Units)."
                )
        elif unit not in TEMPORAL_UNIT_VOCABULARY:
            errors.append(
                f"Temporal axis 't' unit '{unit}' is not in the OME-NGFF v0.5 "
                "temporal unit vocabulary "
                "(ome-zarr-metadata.rst § Units)."
            )
        # Folded into this validator rather than added as a second one: pydantic
        # aborts the remaining mode="after" validators at the first raise, so a
        # variable-sampling store would emit the SHOULD above and never reach the
        # timestamps MUSTs — losing exactly the checks that make the relaxation
        # verifiable. One accumulating validator reports both.
        errors.extend(self._variable_time_sampling_errors())
        if errors:
            raise ValueError("\n".join(errors))
        return self

    def _acquisition(self) -> DCAAcquisitionViewV0_2 | None:
        """This group's ``dca.acquisition`` view, when it carries one."""
        return getattr(self.dca, "acquisition", None) if self.dca else None

    def _declares_variable_sampling(self) -> bool:
        """Whether this group declares ``acquisition.time_sampling = "variable"``."""
        return getattr(self._acquisition(), "time_sampling", None) == "variable"

    # ------------------------------------------------------------------
    # ome-zarr-metadata.rst § Units / experimental-metadata.rst § Acquisition
    # Declaring variable sampling is an assertion about the data, so it carries
    # its own MUSTs. Everything here is inside the 'variable' branch: a store
    # that does not declare it is unaffected.
    # ------------------------------------------------------------------

    def _timestamps_errors(self, timestamps: list[float]) -> list[str]:
        """Shape rules for a present ``timestamps`` array, whatever the mode.

        Applied whenever the array is there, not only under ``variable``: a
        ``uniform`` group may carry per-frame times too, and a mis-sized or
        out-of-order array is as wrong there as anywhere.
        """
        errors: list[str] = []
        # Level 0's T. levels[0].shape is [t, c, z, y, x] per the axes MUST above,
        # and is the same shape source the multiresolution rules read.
        expected = self.levels[0].shape[0] if self.levels else None
        if expected is not None and len(timestamps) != expected:
            errors.append(
                f"'acquisition.timestamps' has {len(timestamps)} entries but the "
                f"level-0 array has T={expected}. Lengths MUST match: one "
                "timestamp per timepoint."
            )
        if any(b < a for a, b in zip(timestamps, timestamps[1:])):
            errors.append(
                "'acquisition.timestamps' MUST be non-decreasing; frames are "
                "recorded in acquisition order."
            )
        return errors

    def _variable_time_sampling_errors(self) -> list[str]:
        acquisition = self._acquisition()
        declared = getattr(acquisition, "time_sampling", None)
        timestamps = getattr(acquisition, "timestamps", None)

        if declared is None:
            return self._timestamps_errors(timestamps) if timestamps else []
        if declared not in TIME_SAMPLING_VALUES:
            return [
                f"'acquisition.time_sampling' is '{declared}', which is not one of "
                f"{sorted(TIME_SAMPLING_VALUES)}."
            ]
        if declared != "variable":
            return self._timestamps_errors(timestamps) if timestamps else []

        errors: list[str] = []
        if not timestamps:
            errors.append(
                "'acquisition.time_sampling' is 'variable' but "
                "'acquisition.timestamps' is absent. Declaring variable sampling MUST "
                "carry the per-frame timestamps it points at, otherwise the claim "
                "cannot be checked."
            )
        else:
            errors.extend(self._timestamps_errors(timestamps))

        t_scale = self.axis_scales.get("t")
        if t_scale is not None and t_scale != 1.0:
            errors.append(
                f"'acquisition.time_sampling' is 'variable' but the level-0 't' scale "
                f"is {t_scale}. It MUST be 1, so the axis carries frame indices "
                "rather than a uniform duration the data does not have."
            )
        return errors

    # ------------------------------------------------------------------
    # channel-metadata.rst § Location in Zarr store — MUST
    # The 'dca' key must be present.
    # Use @field_validator (not @model_validator mode="after") so this runs
    # independently of whether nested 'levels' validation succeeds.
    # ------------------------------------------------------------------

    @field_validator("dca", mode="before")
    @classmethod
    def check_dca_present(
        cls, v: "DCAGroupMetadataV0_2 | dict | None"
    ) -> "DCAGroupMetadataV0_2 | dict | None":
        if v is None:
            raise ValueError(
                "Missing 'dca' key in zarr.json. "
                "DCA v0.2 MUST include a 'dca' object containing "
                "version, url, channels, and normalization_statistics "
                "(schema.json dca.required)."
            )
        return v

    # ------------------------------------------------------------------
    # channel-metadata.rst § Location in Zarr store — SHOULD
    # channels SHOULD include metadata for each channel in the image.
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_channel_count_matches_c_axis(self) -> "DCAStoreSpecV0_2":
        if self.dca is None or not self.levels:
            return self
        c_axis_size = self.levels[0].shape[1]  # C is index 1 in [t, c, z, y, x]
        n_channel_entries = len(self.dca.channels)
        if n_channel_entries != c_axis_size:
            suffix = "y" if n_channel_entries == 1 else "ies"
            raise ValueError(
                f"[SHOULD] dca.channels has {n_channel_entries} entr{suffix} "
                f"but the C axis at level 0 has size {c_axis_size}. "
                f"channels SHOULD cover all {c_axis_size} channels "
                f"(channel-metadata.rst § Location in Zarr store)."
            )
        return self


# ---------------------------------------------------------------------------
# Label store-level model
# ---------------------------------------------------------------------------


class DCALabelStoreSpecV0_2(BaseModel):
    """
    Root DCA v0.2 label image model. Validates a label image group (IMAGE_LABEL node).

    Call DCALabelStoreSpecV0_2.model_validate(data) where data is the dict
    produced by validation.validator._build_node_dict() extended with:
        spec_version — "0.2"

    Differences from DCAStoreSpecV0_2:
      - No 'dca' field; label groups do not carry channel metadata.
      - Uses DCALabelScaleLevelSpecV0_2 for per-level validation, which
        overrides the dtype SHOULD to require uint32 instead of uint8/uint16.
      - Shares axes, coarse-level, and downsampling rules with the image store.
    """

    model_config = ConfigDict(extra="allow")

    spec_version: Literal["0.2"]
    axes: list[str]
    levels: list[DCALabelScaleLevelSpecV0_2]

    # ------------------------------------------------------------------
    # array-standard.rst § Dimensions — MUST
    # ------------------------------------------------------------------

    @field_validator("axes")
    @classmethod
    def axes_must_be_tczyx(cls, v: list[str]) -> list[str]:
        if v != AXES_REQUIRED:
            raise ValueError(
                f"Axes must be {AXES_REQUIRED}; got {v}. "
                "All datasets must be 5D with dimensions in t, c, z, y, x order."
            )
        return v

    # ------------------------------------------------------------------
    # array-standard.rst § Multiresolution — MUST
    # At least one level must have every spatial dimension ≤ 2048.
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_coarse_level_exists(self) -> "DCALabelStoreSpecV0_2":
        for level in self.levels:
            _t, _c, z, y, x = level.shape
            if (
                z <= COARSE_LEVEL_MAX_SPATIAL
                and y <= COARSE_LEVEL_MAX_SPATIAL
                and x <= COARSE_LEVEL_MAX_SPATIAL
            ):
                return self
        shapes = [(lv.path, lv.shape) for lv in self.levels]
        raise ValueError(
            f"No scale level has all spatial dimensions ≤ "
            f"{COARSE_LEVEL_MAX_SPATIAL}. "
            f"MUST include a level-of-detail where every spatial dim ≤ "
            f"{COARSE_LEVEL_MAX_SPATIAL} "
            f"(array-standard.rst § Multiresolution). Level shapes: {shapes}"
        )

    # ------------------------------------------------------------------
    # array-standard.rst § Multiresolution — MUST (t/c), SHOULD (spatial)
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_downsampling_factors(self) -> "DCALabelStoreSpecV0_2":
        if len(self.levels) < 2:
            return self

        must_errors: list[str] = []
        should_warnings: list[str] = []

        for i in range(len(self.levels) - 1):
            fine = self.levels[i].shape
            coarse = self.levels[i + 1].shape
            chunks = self.levels[i].chunk_shape

            for dim_idx, dim_name in enumerate(AXES_REQUIRED):
                fine_dim = fine[dim_idx]
                coarse_dim = coarse[dim_idx]
                chunk_dim = chunks[dim_idx]

                if coarse_dim == 0:
                    continue

                factor = fine_dim / coarse_dim

                if dim_name in NON_SPATIAL_AXES:
                    if fine_dim != coarse_dim:
                        must_errors.append(
                            f"Level {i}→{i+1}: downsampling factor for '{dim_name}' "
                            f"must be 1 (got {fine_dim}/{coarse_dim}={factor:.2f})"
                        )
                else:
                    # Downsampling legitimately stops once an axis reaches its chunk
                    # floor, so skip when fine_dim is already <= its chunk size.
                    if fine_dim > chunk_dim and not math.isclose(
                        factor, DOWNSAMPLING_FACTOR_RECOMMENDED, rel_tol=0.05
                    ):
                        should_warnings.append(
                            f"[SHOULD] Level {i}→{i+1}: downsampling factor for '{dim_name}' "
                            f"should be {DOWNSAMPLING_FACTOR_RECOMMENDED} "
                            f"(got {fine_dim}/{coarse_dim}≈{factor:.2f})"
                        )

        all_issues = must_errors + should_warnings
        if all_issues:
            raise ValueError("\n".join(all_issues))

        return self
