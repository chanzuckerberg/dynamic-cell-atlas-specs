"""
Pydantic models for Dynamic Cell Atlas spec v0.1.

Each model encodes exactly the rules from docs/v0.1/array-standard.rst.

MUST violations → raise ValueError without a prefix → classified as ERROR by the runner.
SHOULD violations → raise ValueError prefixed "[SHOULD]" → classified as WARNING by the runner.

The root model DCAStoreSpecV0_1 is the single entry point:

    DCAStoreSpecV0_1.model_validate(data)

where `data` is a dict produced by validation.validator._build_node_dict().
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants from the spec
# ---------------------------------------------------------------------------

CHUNK_MIN_BYTES = 512 * 1024  # 512 KB  — MUST
CHUNK_REC_BYTES = 1024 * 1024  # 1 MB    — SHOULD

SHARD_MIN_BYTES = 1024**3  # 1 GB    — MUST (when array >= 1 GB)
SHARD_MAX_BYTES = 5 * 1024**4  # 5 TB    — MUST
SHARD_REC_BYTES = 5 * 1024**3  # 5 GB    — SHOULD

ALLOWED_CODECS = frozenset({"zstd", "lz4", "blosc"})
RECOMMENDED_CODEC = "zstd"
RECOMMENDED_LEVEL_MAX = 3

AXES_REQUIRED = ["t", "c", "z", "y", "x"]
SPATIAL_AXES = {"z", "y", "x"}
NON_SPATIAL_AXES = {"t", "c"}

C_CHUNK_RECOMMENDED = 1  # SHOULD — channel dims
SPATIAL_SHARD_MAX_RECOMMENDED = 2048  # SHOULD — each spatial shard dim
TIME_SHARD_MIN_RECOMMENDED = 16  # SHOULD — time shard dim

RAW_DTYPES_RECOMMENDED = frozenset({"uint8", "uint16"})
LABEL_DTYPE_RECOMMENDED = "uint32"

MULTISCALE_MIN_LEVELS = 3  # MUST
DOWNSAMPLING_FACTOR_RECOMMENDED = 2  # SHOULD — spatial dims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nbytes(shape: list[int], dtype: str) -> int:
    return math.prod(shape) * np.dtype(dtype).itemsize


# ---------------------------------------------------------------------------
# Per-level model
# ---------------------------------------------------------------------------


class DCAScaleLevelSpec(BaseModel):
    """
    Validates one scale level (one Zarr array) inside a DCA multiscale store.

    Fields populated by validation.validator._build_node_dict():
        level_index — position in the multiscale pyramid (0 = base / highest
                      resolution, > 0 = downsampled). NGFF v0.5 guarantees
                      multiscales.datasets is ordered highest→lowest resolution.
        path        — relative path within the store, e.g. "0", "1"
        dtype       — NumPy dtype string, e.g. "uint8"
        shape       — array shape as [t, c, z, y, x]
        chunk_shape — inner chunk shape as [t, c, z, y, x]
        shard_shape — outer shard shape as [t, c, z, y, x], or None
        codec_id    — lowercase codec name, e.g. "zstd"
        codec_level — compression level integer, or None
    """

    model_config = ConfigDict(extra="allow")

    level_index: int = 0
    path: str
    dtype: str
    shape: list[int]
    chunk_shape: list[int]
    shard_shape: list[int] | None = None
    codec_id: str
    codec_level: int | None = None

    # ------------------------------------------------------------------
    # Compression — MUST
    # ------------------------------------------------------------------

    @field_validator("codec_id")
    @classmethod
    def codec_must_be_allowed(cls, v: str) -> str:
        normalized = v.lower()
        if normalized not in ALLOWED_CODECS:
            raise ValueError(
                f"Codec '{v}' is not permitted; must be one of {sorted(ALLOWED_CODECS)}"
            )
        return normalized

    # ------------------------------------------------------------------
    # Chunk size — MUST >= 512 KB, SHOULD >= 1 MB
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_chunk_size(self) -> "DCAScaleLevelSpec":
        # Chunk-size rules apply only to the base resolution level.
        # See array-standard.rst § Chunk Size / Edge Cases.
        if self.level_index > 0:
            return self

        channel_idx = AXES_REQUIRED.index("c")

        nbytes = _nbytes(self.chunk_shape, self.dtype)
        # When comparing chunk size with array size, ignore the channel dimension
        # because chunk sizing should not be driven by channel count.
        chunk_nbytes = _nbytes(
            self.chunk_shape[:channel_idx] + self.chunk_shape[channel_idx + 1 :],
            self.dtype,
        )
        total_nbytes = _nbytes(
            self.shape[:channel_idx] + self.shape[channel_idx + 1 :], self.dtype
        )
        kb = nbytes / 1024

        # Only enforce min chunk size if:
        # chunk(minus channel) is smaller than the whole array (minus channel)
        if nbytes < CHUNK_MIN_BYTES and chunk_nbytes < total_nbytes:
            raise ValueError(
                f"Uncompressed chunk size {kb:.0f} KB is below the required 512 KB minimum "
                f"(level={self.path!r}, chunk_shape={self.chunk_shape}, dtype={self.dtype}). "
                "See array-standard.rst § Chunk Size edge cases."
            )

        if nbytes < CHUNK_REC_BYTES and chunk_nbytes < total_nbytes:
            raise ValueError(
                f"[SHOULD] Uncompressed chunk size {kb:.0f} KB is below the recommended "
                f"1 MB (level={self.path!r}, chunk_shape={self.chunk_shape}, dtype={self.dtype})"
            )

        return self

    # ------------------------------------------------------------------
    # Chunk shape recommendations — SHOULD
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_chunk_shape_recommendations(self) -> "DCAScaleLevelSpec":
        c = self.chunk_shape[1]  # channel dim (axes order [t, c, z, y, x])

        warnings: list[str] = []
        if c != C_CHUNK_RECOMMENDED:
            warnings.append(
                f"[SHOULD] Channel chunk size should be {C_CHUNK_RECOMMENDED}; got {c}"
            )
        if warnings:
            raise ValueError("\n".join(warnings))

        return self

    # ------------------------------------------------------------------
    # Shard size — MUST < 5 TB; MUST >= 1 GB when array >= 1 GB
    #              SHOULD < 5 GB; SHOULD spatial <= 2048; SHOULD time >= 16
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_shard_size(self) -> "DCAScaleLevelSpec":
        if self.shard_shape is None or self.level_index > 0:
            return self

        shard_bytes = _nbytes(self.shard_shape, self.dtype)
        array_bytes = _nbytes(self.shape, self.dtype)

        # MUST: < 5 TB
        if shard_bytes >= SHARD_MAX_BYTES:
            raise ValueError(
                f"Uncompressed shard size {shard_bytes / 1024**4:.1f} TB must be < 5 TB"
            )

        # MUST: >= 1 GB when array itself >= 1 GB
        if array_bytes >= SHARD_MIN_BYTES and shard_bytes < SHARD_MIN_BYTES:
            raise ValueError(
                f"Uncompressed shard size {shard_bytes / 1024**3:.2f} GB must be >= 1 GB "
                f"for arrays >= 1 GB (array is {array_bytes / 1024**3:.2f} GB)"
            )

        warnings: list[str] = []

        # SHOULD: < 5 GB
        if shard_bytes > SHARD_REC_BYTES:
            warnings.append(
                f"[SHOULD] Uncompressed shard size {shard_bytes / 1024**3:.1f} GB "
                f"exceeds the recommended 5 GB"
            )

        # SHOULD: spatial shard dims <= 2048
        _t, _c, z, y, x = self.shard_shape
        for dim_name, dim_val in (("z", z), ("y", y), ("x", x)):
            if dim_val > SPATIAL_SHARD_MAX_RECOMMENDED:
                warnings.append(
                    f"[SHOULD] Shard size in {dim_name} ({dim_val}) exceeds "
                    f"the recommended maximum of {SPATIAL_SHARD_MAX_RECOMMENDED}"
                )

        # SHOULD: time shard >= 16, but only when the array has >= 16 timepoints
        if (
            _t < TIME_SHARD_MIN_RECOMMENDED
            and self.shape[0] >= TIME_SHARD_MIN_RECOMMENDED
        ):
            warnings.append(
                f"[SHOULD] Time shard size ({_t}) is below the recommended "
                f"minimum of {TIME_SHARD_MIN_RECOMMENDED}"
            )

        if warnings:
            raise ValueError("\n".join(warnings))

        return self

    # ------------------------------------------------------------------
    # Compression recommendations — SHOULD
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_compression_recommendation(self) -> "DCAScaleLevelSpec":
        warnings: list[str] = []
        dtype_kind = np.dtype(
            self.dtype
        ).kind  # 'u' = unsigned int, 'i' = signed int, 'f' = float

        if dtype_kind in ("u", "i"):
            # Integers: SHOULD level <= 3
            if (
                self.codec_id == RECOMMENDED_CODEC
                and self.codec_level is not None
                and self.codec_level > RECOMMENDED_LEVEL_MAX
            ):
                warnings.append(
                    f"[SHOULD] Compression level {self.codec_level} exceeds the recommended "
                    f"maximum of {RECOMMENDED_LEVEL_MAX}"
                    " (high levels increase decompression overhead)"
                )

        elif dtype_kind == "f":
            # Floats: SHOULD use byte shuffle + zstd (not blosc), level <= 3
            if self.codec_id == "blosc":
                warnings.append(
                    "[SHOULD] Float data should use byte shuffle + zstd (not blosc)"
                )
            if (
                self.codec_id == RECOMMENDED_CODEC
                and self.codec_level is not None
                and self.codec_level > RECOMMENDED_LEVEL_MAX
            ):
                warnings.append(
                    f"[SHOULD] Compression level {self.codec_level} exceeds the recommended "
                    f"maximum of {RECOMMENDED_LEVEL_MAX}"
                )

        if warnings:
            raise ValueError("\n".join(warnings))

        return self

    # ------------------------------------------------------------------
    # Data type recommendations — SHOULD
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_dtype_recommendation(self) -> "DCAScaleLevelSpec":
        normalized = str(np.dtype(self.dtype))
        if (
            normalized not in RAW_DTYPES_RECOMMENDED
            and normalized != LABEL_DTYPE_RECOMMENDED
        ):
            raise ValueError(
                f"[SHOULD] dtype '{self.dtype}' is unusual; raw arrays should use "
                f"uint8 or uint16, label arrays should use uint32"
            )
        return self


# ---------------------------------------------------------------------------
# Store-level model (root entry point)
# ---------------------------------------------------------------------------


class DCAStoreSpecV0_1(BaseModel):
    """
    Root DCA v0.1 model. Validates a full multiscale Zarr store.

    Call DCAStoreSpecV0_1.model_validate(data) where data is the dict
    produced by validation.validator._build_node_dict().
    """

    model_config = ConfigDict(extra="allow")

    spec_version: Literal["0.1"]
    axes: list[str]
    multiscale_level_count: int
    levels: list[DCAScaleLevelSpec]

    # ------------------------------------------------------------------
    # Axes — MUST be ["t", "c", "z", "y", "x"]
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
    # Multiresolution level count — MUST >= 3
    # ------------------------------------------------------------------

    @field_validator("multiscale_level_count")
    @classmethod
    def must_have_minimum_scales(cls, v: int) -> int:
        if v < MULTISCALE_MIN_LEVELS:
            raise ValueError(
                f"At least {MULTISCALE_MIN_LEVELS} spatial scales are required; found {v}"
            )
        return v

    # ------------------------------------------------------------------
    # Downsampling factors across levels — MUST be 1 for t/c, SHOULD be 2 for spatial
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_downsampling_factors(self) -> "DCAStoreSpecV0_1":
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

                # Avoid divide-by-zero for degenerate dims
                if coarse_dim == 0:
                    continue

                factor = fine_dim / coarse_dim

                if dim_name in NON_SPATIAL_AXES:
                    # MUST: t and c must not be downsampled
                    if fine_dim != coarse_dim:
                        must_errors.append(
                            f"Level {i}→{i+1}: downsampling factor for '{dim_name}' "
                            f"must be 1 (got {fine_dim}/{coarse_dim}={factor:.2f})"
                        )
                else:
                    # SHOULD: spatial dims should have factor 2. Downsampling
                    # legitimately stops once an axis reaches its chunk floor, so
                    # skip when fine_dim is already <= its chunk size.
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
