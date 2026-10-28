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

import logging
import math
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails

logger = logging.getLogger(__name__)


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
AXES_ACCEPTED = (AXES_REQUIRED, [a.upper() for a in AXES_REQUIRED])
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


def raw_rule_messages(rule: Callable[..., list[str]], *args: Any) -> list[str]:
    try:
        return rule(*args)
    except Exception:
        logger.warning("Raw-input rule %s failed", rule.__name__, exc_info=True)
        return []


def merge_validation_error(
    exc: ValidationError, data: object, messages: list[str]
) -> ValidationError:
    lines: list[InitErrorDetails] = [
        {
            "type": error["type"],
            "loc": error["loc"],
            "input": error["input"],
            "ctx": error.get("ctx", {}),
        }
        for error in exc.errors(include_url=False)
    ]
    lines += [
        {
            "type": "value_error",
            "loc": (),
            "input": data,
            "ctx": {"error": ValueError(message)},
        }
        for message in messages
    ]
    return ValidationError.from_exception_data(exc.title, lines)


# ---------------------------------------------------------------------------
# Array rules — plain functions of any object with the level fields
# ---------------------------------------------------------------------------


# ------------------------------------------------------------------
# Chunk size — MUST >= 512 KB, SHOULD >= 1 MB
# ------------------------------------------------------------------


def chunk_size_messages(level: Any) -> list[str]:
    # Chunk-size rules apply only to the base resolution level.
    # See array-standard.rst § Chunk Size / Edge Cases.
    if level.level_index > 0:
        return []

    channel_idx = AXES_REQUIRED.index("c")

    nbytes = _nbytes(level.chunk_shape, level.dtype)
    # When comparing chunk size with array size, ignore the channel dimension
    # because chunk sizing should not be driven by channel count.
    chunk_nbytes = _nbytes(
        level.chunk_shape[:channel_idx] + level.chunk_shape[channel_idx + 1 :],
        level.dtype,
    )
    total_nbytes = _nbytes(
        level.shape[:channel_idx] + level.shape[channel_idx + 1 :], level.dtype
    )
    kb = nbytes / 1024

    # Only enforce min chunk size if:
    # chunk(minus channel) is smaller than the whole array (minus channel)
    if nbytes < CHUNK_MIN_BYTES and chunk_nbytes < total_nbytes:
        return [
            f"Uncompressed chunk size {kb:.0f} KB is below the required 512 KB minimum "
            f"(level={level.path!r}, chunk_shape={level.chunk_shape}, dtype={level.dtype}). "
            "See array-standard.rst § Chunk Size edge cases."
        ]

    if nbytes < CHUNK_REC_BYTES and chunk_nbytes < total_nbytes:
        return [
            f"[SHOULD] Uncompressed chunk size {kb:.0f} KB is below the recommended "
            f"1 MB (level={level.path!r}, chunk_shape={level.chunk_shape}, dtype={level.dtype})"
        ]

    return []


# ------------------------------------------------------------------
# Chunk shape recommendations — SHOULD
# ------------------------------------------------------------------


def chunk_shape_messages(level: Any) -> list[str]:
    c = level.chunk_shape[1]  # channel dim (axes order [t, c, z, y, x])

    if c != C_CHUNK_RECOMMENDED:
        return [f"[SHOULD] Channel chunk size should be {C_CHUNK_RECOMMENDED}; got {c}"]
    return []


# ------------------------------------------------------------------
# Shard size — MUST < 5 TB; MUST >= 1 GB when array >= 1 GB
#              SHOULD < 5 GB; SHOULD spatial <= 2048; SHOULD time >= 16
# ------------------------------------------------------------------


def shard_size_messages(level: Any) -> list[str]:
    if level.shard_shape is None or level.level_index > 0:
        return []

    shard_bytes = _nbytes(level.shard_shape, level.dtype)
    array_bytes = _nbytes(level.shape, level.dtype)

    errors: list[str] = []

    # MUST: < 5 TB
    if shard_bytes >= SHARD_MAX_BYTES:
        errors.append(
            f"Uncompressed shard size {shard_bytes / 1024**4:.1f} TB must be < 5 TB"
        )

    # MUST: >= 1 GB when array itself >= 1 GB
    if array_bytes >= SHARD_MIN_BYTES and shard_bytes < SHARD_MIN_BYTES:
        errors.append(
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
    _t, _c, z, y, x = level.shard_shape
    for dim_name, dim_val in (("z", z), ("y", y), ("x", x)):
        if dim_val > SPATIAL_SHARD_MAX_RECOMMENDED:
            warnings.append(
                f"[SHOULD] Shard size in {dim_name} ({dim_val}) exceeds "
                f"the recommended maximum of {SPATIAL_SHARD_MAX_RECOMMENDED}"
            )

    # SHOULD: time shard >= 16, but only when the array has >= 16 timepoints
    if _t < TIME_SHARD_MIN_RECOMMENDED and level.shape[0] >= TIME_SHARD_MIN_RECOMMENDED:
        warnings.append(
            f"[SHOULD] Time shard size ({_t}) is below the recommended "
            f"minimum of {TIME_SHARD_MIN_RECOMMENDED}"
        )

    return errors + warnings


# ------------------------------------------------------------------
# Compression recommendations — SHOULD
# ------------------------------------------------------------------


def compression_messages(level: Any) -> list[str]:
    warnings: list[str] = []
    dtype_kind = np.dtype(
        level.dtype
    ).kind  # 'u' = unsigned int, 'i' = signed int, 'f' = float

    if dtype_kind in ("u", "i"):
        # Integers: SHOULD level <= 3
        if (
            level.codec_id == RECOMMENDED_CODEC
            and level.codec_level is not None
            and level.codec_level > RECOMMENDED_LEVEL_MAX
        ):
            warnings.append(
                f"[SHOULD] Compression level {level.codec_level} exceeds the recommended "
                f"maximum of {RECOMMENDED_LEVEL_MAX}"
                " (high levels increase decompression overhead)"
            )

    elif dtype_kind == "f":
        # Floats: SHOULD use byte shuffle + zstd (not blosc), level <= 3
        if level.codec_id == "blosc":
            warnings.append(
                "[SHOULD] Float data should use byte shuffle + zstd (not blosc)"
            )
        if (
            level.codec_id == RECOMMENDED_CODEC
            and level.codec_level is not None
            and level.codec_level > RECOMMENDED_LEVEL_MAX
        ):
            warnings.append(
                f"[SHOULD] Compression level {level.codec_level} exceeds the recommended "
                f"maximum of {RECOMMENDED_LEVEL_MAX}"
            )

    return warnings


# ------------------------------------------------------------------
# Data type recommendations — SHOULD
# ------------------------------------------------------------------


def dtype_messages(level: Any) -> list[str]:
    normalized = str(np.dtype(level.dtype))
    if (
        normalized not in RAW_DTYPES_RECOMMENDED
        and normalized != LABEL_DTYPE_RECOMMENDED
    ):
        return [
            f"[SHOULD] dtype '{level.dtype}' is unusual; raw arrays should use "
            f"uint8 or uint16, label arrays should use uint32"
        ]
    return []


def array_rule_messages(
    level: Any, dtype_rule: Callable[[Any], list[str]] = dtype_messages
) -> list[str]:
    return (
        chunk_size_messages(level)
        + chunk_shape_messages(level)
        + shard_size_messages(level)
        + compression_messages(level)
        + dtype_rule(level)
    )


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
    # Dimensions — MUST be 5, in [t, c, z, y, x] order
    # ------------------------------------------------------------------

    @field_validator("shape", "chunk_shape", "shard_shape")
    @classmethod
    def array_must_be_5d(
        cls, v: list[int] | None, info: ValidationInfo
    ) -> list[int] | None:
        if v is not None and len(v) != len(AXES_REQUIRED):
            raise ValueError(
                f"{info.field_name} has {len(v)} dimensions; arrays MUST have "
                f"{len(AXES_REQUIRED)} dimensions in {AXES_REQUIRED} order "
                "(array-standard.rst § Dimensions)."
            )
        return v

    @field_validator("dtype")
    @classmethod
    def dtype_must_be_numpy(cls, v: str) -> str:
        try:
            np.dtype(v)
        except (TypeError, ValueError):
            raise ValueError(f"dtype '{v}' is not a valid NumPy dtype.") from None
        return v

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

    @staticmethod
    def dtype_rule(level: Any) -> list[str]:
        return dtype_messages(level)

    @model_validator(mode="wrap")
    @classmethod
    def collect_array_rules_on_field_failure(cls, data: Any, handler: Any) -> Any:
        try:
            return handler(data)
        except ValidationError as exc:
            messages = raw_rule_messages(raw_array_rule_messages, data, cls.dtype_rule)
            if not messages:
                raise
            raise merge_validation_error(exc, data, messages) from None

    # ------------------------------------------------------------------
    # One validator for every array rule. Pydantic stops the after-validator
    # chain at the first raise, so separate validators would let an earlier
    # SHOULD hide a later MUST.
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_array_rules(self) -> "DCAScaleLevelSpec":
        messages = array_rule_messages(self, type(self).dtype_rule)
        if messages:
            raise ValueError("\n".join(messages))
        return self


def downsampling_factor_messages(levels: Sequence[Any]) -> list[str]:
    if len(levels) < 2:
        return []

    must_errors: list[str] = []
    should_warnings: list[str] = []

    for i in range(len(levels) - 1):
        fine = levels[i].shape
        coarse = levels[i + 1].shape
        chunks = levels[i].chunk_shape

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

    return must_errors + should_warnings


def _is_int_list(value: object, length: int = 5) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    )


def raw_shape_view(data: object) -> SimpleNamespace | None:
    if not isinstance(data, dict):
        return None
    shape, chunk_shape = data.get("shape"), data.get("chunk_shape")
    if not (_is_int_list(shape) and _is_int_list(chunk_shape)):
        return None
    return SimpleNamespace(
        path=str(data.get("path", "")),
        shape=list(shape),
        chunk_shape=list(chunk_shape),
    )


def raw_shape_views(levels: object) -> list[SimpleNamespace] | None:
    if not isinstance(levels, list):
        return None
    views = [raw_shape_view(level) for level in levels]
    return None if any(view is None for view in views) else views


def raw_level_view(data: object) -> SimpleNamespace | None:
    view = raw_shape_view(data)
    if view is None:
        return None
    shard_shape, dtype, codec_id = (
        data.get("shard_shape"),
        data.get("dtype"),
        data.get("codec_id"),
    )
    level_index, codec_level = data.get("level_index", 0), data.get("codec_level")
    if not (
        isinstance(dtype, str)
        and isinstance(codec_id, str)
        and isinstance(level_index, int)
        and not isinstance(level_index, bool)
        and (shard_shape is None or _is_int_list(shard_shape))
    ):
        return None
    try:
        np.dtype(dtype)
    except (TypeError, ValueError):
        return None
    view.level_index = level_index
    view.dtype = dtype
    view.shard_shape = None if shard_shape is None else list(shard_shape)
    view.codec_id = codec_id.lower()
    view.codec_level = (
        codec_level
        if isinstance(codec_level, int) and not isinstance(codec_level, bool)
        else None
    )
    return view


def raw_array_rule_messages(
    data: object, dtype_rule: Callable[[Any], list[str]] = dtype_messages
) -> list[str]:
    view = raw_level_view(data)
    return [] if view is None else array_rule_messages(view, dtype_rule)


def raw_downsampling_messages(levels: object) -> list[str]:
    views = raw_shape_views(levels)
    return [] if views is None else downsampling_factor_messages(views)


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
        if v not in AXES_ACCEPTED:
            raise ValueError(
                f"Axes must be {AXES_REQUIRED} (all lowercase or all uppercase); "
                f"got {v}. "
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

    @model_validator(mode="wrap")
    @classmethod
    def collect_downsampling_on_field_failure(cls, data: Any, handler: Any) -> Any:
        try:
            return handler(data)
        except ValidationError as exc:
            levels = data.get("levels") if isinstance(data, dict) else None
            messages = raw_rule_messages(raw_downsampling_messages, levels)
            if not messages:
                raise
            raise merge_validation_error(exc, data, messages) from None

    # ------------------------------------------------------------------
    # Downsampling factors across levels — MUST be 1 for t/c, SHOULD be 2 for spatial
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def check_downsampling_factors(self) -> "DCAStoreSpecV0_1":
        messages = downsampling_factor_messages(self.levels)
        if messages:
            raise ValueError("\n".join(messages))
        return self
