"""
Compute spec-compliant write parameters for Dynamic Cell Atlas OME-Zarr arrays.
Writing is done via acquire-zarr. Other libraries may be added as needed.

Spec reference: docs/v0.1/array-standard.rst
"""

import logging
import math
from pathlib import Path

import acquire_zarr as aqz
import numpy as np

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

MIN_CHUNK_BYTES = 512 * 1024  # 512 KB minimum (spec § "Chunk Size")

_NUMPY_TO_AQZ_DTYPE = {
    np.dtype("uint8"): aqz.DataType.UINT8,
    np.dtype("uint16"): aqz.DataType.UINT16,
    np.dtype("uint32"): aqz.DataType.UINT32,
    np.dtype("uint64"): aqz.DataType.UINT64,
    np.dtype("int8"): aqz.DataType.INT8,
    np.dtype("int16"): aqz.DataType.INT16,
    np.dtype("int32"): aqz.DataType.INT32,
    np.dtype("int64"): aqz.DataType.INT64,
    np.dtype("float32"): aqz.DataType.FLOAT32,
    np.dtype("float64"): aqz.DataType.FLOAT64,
}  # Needed for `uint16` for some reason

MAX_SHARD_DIMENSION_PIXELS = (
    2048  # max pixels in any spatial dimension of a shard (spec § "Shard Size")
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _chunk_bytes(t: int, c: int, z: int, y: int, x: int, dtype_size: int) -> int:
    return t * c * z * y * x * dtype_size


# ── Public API ─────────────────────────────────────────────────────────────────


# Spec ref: docs/v0.1/array-standard.rst § "Chunk Size"
def compute_optimal_chunks(
    shape: tuple[int, int, int, int, int],
    dtype: np.dtype,
) -> tuple[int, int, int, int, int]:
    """
    Compute chunk sizes following the Dynamic Cell Atlas array standard.

    Args:
        shape: Array shape as (T, C, Z, Y, X).
        dtype: Array data type.

    Returns:
        Chunk sizes as (t, c, z, y, x).
    """
    dtype_size = dtype.itemsize
    t_dim, c_dim, z_dim, y_dim, x_dim = shape

    t_chunk = 1
    c_chunk = 1
    z_chunk = min(128, z_dim)
    y_chunk = min(128, y_dim)
    x_chunk = min(128, x_dim)

    current_size = _chunk_bytes(t_chunk, c_chunk, z_chunk, y_chunk, x_chunk, dtype_size)

    # If below minimum, grow spatial dims first (keep T=1, C=1 for independence)
    if current_size < MIN_CHUNK_BYTES:
        logger.debug("Chunk < 512 KB, growing time dims...")
        while current_size < MIN_CHUNK_BYTES and t_chunk < t_dim:
            t_chunk = min(t_chunk * 2, t_dim)
            current_size = _chunk_bytes(
                t_chunk, c_chunk, z_chunk, y_chunk, x_chunk, dtype_size
            )

        logger.debug("Chunk < 512 KB, growing spatial dims...")

        while current_size < MIN_CHUNK_BYTES:
            grew = False

            if y_chunk < y_dim:
                y_chunk = min(y_chunk * 2, y_dim)
                grew = True

            new_size = _chunk_bytes(
                t_chunk, c_chunk, z_chunk, y_chunk, x_chunk, dtype_size
            )

            if new_size < MIN_CHUNK_BYTES and x_chunk < x_dim:
                x_chunk = min(x_chunk * 2, x_dim)
                grew = True

            if not grew:
                break

            current_size = _chunk_bytes(
                t_chunk, c_chunk, z_chunk, y_chunk, x_chunk, dtype_size
            )

    final_size = _chunk_bytes(t_chunk, c_chunk, z_chunk, y_chunk, x_chunk, dtype_size)
    logger.info(
        "Chunk shape: (%d, %d, %d, %d, %d) = %.1f KB",
        t_chunk,
        c_chunk,
        z_chunk,
        y_chunk,
        x_chunk,
        final_size / 1024,
    )

    if final_size < MIN_CHUNK_BYTES:
        logger.warning("Could not reach 512 KB minimum (array too small)")

    return (t_chunk, c_chunk, z_chunk, y_chunk, x_chunk)


# Spec ref: docs/v0.1/array-standard.rst § "Shard Size"
def compute_shard_sizes(
    shape: tuple[int, int, int, int, int],
    chunks: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    """
    Compute shard sizes (in units of chunks) following the Dynamic Cell Atlas array standard.

    - Time shard size should be <= 16 (for efficient time slicing)
    - Channel shard size = 1 (for independent access)
    - Spatial shard size should equal array extent, but <= 2048 total pixels

    Args:
        shape:  Array shape as (T, C, Z, Y, X).
        chunks: Chunk sizes as (t, c, z, y, x).

    Returns:
        Shard sizes in chunks as (t, c, z, y, x).
    """
    t_chunk, c_chunk, z_chunk, y_chunk, x_chunk = chunks
    t_dim, c_dim, z_dim, y_dim, x_dim = shape

    t_shard = min(math.ceil(t_dim / t_chunk), 16)
    c_shard = 1
    z_shard = min(math.ceil(z_dim / z_chunk), MAX_SHARD_DIMENSION_PIXELS // z_chunk)
    y_shard = min(math.ceil(y_dim / y_chunk), MAX_SHARD_DIMENSION_PIXELS // y_chunk)
    x_shard = min(math.ceil(x_dim / x_chunk), MAX_SHARD_DIMENSION_PIXELS // x_chunk)

    logger.info(
        "Shard shape: (%d, %d, %d, %d, %d) chunks",
        t_shard,
        c_shard,
        z_shard,
        y_shard,
        x_shard,
    )

    return (t_shard, c_shard, z_shard, y_shard, x_shard)


# Spec ref: docs/v0.1/array-standard.rst § "Compression"
def compression_settings(
    dtype: np.dtype, compression_level: int = 3
) -> aqz.CompressionSettings:
    """Compute compression settings following the Dynamic Cell Atlas array standard."""
    return aqz.CompressionSettings(
        compressor=aqz.Compressor.BLOSC1,
        codec=aqz.CompressionCodec.BLOSC_ZSTD,
        level=compression_level,
        shuffle=1 if np.issubdtype(dtype, np.floating) else 0,
    )


def _build_array_settings(
    shape: tuple[int, int, int, int, int],
    dtype: np.dtype,
    chunks: tuple[int, int, int, int, int],
    shards: tuple[int, int, int, int, int],
    pixel_size_um: tuple[float, float, float],
    is_label: bool,
    time_interval_s: float = 1.0,
    output_key: str = "",
) -> aqz.ArraySettings:
    """Build acquire_zarr ArraySettings for an array."""

    aqz_dtype = _NUMPY_TO_AQZ_DTYPE.get(dtype)
    if aqz_dtype is None:
        raise ValueError(f"No acquire_zarr DataType mapping for numpy dtype: {dtype}")

    t_dim, c_dim, z_dim, y_dim, x_dim = shape
    t_chunk, c_chunk, z_chunk, y_chunk, x_chunk = chunks
    t_shard, c_shard, z_shard, y_shard, x_shard = shards
    z_um, y_um, x_um = pixel_size_um

    # Spec ref: docs/v0.1/array-standard.rst § "Multiresolution"
    ds_method = aqz.DownsamplingMethod.MAX if is_label else aqz.DownsamplingMethod.MEAN
    compression = compression_settings(dtype)
    return aqz.ArraySettings(
        output_key=output_key,
        data_type=aqz_dtype,
        dimensions=[
            aqz.Dimension(
                name="t",
                kind=aqz.DimensionType.TIME,
                array_size_px=t_dim,
                chunk_size_px=t_chunk,
                shard_size_chunks=t_shard,
                scale=time_interval_s,
                unit="second",
            ),
            aqz.Dimension(
                name="c",
                kind=aqz.DimensionType.CHANNEL,
                array_size_px=c_dim,
                chunk_size_px=c_chunk,
                shard_size_chunks=c_shard,
                scale=1.0,
            ),
            aqz.Dimension(
                name="z",
                kind=aqz.DimensionType.SPACE,
                array_size_px=z_dim,
                chunk_size_px=z_chunk,
                shard_size_chunks=z_shard,
                scale=z_um,
                unit="micrometer",
            ),
            aqz.Dimension(
                name="y",
                kind=aqz.DimensionType.SPACE,
                array_size_px=y_dim,
                chunk_size_px=y_chunk,
                shard_size_chunks=y_shard,
                scale=y_um,
                unit="micrometer",
            ),
            aqz.Dimension(
                name="x",
                kind=aqz.DimensionType.SPACE,
                array_size_px=x_dim,
                chunk_size_px=x_chunk,
                shard_size_chunks=x_shard,
                scale=x_um,
                unit="micrometer",
            ),
        ],
        compression=compression,
        downsampling_method=ds_method,
    )


def write_zarr_array(
    array_path: Path,
    output_key: str,
    is_label: bool,
    array: np.ndarray,
    pixel_size_um: tuple[float, float, float],
    time_interval_s: float = 1.0,
) -> Path:
    """A simple function to write a ome-zarr array.

    More complicated procedures may need to implement their own function.
    """

    chunks = compute_optimal_chunks(array.shape, array.dtype)
    shards = compute_shard_sizes(array.shape, chunks)
    image_settings = _build_array_settings(
        shape=array.shape,
        dtype=array.dtype,
        chunks=chunks,
        shards=shards,
        pixel_size_um=pixel_size_um,
        time_interval_s=time_interval_s,
        is_label=is_label,
        output_key=output_key,
    )
    arrays = [image_settings]
    settings = aqz.StreamSettings(
        store_path=str(array_path),
        overwrite=True,
        arrays=arrays,
    )
    settings.version = aqz.ZarrVersion.V3
    settings.multiscale = True  # type: ignore
    stream = aqz.ZarrStream(settings)

    t_dim, c_dim = array.shape[:2]
    total_blocks = t_dim * c_dim
    log_every = max(1, total_blocks // 10)

    logger.info(f"Writing {total_blocks} blocks...")
    for t in range(t_dim):
        for c in range(c_dim):
            block = array[t : t + 1, c : c + 1, :, :, :]
            stream.append(block, key=output_key)

            block_num = t * c_dim + c + 1
            if block_num % log_every == 0 or block_num == total_blocks:
                pct = block_num / total_blocks * 100
                logger.info(f"Progress: {block_num}/{total_blocks} ({pct:.1f}%)")

    stream.close()
    return array_path
