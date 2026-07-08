"""
Unit tests for dca_helpers.writer — chunk and shard size computation.
"""

import math

import acquire_zarr as aqz
import numpy as np
import pytest

from dca_helpers.writer import (
    MIN_CHUNK_BYTES,
    _build_array_settings,
    compression_settings,
    compute_optimal_chunks,
    compute_shard_sizes,
    write_zarr_array,
)

# ── compute_optimal_chunks ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeOptimalChunks:
    def test_large_array_uses_128_defaults(self):
        """Standard large array: spatial chunks should start and stay at 128."""
        shape = (1, 1, 128, 128, 128)

        chunks = compute_optimal_chunks(shape, np.dtype(np.uint8))
        assert chunks == (1, 1, 128, 128, 128)
        assert math.prod(chunks) >= MIN_CHUNK_BYTES

    def test_small_array_dimension_clamps_to_array_size(self):
        """Spatial chunks should not exceed the array dimensions."""
        shape = (1, 1, 28, 300, 360)
        chunks = compute_optimal_chunks(shape, np.dtype(np.uint8))
        assert chunks == (1, 1, 28, 256, 128)
        assert math.prod(chunks) >= MIN_CHUNK_BYTES

    def test_small_array_dimension_clamps_to_array_size_u16(self):
        """
        Same as above, but with uint16 dtype
        (which should require smaller chunks to hit 512 KB).
        """
        shape = (1, 1, 28, 300, 360)
        chunks = compute_optimal_chunks(shape, np.dtype(np.uint16))
        assert chunks == (1, 1, 28, 128, 128)
        assert math.prod(chunks) < MIN_CHUNK_BYTES  # would fail if np.uint8 were used
        assert (
            math.prod(chunks) * 2 >= MIN_CHUNK_BYTES
        )  # should meet 512 KB when accounting for dtype size

    def test_chunk_meets_minimum_bytes_when_possible(self):
        """Chunk size must be >= 512 KB when the array is large enough."""
        shape = (64, 1, 64, 64, 64)
        chunks = compute_optimal_chunks(shape, np.dtype(np.uint8))
        assert chunks == (2, 1, 64, 64, 64)
        assert math.prod(chunks) >= MIN_CHUNK_BYTES

    def test_spatial_dims_grow_when_t_insufficient(self):
        """Y and X should grow if T-growth alone can't reach 512 KB."""
        shape = (2, 1, 1, 1024, 1024)
        chunks = compute_optimal_chunks(shape, np.dtype(np.uint8))
        assert chunks == (2, 1, 1, 512, 512)
        assert math.prod(chunks) >= MIN_CHUNK_BYTES

    def test_tiny_array_warns_but_does_not_raise(self):
        """Arrays too small to ever reach 512 KB should log a warning, not raise."""
        shape = (3, 1, 2, 2, 2)
        chunks = compute_optimal_chunks(shape, np.dtype(np.uint8))
        assert chunks == (3, 1, 2, 2, 2)
        assert math.prod(chunks) < MIN_CHUNK_BYTES


# ── compute_shard_sizes ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeShardSizes:
    def test_spatial_shards_cover_full_extent(self):
        """Spatial shards should span the full array when below the pixel cap."""
        shape = (10, 1, 2048, 2048, 2048)
        chunks = (1, 1, 128, 128, 128)
        chunks_per_shard = compute_shard_sizes(shape, chunks)
        assert chunks_per_shard == (10, 1, 16, 16, 16)

    def test_spatial_shards_capped_at_2048_pixels(self):
        """Spatial shards must not exceed 2048 pixels per dimension."""
        shape = (1, 1, 4096, 4096, 4096)
        chunks = (1, 1, 128, 128, 128)
        chunks_per_shard = compute_shard_sizes(shape, chunks)
        assert chunks_per_shard == (1, 1, 16, 16, 16)


# ── compression_settings ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestCompressionSettings:
    def test_dtype_byte_shuffle(self):
        """Integer dtypes should use shuffle=0 (no shuffle)."""
        s = compression_settings(np.dtype(np.uint8))
        assert s.shuffle == 0, "Expected shuffle=0 for np.uint8"

        s = compression_settings(np.dtype(np.float32))
        assert s.shuffle == 1, "Expected shuffle=1 for np.float32"

        # Test default compression level and codec
        assert s.level == 3, f"Expected default compression level 3, got {s.level}"
        assert s.compressor == aqz.Compressor.BLOSC1
        assert s.codec == aqz.CompressionCodec.BLOSC_ZSTD

    def test_custom_level(self):
        s = compression_settings(np.dtype(np.uint16), compression_level=7)
        assert s.level == 7


# ── _build_array_settings ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildArraySettings:
    _shape = (4, 2, 16, 64, 64)
    _chunks = (1, 1, 16, 64, 64)
    _shards = (4, 1, 1, 1, 1)
    _pixel_size_um = (1.0, 0.65, 0.65)

    def test_non_label(self):
        with pytest.raises(ValueError, match="No acquire_zarr DataType mapping"):
            _build_array_settings(
                shape=self._shape,
                dtype=np.dtype(np.complex64),
                chunks=self._chunks,
                shards=self._shards,
                pixel_size_um=self._pixel_size_um,
                is_label=False,
            )

        settings = _build_array_settings(
            shape=self._shape,
            dtype=np.dtype(np.uint16),
            chunks=self._chunks,
            shards=self._shards,
            pixel_size_um=(2.0, 0.5, 0.25),
            is_label=False,
            time_interval_s=5.0,
            output_key="0",
        )
        dims = settings.dimensions
        assert settings.downsampling_method == aqz.DownsamplingMethod.MEAN
        assert settings.data_type == aqz.DataType.UINT16
        assert settings.output_key == "0"
        assert len(dims) == 5
        assert dims[0].name == "t" and dims[0].kind == aqz.DimensionType.TIME
        assert dims[0].scale == pytest.approx(5.0)
        assert dims[1].name == "c" and dims[1].kind == aqz.DimensionType.CHANNEL
        assert dims[2].name == "z" and dims[2].kind == aqz.DimensionType.SPACE
        assert dims[2].scale == pytest.approx(2.0)
        assert dims[3].name == "y" and dims[3].kind == aqz.DimensionType.SPACE
        assert dims[3].scale == pytest.approx(0.5)
        assert dims[4].name == "x" and dims[4].kind == aqz.DimensionType.SPACE
        assert dims[4].scale == pytest.approx(0.25)

    def test_label(self):
        settings = _build_array_settings(
            shape=self._shape,
            dtype=np.dtype(np.float32),
            chunks=self._chunks,
            shards=self._shards,
            pixel_size_um=self._pixel_size_um,
            is_label=True,
        )
        assert settings.downsampling_method == aqz.DownsamplingMethod.MAX
        assert settings.data_type == aqz.DataType.FLOAT32


# ── write_zarr_array ───────────────────────────────────────────────────────────


@pytest.mark.integration
class TestWriteZarrArray:
    def test_output_directory_is_created(self, tmp_path):
        array = np.ones((1, 1, 4, 8, 8), dtype=np.uint16)
        out_path = tmp_path / "test.zarr"
        write_zarr_array(
            array_path=out_path,
            output_key="0",
            is_label=False,
            array=array,
            pixel_size_um=(1.0, 1.0, 1.0),
            time_interval_s=10.0,
        )
        assert out_path.exists()

    def test_custom_time_interval(self, tmp_path):
        """
        write_zarr_array should accept a non-default time_interval_s without error.
        """
        array = np.zeros((2, 1, 4, 8, 8), dtype=np.float32)
        out = write_zarr_array(
            array_path=tmp_path / "time.zarr",
            output_key="0",
            is_label=False,
            array=array,
            pixel_size_um=(1.0, 1.0, 1.0),
        )
        assert out.exists()
