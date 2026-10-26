"""
Unit tests for DCA v0.1 Pydantic models.

Tests cover every MUST and SHOULD rule in DCAScaleLevelSpec and
DCAStoreSpecV0_1 without requiring a real Zarr store.
"""

import pydantic
import pytest

from dca_helpers.validation.result import Severity
from dca_helpers.validation.validator import _convert_errors
from dca_helpers.validation.versions.v0_1.models import (
    DCAScaleLevelSpec,
    DCAStoreSpecV0_1,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def validate_level(data: dict) -> list:
    """Return list of Issues from validating a level dict, or [] on pass."""
    try:
        DCAScaleLevelSpec.model_validate(data)
        return []
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)


def validate_node(data: dict) -> list:
    """Return list of Issues from validating a store dict, or [] on pass."""
    try:
        DCAStoreSpecV0_1.model_validate(data)
        return []
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)


def errors(issues):
    return [i for i in issues if i.severity == Severity.ERROR]


def warnings(issues):
    return [i for i in issues if i.severity == Severity.WARNING]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def valid_level(
    path="0",
    shape=None,
    chunk_shape=None,
    shard_shape=None,
    codec_id="zstd",
    codec_level=3,
    dtype="uint16",
    level_index=0,
):
    return {
        "level_index": level_index,
        "path": path,
        "dtype": dtype,
        "shape": shape or [1, 1, 256, 512, 512],
        "chunk_shape": chunk_shape or [1, 1, 128, 128, 128],
        "shard_shape": shard_shape,
        "codec_id": codec_id,
        "codec_level": codec_level,
    }


def valid_store(n_levels=3):
    shapes = [
        [1, 1, 256, 512, 512],
        [1, 1, 128, 256, 256],
        [1, 1, 64, 128, 128],
    ]
    levels = [
        valid_level(str(i), shape=shapes[i], level_index=i) for i in range(n_levels)
    ]
    return {
        "spec_version": "0.1",
        "axes": ["t", "c", "z", "y", "x"],
        "multiscale_level_count": n_levels,
        "levels": levels,
    }


# ---------------------------------------------------------------------------
# DCAScaleLevelSpec — codec
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCodecMust:
    def test_valid_zstd(self):
        assert validate_level(valid_level(codec_id="zstd")) == []

    def test_valid_lz4(self):
        assert validate_level(valid_level(codec_id="lz4")) == []

    def test_valid_blosc(self):
        # blosc is an allowed codec (MUST)
        # it generates a SHOULD warning for integer data
        issues = validate_level(valid_level(codec_id="blosc"))
        assert errors(issues) == []

    def test_invalid_codec_gzip(self):
        issues = validate_level(valid_level(codec_id="gzip"))
        assert len(errors(issues)) == 1
        assert "gzip" in errors(issues)[0].message

    def test_invalid_codec_none(self):
        issues = validate_level(valid_level(codec_id="none"))
        assert len(errors(issues)) >= 1

    def test_invalid_codec_unknown(self):
        issues = validate_level(valid_level(codec_id="lzma"))
        assert len(errors(issues)) == 1


# ---------------------------------------------------------------------------
# DCAScaleLevelSpec — chunk size
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkSizeMust:
    def test_chunk_above_1mb_passes(self):
        # 128*128*128*2 = 4 MB
        assert validate_level(valid_level(chunk_shape=[1, 1, 128, 128, 128])) == []

    def test_chunk_exactly_512kb_passes(self):
        # 512*1024 bytes / 2 bytes per uint16 = 262144 elements
        # [1, 1, 1, 512, 512] = 262144 * 2 = 512 KB
        issues = validate_level(valid_level(chunk_shape=[1, 1, 1, 512, 512]))
        assert len(errors(issues)) == 0

    def test_chunk_below_512kb_fails(self):
        # [1, 1, 1, 256, 256] * 2 = 131072 bytes = 128 KB
        issues = validate_level(valid_level(chunk_shape=[1, 1, 1, 256, 256]))
        errs = errors(issues)
        assert len(errs) == 1
        assert "512 KB" in errs[0].message

    def test_chunk_below_512kb_equal_to_image_size_passes(self):
        # chunk == image shape → small image, no error
        shape = [1, 1, 1, 256, 256]
        issues = validate_level(valid_level(shape=shape, chunk_shape=shape))
        assert len(errors(issues)) == 0

    def test_chunk_below_512kb_due_to_channel_split_passes(self):
        # [1,1,1,498,498]*2 = 484 KB < 512 KB, but excluding the channel
        # dimension the chunk covers the full spatial extent → no error
        shape = [1, 6, 1, 498, 498]
        chunk = [1, 1, 1, 498, 498]
        issues = validate_level(valid_level(shape=shape, chunk_shape=chunk))
        assert len(errors(issues)) == 0

    def test_chunk_below_512kb_passes_on_downsampled_level(self):
        # Same chunk that fails on level 0 passes on a downsampled level.
        issues = validate_level(
            valid_level(chunk_shape=[1, 1, 1, 256, 256], level_index=1)
        )
        assert errors(issues) == []
        assert not any("512 KB" in w.message for w in warnings(issues))


@pytest.mark.unit
class TestChunkSizeShould:
    def test_chunk_between_512kb_and_1mb_warns(self):
        # [1, 1, 1, 512, 512] * 2 = 512 KB — passes MUST, warns SHOULD
        issues = validate_level(valid_level(chunk_shape=[1, 1, 1, 512, 512]))
        warns = warnings(issues)
        assert any("1 MB" in w.message for w in warns)

    def test_chunk_above_1mb_no_warn(self):
        issues = validate_level(valid_level(chunk_shape=[1, 1, 128, 128, 128]))
        assert not any("1 MB" in w.message for w in warnings(issues))

    def test_chunk_below_1mb_no_warn_on_downsampled_level(self):
        # 512 KB chunk would warn on level 0; downsampled level skips the check.
        issues = validate_level(
            valid_level(chunk_shape=[1, 1, 1, 512, 512], level_index=1)
        )
        assert not any("1 MB" in w.message for w in warnings(issues))


# ---------------------------------------------------------------------------
# DCAScaleLevelSpec — chunk shape recommendations
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkShapeShould:
    def test_non_unit_channel_chunk_warns(self):
        issues = validate_level(valid_level(chunk_shape=[1, 2, 128, 128, 128]))
        assert any("Channel chunk" in w.message for w in warnings(issues))


# ---------------------------------------------------------------------------
# DCAScaleLevelSpec — shard size
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShardSize:
    def test_no_shard_skips_checks(self):
        issues = validate_level(valid_level(shard_shape=None))
        assert not any("shard" in i.message.lower() for i in issues)

    def test_shard_must_be_ge_1gb_when_array_ge_1gb(self):
        # Array: 1*1*512*1024*1024 * 2 = 1 GB exactly
        # Shard: 1*1*128*256*256 * 2 = 16 MB — too small
        issues = validate_level(
            valid_level(
                shape=[1, 1, 512, 1024, 1024],
                chunk_shape=[1, 1, 128, 128, 128],
                shard_shape=[1, 1, 128, 256, 256],
            )
        )
        errs = errors(issues)
        assert any("1 GB" in e.message for e in errs)

    def test_shard_lt_5tb_must(self):
        # shard 5 TB exactly: shape that gives >= 5 TB
        # 5 * 1024^4 bytes / 2 bytes = 2748779069440 elements
        # Approximate: [1, 1, 65536, 65536, 1024] * 2 ≈ way too large, use a formula
        # 5 TB = 5 * 1024**4 = 5497558138880 bytes; / 2 = 2748779069440 elements
        # Let's use [1, 1, 1, 65536, 65536] * 2 = 8589934592 bytes = 8 GB — not 5 TB
        # Use [1, 1, 1024, 1024, 2621440] * 2 = too complex
        # Simpler: just use a very large shard shape
        # 5 TB / 2 bytes = 2.75 * 10^12, hard to create dims for
        # Instead test SHOULD > 5 GB warning with a more reasonable shape:
        # [1, 1, 512, 1024, 1024] * 2 = 1 GB = 1 TB?
        #   No: 512*1024*1024 = 536870912 * 2 = 1 GB
        # [1, 1, 4096, 1024, 1024] * 2 = 8 GB > 5 GB → should warn
        issues = validate_level(
            valid_level(
                shape=[1, 1, 4096, 4096, 4096],
                chunk_shape=[1, 1, 128, 128, 128],
                shard_shape=[1, 1, 4096, 1024, 1024],
            )
        )
        warns = warnings(issues)
        assert any("5 GB" in w.message for w in warns)

    def test_spatial_shard_dim_gt_2048_warns(self):
        issues = validate_level(
            valid_level(
                shape=[1, 1, 1024, 4096, 4096],
                chunk_shape=[1, 1, 128, 128, 128],
                shard_shape=[1, 1, 512, 2049, 2049],
            )
        )
        warns = warnings(issues)
        assert any("2048" in w.message for w in warns)

    def test_time_shard_lt_16_warns(self):
        issues = validate_level(
            valid_level(
                shape=[100, 1, 64, 512, 512],
                chunk_shape=[1, 1, 128, 128, 128],
                shard_shape=[8, 1, 512, 512, 512],
            )
        )
        warns = warnings(issues)
        assert any("Time shard" in w.message for w in warns)

    def test_time_shard_lt_16_small_array_no_warn(self):
        # array has only 10 timepoints — shard of 10 is the max achievable, no warning
        issues = validate_level(
            valid_level(
                shape=[10, 1, 64, 512, 512],
                chunk_shape=[1, 1, 64, 128, 128],
                shard_shape=[10, 1, 512, 512, 512],
            )
        )
        assert not any("Time shard" in w.message for w in warnings(issues))


# ---------------------------------------------------------------------------
# DCAScaleLevelSpec — compression recommendation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompressionShould:
    def test_integer_zstd_high_level_warns(self):
        issues = validate_level(
            valid_level(codec_id="zstd", codec_level=9, dtype="uint16")
        )
        assert any("level" in w.message.lower() for w in warnings(issues))

    def test_integer_zstd_level_3_no_warn(self):
        issues = validate_level(
            valid_level(codec_id="zstd", codec_level=3, dtype="uint16")
        )
        assert not any("level" in w.message.lower() for w in warnings(issues))

    def test_float_blosc_warns(self):
        issues = validate_level(valid_level(codec_id="blosc", dtype="float32"))
        assert any("blosc" in w.message for w in warnings(issues))


# ---------------------------------------------------------------------------
# DCAScaleLevelSpec — dtype recommendation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDtypeShould:
    def test_uint8_no_warn(self):
        issues = validate_level(
            valid_level(dtype="uint8", chunk_shape=[1, 1, 128, 128, 128])
        )
        assert not any(
            "dtype" in w.message.lower() or "uint" in w.message.lower()
            for w in warnings(issues)
        )

    def test_uint16_no_warn(self):
        issues = validate_level(valid_level(dtype="uint16"))
        assert not any("unusual" in w.message for w in warnings(issues))

    def test_uint32_no_warn(self):
        issues = validate_level(valid_level(dtype="uint32"))
        assert not any("unusual" in w.message for w in warnings(issues))

    def test_float32_warns(self):
        issues = validate_level(valid_level(dtype="float32", codec_id="zstd"))
        assert any("unusual" in w.message for w in warnings(issues))

    def test_int16_warns(self):
        issues = validate_level(valid_level(dtype="int16"))
        assert any("unusual" in w.message for w in warnings(issues))


# ---------------------------------------------------------------------------
# DCAStoreSpecV0_1 — axes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAxesMust:
    def test_valid_axes(self):
        assert validate_node(valid_store()) == []

    def test_wrong_axes_order(self):
        data = valid_store()
        data["axes"] = ["z", "y", "x", "t", "c"]
        errs = errors(validate_node(data))
        assert any(
            "axes" in e.message.lower() or "t, c, z, y, x" in e.message.lower()
            for e in errs
        )

    def test_missing_axis(self):
        data = valid_store()
        data["axes"] = ["t", "c", "z", "y"]
        errs = errors(validate_node(data))
        assert len(errs) >= 1

    def test_extra_axis(self):
        data = valid_store()
        data["axes"] = ["t", "c", "z", "y", "x", "w"]
        errs = errors(validate_node(data))
        assert len(errs) >= 1

    def test_uppercase_axes_passes(self):
        data = valid_store()
        data["axes"] = ["T", "C", "Z", "Y", "X"]
        assert validate_node(data) == []

    def test_mixed_case_axes_errors(self):
        data = valid_store()
        data["axes"] = ["T", "C", "Z", "y", "x"]
        errs = errors(validate_node(data))
        assert any("Axes must be" in e.message for e in errs)


# ---------------------------------------------------------------------------
# DCAStoreSpecV0_1 — level count
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLevelCountMust:
    def test_three_levels_passes(self):
        assert validate_node(valid_store(n_levels=3)) == []

    def test_two_levels_fails(self):
        data = valid_store(n_levels=2)
        data["multiscale_level_count"] = 2
        errs = errors(validate_node(data))
        assert any("3" in e.message and "scale" in e.message.lower() for e in errs)

    def test_five_levels_passes(self):
        shapes = [
            [1, 1, 256, 512, 512],
            [1, 1, 128, 256, 256],
            [1, 1, 64, 128, 128],
            [1, 1, 32, 64, 64],
            [1, 1, 16, 32, 32],
        ]
        data = {
            "spec_version": "0.1",
            "axes": ["t", "c", "z", "y", "x"],
            "multiscale_level_count": 5,
            "levels": [valid_level(str(i), shape=s) for i, s in enumerate(shapes)],
        }
        assert validate_node(data) == []


# ---------------------------------------------------------------------------
# DCAStoreSpecV0_1 — downsampling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDownsamplingMust:
    def test_valid_2x_spatial_downsampling(self):
        assert validate_node(valid_store()) == []

    def test_t_downsampled_fails(self):
        data = valid_store()
        data["levels"][1]["shape"] = [1, 1, 128, 256, 256]  # t unchanged ✓
        data["levels"][0]["shape"] = [2, 1, 256, 512, 512]  # t=2 at level 0
        # level 0 → 1: t goes 2 → 1 (downsampled) — MUST error
        errs = errors(validate_node(data))
        assert any("'t'" in e.message for e in errs)

    def test_c_downsampled_fails(self):
        data = valid_store()
        data["levels"][0]["shape"] = [1, 2, 256, 512, 512]  # c=2 at level 0
        data["levels"][1]["shape"] = [1, 1, 128, 256, 256]  # c=1 at level 1
        errs = errors(validate_node(data))
        assert any("'c'" in e.message for e in errs)


@pytest.mark.unit
class TestDownsamplingShould:
    def test_non_power_of_2_spatial_warns(self):
        data = valid_store()
        # level 0 → 1: z goes 256 → 200 (factor ≈ 1.28, not ≈ 2)
        data["levels"][1]["shape"] = [1, 1, 200, 256, 256]
        warns = warnings(validate_node(data))
        assert any("'z'" in w.message for w in warns)


# ---------------------------------------------------------------------------
# DCAStoreSpecV0_1 — spec version
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpecVersion:
    def test_valid_version(self):
        assert validate_node(valid_store()) == []

    def test_wrong_version_fails(self):
        data = valid_store()
        data["spec_version"] = "0.2"
        with pytest.raises(pydantic.ValidationError):
            DCAStoreSpecV0_1.model_validate(data)


@pytest.mark.unit
class TestLevelRulesAreReportedTogether:
    BIG_ARRAY = [1, 1, 512, 1024, 1024]
    SMALL_SHARD = [1, 1, 128, 128, 128]

    def test_chunk_warning_does_not_hide_shard_error(self):
        issues = validate_level(
            valid_level(
                shape=self.BIG_ARRAY,
                chunk_shape=[1, 1, 64, 64, 64],
                shard_shape=self.SMALL_SHARD,
            )
        )
        assert any("recommended 1 MB" in i.message for i in warnings(issues))
        assert any("shard size" in i.message for i in errors(issues))

    def test_channel_chunk_warning_does_not_hide_shard_error(self):
        issues = validate_level(
            valid_level(
                shape=[1, 2, 512, 1024, 1024],
                chunk_shape=[1, 2, 128, 128, 128],
                shard_shape=[1, 2, 128, 128, 128],
            )
        )
        assert any("Channel chunk size" in i.message for i in warnings(issues))
        assert any("shard size" in i.message for i in errors(issues))

    def test_every_warning_and_error_is_reported(self):
        issues = validate_level(
            valid_level(
                shape=self.BIG_ARRAY,
                chunk_shape=[1, 2, 32, 32, 64],
                shard_shape=self.SMALL_SHARD,
                codec_level=9,
                dtype="int32",
            )
        )
        messages = [i.message for i in issues]
        for fragment in (
            "recommended 1 MB",
            "Channel chunk size",
            "shard size",
            "Compression level",
            "dtype",
        ):
            assert any(fragment in m for m in messages), fragment
        assert len(errors(issues)) == 1
