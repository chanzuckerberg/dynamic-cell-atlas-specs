"""
Unit tests for DCA v0.2 Pydantic models.

Tests cover every MUST and SHOULD rule introduced in v0.2, plus verification
that inherited v0.1 array-level rules still fire on the v0.2 model classes.
No real Zarr store is required.

Test classes are organised by the model and spec section they exercise:

  TestDCAScaleLevelSpecV0_2
      — inherited array rules (codec, chunk, shard, dtype)
  TestDCALabelScaleLevelSpecV0_2
      — dtype SHOULD override for label arrays (uint32 required instead of uint8/uint16)
  TestCoarseLevelMust
      — array-standard.rst § Multiresolution (replaces "≥ 3 levels")
  TestNoMinimumLevelCount
      — v0.1 "≥ 3 levels" rule is gone in v0.2
  TestDownsamplingV0_2
      — array-standard.rst § Multiresolution (t/c MUST, spatial SHOULD)
  TestAxesV0_2
      — array-standard.rst § Dimensions
  TestDcaKeyMust
      — channel-metadata.rst § Location in Zarr store (dca MUST)
  TestDcaVersionMust
      — channel-metadata.rst § Location in Zarr store (version MUST)
  TestDcaUrlMust
      — channel-metadata.rst § Location in Zarr store (url MUST)
  TestChannelRequiredFieldsMust
      — channel-metadata.rst § Channel Fields / Required Fields
  TestChannelIndexMust
      — channel-metadata.rst § Channel Fields / Required Fields (index)
  TestChannelIndicesUniqueMust
      — channel-metadata.rst § Channel Fields / Required Fields
  TestChannelCountShould
      — channel-metadata.rst § Location in Zarr store (SHOULD)
  TestChannelTypeShould
      — channel-metadata.rst § Guidance on Channel Type
  TestBioAnnotationShould
      — channel-metadata.rst § Channel Fields / Optional Fields
  TestMarkerTypeShould
      — channel-metadata.rst § Biological Annotation Entries /
        Guidance on Marker Type Vocabulary
  TestDcaMissingWithLevelWarnings
      — regression: dca MUST fires even when levels carry [SHOULD] warnings
  TestLabelStoreSpecV0_2
      — DCALabelStoreSpecV0_2: axes, coarse-level, downsampling, no dca field
  TestSpecVersionPin
      — spec_version Literal["0.2"] enforcement
  TestDimensionNamesMust
      — ome-zarr-metadata.rst § Dimensions (dimension_names MUST match axes)
  TestAxisUnitsMust
      — ome-zarr-metadata.rst § Units (spatial/temporal axes MUST use OME-NGFF v0.5 vocabulary)
"""

import pydantic
import pytest

from dca_helpers.validation.result import Severity
from dca_helpers.validation.validator import _convert_errors
from dca_helpers.validation.versions.v0_2.models import (
    CHANNEL_TYPES,
    COARSE_LEVEL_MAX_SPATIAL,
    CPG_LABELED_MOLECULES,
    CPG_LABELED_STRUCTURES,
    MARKER_TYPES,
    SPATIAL_UNIT_VOCABULARY,
    TEMPORAL_UNIT_VOCABULARY,
    DCABiologicalAnnotation,
    DCAChannelMetadataV0_2,
    DCAIntensityStatistics,
    DCALabelScaleLevelSpecV0_2,
    DCALabelStoreSpecV0_2,
    DCANormalizationMetadata,
    DCANormalizationStatistics,
    DCAScaleLevelSpecV0_2,
    DCAStoreSpecV0_2,
)

_SPEC_URL = (
    "https://github.com/chanzuckerberg/dynamic-cell-atlas-specs"
    "/tree/main/docs/v0.2/schema.json"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def validate_store(data: dict) -> list:
    """Return Issues from validating a store dict, or [] on clean pass."""
    try:
        DCAStoreSpecV0_2.model_validate(data)
        return []
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)


def validate_label_store(data: dict) -> list:
    """Return Issues from validating a label store dict, or [] on clean pass."""
    try:
        DCALabelStoreSpecV0_2.model_validate(data)
        return []
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)


def validate_level(data: dict) -> list:
    """Return Issues from validating a single level dict."""
    try:
        DCAScaleLevelSpecV0_2.model_validate(data)
        return []
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)


def validate_label_level(data: dict) -> list:
    """Return Issues from validating a single label level dict."""
    try:
        DCALabelScaleLevelSpecV0_2.model_validate(data)
        return []
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)


def validate_channel(data: dict) -> list:
    """Return Issues from validating a single channel metadata dict."""
    try:
        DCAChannelMetadataV0_2.model_validate(data)
        return []
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)


def validate_bio_annotation(data: dict) -> list:
    """Return Issues from validating a biological annotation dict."""
    try:
        DCABiologicalAnnotation.model_validate(data)
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


_SENTINEL = object()


def valid_level(
    path="0",
    shape=None,
    chunk_shape=None,
    shard_shape=None,
    codec_id="zstd",
    codec_level=3,
    dtype="uint16",
    level_index=0,
    dimension_names=_SENTINEL,
):
    d = {
        "level_index": level_index,
        "path": path,
        "dtype": dtype,
        "shape": shape or [1, 1, 256, 512, 512],
        "chunk_shape": chunk_shape or [1, 1, 128, 128, 128],
        "shard_shape": shard_shape,
        "codec_id": codec_id,
        "codec_level": codec_level,
        "dimension_names": (
            ["t", "c", "z", "y", "x"]
            if dimension_names is _SENTINEL
            else dimension_names
        ),
    }
    return d


def valid_label_level(
    path="0",
    shape=None,
    chunk_shape=None,
    shard_shape=None,
    codec_id="zstd",
    codec_level=3,
    dtype="uint32",
    level_index=0,
    dimension_names=_SENTINEL,
):
    """Like valid_level but defaults to uint32 (the recommended dtype for label arrays)."""
    return valid_level(
        path=path,
        shape=shape,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        codec_id=codec_id,
        codec_level=codec_level,
        dtype=dtype,
        level_index=level_index,
        dimension_names=dimension_names,
    )


def valid_normalization_statistics(n_channels=1):
    """Minimal valid normalization_statistics with one dataset_statistics entry per channel."""
    return {
        str(i): {"dataset_statistics": {"p1": 1.0, "p5": 5.0, "p95": 95.0, "p99": 99.0}}
        for i in range(n_channels)
    }


def valid_dca(n_channels=1):
    """Minimal valid dca object with n_channels channel entries."""
    return {
        "version": "0.2",
        "url": _SPEC_URL,
        "channels": [
            {"name": f"ch{i}", "index": i, "description": f"channel {i}"}
            for i in range(n_channels)
        ],
        "normalization_statistics": valid_normalization_statistics(n_channels),
    }


VALID_AXIS_UNITS = {
    "t": "millisecond",
    "c": None,
    "z": "micrometer",
    "y": "micrometer",
    "x": "micrometer",
}


def valid_store(n_levels=3, c=1, axis_units=_SENTINEL):
    """
    Minimal valid v0.2 store with n_levels pyramid levels and c channels.
    Level 0 shape is [1, c, 256, 512, 512] downsampled by 2 each level.
    """
    base_shapes = [
        [1, c, 256, 512, 512],
        [1, c, 128, 256, 256],
        [1, c, 64, 128, 128],
        [1, c, 32, 64, 64],
        [1, c, 16, 32, 32],
    ]
    levels = [
        valid_level(str(i), shape=base_shapes[i], level_index=i)
        for i in range(n_levels)
    ]
    return {
        "spec_version": "0.2",
        "axes": ["t", "c", "z", "y", "x"],
        "axis_units": VALID_AXIS_UNITS if axis_units is _SENTINEL else axis_units,
        "levels": levels,
        "dca": valid_dca(c),
    }


def valid_label_store(n_levels=3):
    """
    Minimal valid v0.2 label store with n_levels pyramid levels.
    Uses uint32 dtype and no dca field.
    """
    base_shapes = [
        [1, 1, 256, 512, 512],
        [1, 1, 128, 256, 256],
        [1, 1, 64, 128, 128],
        [1, 1, 32, 64, 64],
        [1, 1, 16, 32, 32],
    ]
    levels = [
        valid_label_level(str(i), shape=base_shapes[i], level_index=i)
        for i in range(n_levels)
    ]
    return {
        "spec_version": "0.2",
        "axes": ["t", "c", "z", "y", "x"],
        "levels": levels,
    }


# ---------------------------------------------------------------------------
# DCAScaleLevelSpecV0_2 — inherited array rules still fire
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDCAScaleLevelSpecV0_2:
    """
    Verifies that all v0.1 array-level rules are inherited unchanged.
    Full coverage of individual rules lives in the v0.1 test_models.py; these
    tests confirm the inheritance is wired correctly.
    """

    def test_valid_level_passes(self):
        assert validate_level(valid_level()) == []

    def test_invalid_codec_still_errors(self):
        errs = errors(validate_level(valid_level(codec_id="gzip")))
        assert len(errs) == 1
        assert "gzip" in errs[0].message

    def test_chunk_below_512kb_still_errors(self):
        # [1,1,1,256,256] * 2 bytes = 128 KB
        errs = errors(validate_level(valid_level(chunk_shape=[1, 1, 1, 256, 256])))
        assert any("512 KB" in e.message for e in errs)

    def test_chunk_below_512kb_passes_on_downsampled_level(self):
        # v0.2 inherits the level-aware chunk-size exception from v0.1.
        issues = validate_level(
            valid_level(chunk_shape=[1, 1, 1, 256, 256], level_index=1)
        )
        assert errors(issues) == []

    def test_shard_below_1gb_when_array_ge_1gb_still_errors(self):
        # array 1 GB exactly; shard only 16 MB
        errs = errors(
            validate_level(
                valid_level(
                    shape=[1, 1, 512, 1024, 1024],
                    chunk_shape=[1, 1, 128, 128, 128],
                    shard_shape=[1, 1, 128, 256, 256],
                )
            )
        )
        assert any("1 GB" in e.message for e in errs)

    def test_unusual_dtype_still_warns(self):
        warns = warnings(validate_level(valid_level(dtype="float32", codec_id="zstd")))
        assert any("unusual" in w.message for w in warns)


# ---------------------------------------------------------------------------
# DCALabelScaleLevelSpecV0_2 — dtype SHOULD override for label arrays
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDCALabelScaleLevelSpecV0_2:
    """
    DCALabelScaleLevelSpecV0_2 overrides the dtype SHOULD from v0.1:
    label arrays SHOULD use uint32, not uint8/uint16.
    """

    def test_uint32_passes(self):
        assert validate_label_level(valid_label_level(dtype="uint32")) == []

    def test_uint8_warns(self):
        warns = warnings(validate_label_level(valid_label_level(dtype="uint8")))
        assert any("uint32" in w.message for w in warns)

    def test_uint16_warns(self):
        warns = warnings(validate_label_level(valid_label_level(dtype="uint16")))
        assert any("uint32" in w.message for w in warns)

    def test_float32_warns(self):
        # float32 is unusual for label arrays — dtype warning fires
        warns = warnings(
            validate_label_level(valid_label_level(dtype="float32", codec_id="zstd"))
        )
        assert any("uint32" in w.message for w in warns)

    def test_inherited_codec_rule_still_fires(self):
        errs = errors(validate_label_level(valid_label_level(codec_id="gzip")))
        assert any("gzip" in e.message for e in errs)

    def test_inherited_chunk_size_rule_still_fires(self):
        errs = errors(
            validate_label_level(
                valid_label_level(chunk_shape=[1, 1, 1, 256, 256], dtype="uint32")
            )
        )
        assert any("512 KB" in e.message for e in errs)


# ---------------------------------------------------------------------------
# array-standard.rst § Multiresolution — coarse level MUST exist
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCoarseLevelMust:
    def test_level_within_2048_in_all_spatial_dims_passes(self):
        assert validate_store(valid_store(n_levels=1)) == []

    def test_exactly_2048_in_all_spatial_dims_passes(self):
        level = valid_level("0", shape=[1, 1, 2048, 2048, 2048])
        data = {**valid_store(n_levels=1), "levels": [level]}
        assert errors(validate_store(data)) == []

    def test_one_spatial_dim_above_2048_at_only_level_fails(self):
        level = valid_level("0", shape=[1, 1, 2049, 512, 512])
        data = {**valid_store(n_levels=1), "levels": [level]}
        errs = errors(validate_store(data))
        assert any(str(COARSE_LEVEL_MAX_SPATIAL) in e.message for e in errs)

    def test_coarse_level_satisfied_at_level_2_passes(self):
        levels = [
            valid_level("0", shape=[1, 1, 4096, 4096, 4096], level_index=0),
            valid_level("1", shape=[1, 1, 2048, 2048, 2048], level_index=1),
            valid_level("2", shape=[1, 1, 1024, 1024, 1024], level_index=2),
        ]
        data = {**valid_store(), "levels": levels}
        assert errors(validate_store(data)) == []

    def test_all_levels_above_2048_fails(self):
        levels = [
            valid_level("0", shape=[1, 1, 8192, 8192, 8192], level_index=0),
            valid_level("1", shape=[1, 1, 4096, 4096, 4096], level_index=1),
            valid_level("2", shape=[1, 1, 2049, 2049, 2049], level_index=2),
        ]
        data = {**valid_store(), "levels": levels}
        errs = errors(validate_store(data))
        assert any(str(COARSE_LEVEL_MAX_SPATIAL) in e.message for e in errs)

    def test_only_one_spatial_dim_exceeds_2048_fails(self):
        # y = 2049 but z and x are within limit — all three must be ≤ 2048
        level = valid_level("0", shape=[1, 1, 64, 2049, 512])
        data = {**valid_store(n_levels=1), "levels": [level]}
        errs = errors(validate_store(data))
        assert any(str(COARSE_LEVEL_MAX_SPATIAL) in e.message for e in errs)


# ---------------------------------------------------------------------------
# array-standard.rst § Multiresolution — "≥ 3 levels" rule absent in v0.2
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoMinimumLevelCount:
    """
    In v0.1 a store with fewer than 3 levels fails. In v0.2 that rule is
    removed — only the coarse-level-≤-2048 MUST applies.
    """

    def test_single_level_passes(self):
        assert validate_store(valid_store(n_levels=1)) == []

    def test_two_levels_pass(self):
        assert validate_store(valid_store(n_levels=2)) == []

    def test_five_levels_pass(self):
        assert validate_store(valid_store(n_levels=5)) == []


# ---------------------------------------------------------------------------
# array-standard.rst § Multiresolution — downsampling factors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDownsamplingV0_2:
    def test_valid_2x_spatial_downsampling_passes(self):
        assert validate_store(valid_store(n_levels=3)) == []

    def test_t_downsampled_errors(self):
        data = valid_store(n_levels=3)
        data["levels"][0]["shape"] = [2, 1, 256, 512, 512]
        data["levels"][1]["shape"] = [1, 1, 128, 256, 256]
        errs = errors(validate_store(data))
        assert any("'t'" in e.message for e in errs)

    def test_c_downsampled_errors(self):
        data = valid_store(n_levels=3, c=2)
        data["levels"][0]["shape"] = [1, 2, 256, 512, 512]
        data["levels"][1]["shape"] = [1, 1, 128, 256, 256]
        errs = errors(validate_store(data))
        assert any("'c'" in e.message for e in errs)

    def test_non_factor_2_spatial_warns(self):
        data = valid_store(n_levels=2)
        data["levels"][1]["shape"] = [1, 1, 200, 256, 256]  # z factor ≈ 1.28
        warns = warnings(validate_store(data))
        assert any("'z'" in w.message for w in warns)

    def test_single_level_skips_downsampling_check(self):
        assert validate_store(valid_store(n_levels=1)) == []


# ---------------------------------------------------------------------------
# array-standard.rst § Dimensions — axes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAxesV0_2:
    def test_valid_axes_passes(self):
        assert validate_store(valid_store()) == []

    def test_wrong_order_errors(self):
        data = valid_store()
        data["axes"] = ["z", "y", "x", "t", "c"]
        assert len(errors(validate_store(data))) >= 1

    def test_missing_axis_errors(self):
        data = valid_store()
        data["axes"] = ["t", "c", "z", "y"]
        assert len(errors(validate_store(data))) >= 1

    def test_extra_axis_errors(self):
        data = valid_store()
        data["axes"] = ["t", "c", "z", "y", "x", "w"]
        assert len(errors(validate_store(data))) >= 1

    def test_uppercase_axes_passes(self):
        data = valid_store()
        data["axes"] = ["T", "C", "Z", "Y", "X"]
        assert validate_store(data) == []

    def test_mixed_case_axes_errors(self):
        data = valid_store()
        data["axes"] = ["T", "C", "Z", "y", "x"]
        errs = errors(validate_store(data))
        assert any("Axes must be" in e.message for e in errs)


# ---------------------------------------------------------------------------
# channel-metadata.rst § Location in Zarr store — dca key MUST be present
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDcaKeyMust:
    def test_dca_present_passes(self):
        assert validate_store(valid_store()) == []

    def test_dca_absent_errors(self):
        data = valid_store()
        del data["dca"]
        errs = errors(validate_store(data))
        assert any("dca" in e.message.lower() for e in errs)

    def test_dca_none_errors(self):
        data = {**valid_store(), "dca": None}
        errs = errors(validate_store(data))
        assert any("dca" in e.message.lower() for e in errs)


# ---------------------------------------------------------------------------
# channel-metadata.rst § Location in Zarr store — version and url MUST
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDcaVersionMust:
    def test_correct_version_passes(self):
        assert validate_store(valid_store()) == []

    def test_wrong_version_errors(self):
        data = {**valid_store(), "dca": {**valid_dca(), "version": "0.1"}}
        assert len(errors(validate_store(data))) >= 1

    def test_missing_version_errors(self):
        dca = valid_dca().copy()
        del dca["version"]
        data = {**valid_store(), "dca": dca}
        assert len(errors(validate_store(data))) >= 1


@pytest.mark.unit
class TestDcaUrlMust:
    def test_url_present_passes(self):
        assert validate_store(valid_store()) == []

    def test_missing_url_errors(self):
        dca = valid_dca().copy()
        del dca["url"]
        data = {**valid_store(), "dca": dca}
        assert len(errors(validate_store(data))) >= 1


# ---------------------------------------------------------------------------
# channel-metadata.rst § Channel Fields / Required Fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChannelRequiredFieldsMust:
    def test_all_required_fields_present_passes(self):
        assert (
            validate_channel({"name": "DAPI", "index": 0, "description": "nucleus"})
            == []
        )

    def test_missing_name_errors(self):
        errs = errors(validate_channel({"index": 0, "description": "d"}))
        assert len(errs) >= 1

    def test_missing_index_errors(self):
        errs = errors(validate_channel({"name": "ch", "description": "d"}))
        assert len(errs) >= 1

    def test_missing_description_errors(self):
        errs = errors(validate_channel({"name": "ch", "index": 0}))
        assert len(errs) >= 1

    def test_extra_fields_ignored(self):
        # extra="allow" — unknown keys pass through without error
        assert (
            validate_channel(
                {"name": "ch", "index": 0, "description": "d", "unknown_key": "x"}
            )
            == []
        )


@pytest.mark.unit
class TestChannelIndexMust:
    def test_zero_index_passes(self):
        assert validate_channel({"name": "ch", "index": 0, "description": "d"}) == []

    def test_positive_index_passes(self):
        assert validate_channel({"name": "ch", "index": 5, "description": "d"}) == []

    def test_negative_index_errors(self):
        errs = errors(validate_channel({"name": "ch", "index": -1, "description": "d"}))
        assert any("greater than or equal to 0" in e.message for e in errs)


# ---------------------------------------------------------------------------
# channel-metadata.rst § Channel Fields / Required Fields — unique indices
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChannelIndicesUniqueMust:
    def test_unique_indices_pass(self):
        dca = {
            "version": "0.2",
            "url": _SPEC_URL,
            "channels": [
                {"name": "a", "index": 0, "description": "d"},
                {"name": "b", "index": 1, "description": "d"},
                {"name": "c", "index": 2, "description": "d"},
            ],
            "normalization_statistics": {},
        }
        assert errors(validate_store({**valid_store(c=3), "dca": dca})) == []

    def test_duplicate_indices_error(self):
        dca = {
            "version": "0.2",
            "url": _SPEC_URL,
            "channels": [
                {"name": "a", "index": 0, "description": "d"},
                {"name": "b", "index": 0, "description": "d"},
            ],
            "normalization_statistics": {},
        }
        data = {**valid_store(c=2), "dca": dca}
        errs = errors(validate_store(data))
        assert any("unique" in e.message for e in errs)

    def test_non_contiguous_indices_allowed(self):
        # The spec example shows index 0 and 2 with no index 1 — this is valid
        dca = {
            "version": "0.2",
            "url": _SPEC_URL,
            "channels": [
                {"name": "Phase", "index": 0, "description": "label-free"},
                {"name": "H2B-GFP", "index": 2, "description": "nucleus"},
            ],
            "normalization_statistics": {},
        }
        data = {**valid_store(c=2), "dca": dca}
        assert errors(validate_store(data)) == []


# ---------------------------------------------------------------------------
# channel-metadata.rst § Location in Zarr store — channel count SHOULD match C axis
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChannelCountShould:
    def test_channel_count_matches_c_axis_passes(self):
        assert validate_store(valid_store(c=3)) == []

    def test_fewer_channels_than_c_axis_warns(self):
        data = valid_store(c=3)
        data["dca"] = valid_dca(n_channels=1)
        warns = warnings(validate_store(data))
        assert any("channel" in w.message.lower() for w in warns)

    def test_more_channels_than_c_axis_warns(self):
        data = valid_store(c=1)
        data["dca"] = valid_dca(n_channels=3)
        warns = warnings(validate_store(data))
        assert any("channel" in w.message.lower() for w in warns)

    def test_empty_channels_list_warns_when_c_gt_0(self):
        data = valid_store(c=2)
        data["dca"] = {
            "version": "0.2",
            "url": _SPEC_URL,
            "channels": [],
            "normalization_statistics": {},
        }
        warns = warnings(validate_store(data))
        assert any("channel" in w.message.lower() for w in warns)

    def test_zero_c_axis_with_empty_channels_passes(self):
        # Placeholder C=0 (unusual but not explicitly forbidden) with no channel entries
        level = valid_level("0", shape=[1, 0, 64, 128, 128])
        data = {
            "spec_version": "0.2",
            "axes": ["t", "c", "z", "y", "x"],
            "levels": [level],
            "dca": {
                "version": "0.2",
                "url": _SPEC_URL,
                "channels": [],
                "normalization_statistics": {},
            },
        }
        assert warnings(validate_store(data)) == []


# ---------------------------------------------------------------------------
# channel-metadata.rst § Guidance on Channel Type — SHOULD
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChannelTypeShould:
    @pytest.mark.parametrize("ct", sorted(CHANNEL_TYPES))
    def test_known_channel_type_passes(self, ct):
        issues = validate_channel(
            {"name": "ch", "index": 0, "description": "d", "channel_type": ct}
        )
        assert errors(issues) == []
        # No vocabulary warning for this channel_type
        assert not any(
            "channel_type" in w.message and "recommended vocabulary" in w.message
            for w in warnings(issues)
        )

    def test_unknown_channel_type_warns(self):
        warns = warnings(
            validate_channel(
                {
                    "name": "ch",
                    "index": 0,
                    "description": "d",
                    "channel_type": "brightfield",
                }
            )
        )
        assert any("channel_type" in w.message for w in warns)

    def test_absent_channel_type_passes_silently(self):
        assert validate_channel({"name": "ch", "index": 0, "description": "d"}) == []


# ---------------------------------------------------------------------------
# channel-metadata.rst § Channel Fields / Optional Fields —
# biological_annotation SHOULD for fluorescence / predicted
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBioAnnotationShould:
    def test_fluorescence_with_bio_annotation_passes(self):
        issues = validate_channel(
            {
                "name": "GFP",
                "index": 0,
                "description": "d",
                "channel_type": "fluorescence",
                "biological_annotation": {"biological_target": "nucleus"},
            }
        )
        assert not any("biological_annotation" in w.message for w in warnings(issues))

    def test_fluorescence_without_bio_annotation_warns(self):
        warns = warnings(
            validate_channel(
                {
                    "name": "GFP",
                    "index": 0,
                    "description": "d",
                    "channel_type": "fluorescence",
                }
            )
        )
        assert any("biological_annotation" in w.message for w in warns)

    def test_predicted_without_bio_annotation_warns(self):
        warns = warnings(
            validate_channel(
                {
                    "name": "VS",
                    "index": 0,
                    "description": "d",
                    "channel_type": "predicted",
                }
            )
        )
        assert any("biological_annotation" in w.message for w in warns)

    def test_labelfree_without_bio_annotation_passes_silently(self):
        issues = validate_channel(
            {
                "name": "Phase",
                "index": 0,
                "description": "d",
                "channel_type": "labelfree",
            }
        )
        assert not any("biological_annotation" in w.message for w in warnings(issues))

    def test_no_channel_type_without_bio_annotation_passes_silently(self):
        assert validate_channel({"name": "ch", "index": 0, "description": "d"}) == []

    def test_unknown_channel_type_without_bio_annotation_does_not_require_it(self):
        # Only fluorescence and predicted require biological_annotation.
        # An unrecognised channel_type produces a vocab WARNING but never a
        # biological_annotation WARNING.
        issues = validate_channel(
            {
                "name": "brightfield",
                "index": 0,
                "description": "d",
                "channel_type": "electrochemical",
            }
        )
        assert not any("biological_annotation" in w.message for w in warnings(issues))
        # Vocabulary warning IS expected since electrochemical is not in CHANNEL_TYPES
        assert any("channel_type" in w.message for w in warnings(issues))


# ---------------------------------------------------------------------------
# channel-metadata.rst § Biological Annotation Entries /
# Guidance on Marker Type Vocabulary — SHOULD
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarkerTypeShould:
    @pytest.mark.parametrize("mt", sorted(MARKER_TYPES))
    def test_known_marker_type_passes(self, mt):
        issues = validate_bio_annotation({"marker_type": mt})
        assert errors(issues) == []
        assert not any("marker_type" in w.message for w in warnings(issues))

    def test_unknown_marker_type_warns(self):
        warns = warnings(validate_bio_annotation({"marker_type": "nanobody"}))
        assert any("marker_type" in w.message for w in warns)

    def test_none_marker_type_passes(self):
        assert validate_bio_annotation({"marker_type": None}) == []

    def test_absent_marker_type_passes(self):
        assert validate_bio_annotation({}) == []

    def test_all_optional_fields_accepted(self):
        assert (
            validate_bio_annotation(
                {
                    "biological_target": "nucleus",
                    "marker_type": "endogenous_tag",
                    "marker": "H2B-GFP",
                    "cpg_labeled_structure": "Nucleus",
                    "cpg_labeled_molecule": "DNA",
                }
            )
            == []
        )


# ---------------------------------------------------------------------------
# schema.json BiologicalAnnotation field names and cpg_* vocabulary — SHOULD
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBiologicalAnnotationFields:
    def test_biological_target_accepted(self):
        assert validate_bio_annotation({"biological_target": "nucleus"}) == []

    def test_absent_fields_pass(self):
        assert validate_bio_annotation({}) == []

    def test_cpg_labeled_structure_known_value_passes(self):
        issues = validate_bio_annotation({"cpg_labeled_structure": "Nucleus"})
        assert not any("cpg_labeled_structure" in w.message for w in warnings(issues))

    def test_cpg_labeled_molecule_known_value_passes(self):
        issues = validate_bio_annotation({"cpg_labeled_molecule": "DNA"})
        assert not any("cpg_labeled_molecule" in w.message for w in warnings(issues))

    def test_cpg_labeled_structure_unknown_warns(self):
        warns = warnings(
            validate_bio_annotation({"cpg_labeled_structure": "cytoskeleton"})
        )
        assert any("cpg_labeled_structure" in w.message for w in warns)

    def test_cpg_labeled_molecule_unknown_warns(self):
        warns = warnings(validate_bio_annotation({"cpg_labeled_molecule": "lipids"}))
        assert any("cpg_labeled_molecule" in w.message for w in warns)

    @pytest.mark.parametrize("s", sorted(CPG_LABELED_STRUCTURES))
    def test_all_cpg_structures_pass(self, s):
        issues = validate_bio_annotation({"cpg_labeled_structure": s})
        assert not any("cpg_labeled_structure" in w.message for w in warnings(issues))

    @pytest.mark.parametrize("m", sorted(CPG_LABELED_MOLECULES))
    def test_all_cpg_molecules_pass(self, m):
        issues = validate_bio_annotation({"cpg_labeled_molecule": m})
        assert not any("cpg_labeled_molecule" in w.message for w in warnings(issues))


# ---------------------------------------------------------------------------
# normalization-statistics.rst + schema.json NormalizationStatistics — MUST
# ---------------------------------------------------------------------------


def validate_normalization_statistics(data: dict) -> list:
    """Return Issues from validating a normalization_statistics dict."""
    try:
        DCANormalizationStatistics.model_validate(data)
        return []
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)


@pytest.mark.unit
class TestNormalizationStatisticsMust:
    def test_empty_object_passes(self):
        assert validate_normalization_statistics({}) == []

    def test_channel_entry_with_dataset_statistics_passes(self):
        data = {
            "0": {
                "dataset_statistics": {"p1": 1.0, "p5": 5.0, "p95": 95.0, "p99": 99.0}
            }
        }
        assert validate_normalization_statistics(data) == []

    def test_channel_entry_missing_dataset_statistics_errors(self):
        errs = errors(validate_normalization_statistics({"0": {}}))
        assert errs

    def test_dataset_statistics_missing_p1_errors(self):
        data = {"0": {"dataset_statistics": {"p5": 5.0, "p95": 95.0, "p99": 99.0}}}
        errs = errors(validate_normalization_statistics(data))
        assert errs

    def test_dataset_statistics_all_required_fields_pass(self):
        stats = {"p1": 1.0, "p5": 5.0, "p95": 95.0, "p99": 99.0}
        data = {"0": {"dataset_statistics": stats}}
        assert errors(validate_normalization_statistics(data)) == []

    def test_optional_intensity_fields_accepted(self):
        stats = {
            "p1": 1.0,
            "p5": 5.0,
            "p95": 95.0,
            "p99": 99.0,
            "mean": 50.0,
            "std": 10.0,
            "median": 48.0,
            "p95_p5": 90.0,
            "p99_p1": 98.0,
            "iqr": 20.0,
        }
        data = {"0": {"dataset_statistics": stats}}
        assert validate_normalization_statistics(data) == []

    def test_metadata_key_accepted(self):
        data = {"metadata": {"grid": 16}}
        assert validate_normalization_statistics(data) == []

    def test_non_digit_non_metadata_key_errors(self):
        errs = errors(validate_normalization_statistics({"bad_key": {}}))
        assert errs

    def test_multiple_channel_entries_pass(self):
        entry = {"dataset_statistics": {"p1": 1.0, "p5": 5.0, "p95": 95.0, "p99": 99.0}}
        data = {"0": entry, "1": entry, "2": entry}
        assert validate_normalization_statistics(data) == []

    def test_normalization_statistics_required_in_dca(self):
        # dca without normalization_statistics MUST error
        data = valid_store()
        del data["dca"]["normalization_statistics"]
        errs = errors(validate_store(data))
        assert any("normalization_statistics" in str(e.loc) for e in errs)

    def test_normalization_statistics_present_passes(self):
        assert validate_store(valid_store()) == []


# ---------------------------------------------------------------------------
# schema.json IntensityStatistics minimum: 0 — p95_p5, p99_p1, iqr
# ---------------------------------------------------------------------------


def validate_intensity_statistics(data: dict) -> list:
    """Return Issues from validating an IntensityStatistics dict."""
    try:
        DCAIntensityStatistics.model_validate(data)
        return []
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)


_REQUIRED_STATS = {"p1": 1.0, "p5": 5.0, "p95": 95.0, "p99": 99.0}


@pytest.mark.unit
class TestNonNegativeStats:
    def test_zero_p95_p5_passes(self):
        assert validate_intensity_statistics({**_REQUIRED_STATS, "p95_p5": 0.0}) == []

    def test_positive_p95_p5_passes(self):
        assert validate_intensity_statistics({**_REQUIRED_STATS, "p95_p5": 5.0}) == []

    def test_negative_p95_p5_errors(self):
        errs = errors(
            validate_intensity_statistics({**_REQUIRED_STATS, "p95_p5": -0.1})
        )
        assert len(errs) >= 1

    def test_negative_p99_p1_errors(self):
        errs = errors(
            validate_intensity_statistics({**_REQUIRED_STATS, "p99_p1": -1.0})
        )
        assert len(errs) >= 1

    def test_negative_iqr_errors(self):
        errs = errors(validate_intensity_statistics({**_REQUIRED_STATS, "iqr": -5.0}))
        assert len(errs) >= 1

    def test_none_optional_stats_passes(self):
        assert validate_intensity_statistics({**_REQUIRED_STATS}) == []


# ---------------------------------------------------------------------------
# schema.json NormalizationMetadata minimum: 1 — grid
# ---------------------------------------------------------------------------


def validate_normalization_metadata(data: dict) -> list:
    """Return Issues from validating a NormalizationMetadata dict."""
    try:
        DCANormalizationMetadata.model_validate(data)
        return []
    except pydantic.ValidationError as exc:
        return _convert_errors(exc)


@pytest.mark.unit
class TestGridPositive:
    def test_grid_one_passes(self):
        assert validate_normalization_metadata({"grid": 1}) == []

    def test_grid_large_passes(self):
        assert validate_normalization_metadata({"grid": 16}) == []

    def test_grid_zero_errors(self):
        errs = errors(validate_normalization_metadata({"grid": 0}))
        assert len(errs) >= 1

    def test_grid_negative_errors(self):
        errs = errors(validate_normalization_metadata({"grid": -1}))
        assert len(errs) >= 1

    def test_grid_none_passes(self):
        assert validate_normalization_metadata({"grid": None}) == []


# ---------------------------------------------------------------------------
# Regression: dca MUST check must fire even when levels has [SHOULD] warnings
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDcaMissingWithLevelWarnings:
    """
    Regression for the @model_validator(mode="after") silent-skip bug.

    When levels contain [SHOULD] violations (e.g. chunk size between 512 KB
    and 1 MB), Pydantic v2 skips mode="after" model validators on the parent
    because nested field validation raised. The fix converts check_dca_present
    to a @field_validator with validate_default=True so it runs unconditionally.
    """

    def test_dca_error_fires_alongside_chunk_size_warning(self):
        # chunk channel dim 2 (!= recommended 1) → [SHOULD] channel-chunk warning
        level = valid_level(
            shape=[1, 2, 128, 128, 128],
            chunk_shape=[1, 2, 128, 128, 128],
            dtype="uint16",
        )
        data = {
            "spec_version": "0.2",
            "axes": ["t", "c", "z", "y", "x"],
            "levels": [level],
            # dca intentionally absent
        }
        issues = validate_store(data)
        dca_errors = [
            i
            for i in issues
            if i.severity == Severity.ERROR and "dca" in i.message.lower()
        ]
        chunk_warnings = [
            i
            for i in issues
            if i.severity == Severity.WARNING and "chunk" in i.message.lower()
        ]
        assert dca_errors, "Missing dca ERROR was not reported alongside chunk warnings"
        assert chunk_warnings, "Chunk size WARNING was not reported"


# ---------------------------------------------------------------------------
# DCALabelStoreSpecV0_2 — axes, coarse-level, downsampling, no dca field
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLabelStoreSpecV0_2:
    """
    DCALabelStoreSpecV0_2 shares the structural rules of DCAStoreSpecV0_2
    (axes MUST, coarse-level MUST, downsampling MUST/SHOULD) but carries no
    dca field — label groups do not carry channel metadata.
    """

    def test_valid_label_store_passes(self):
        assert validate_label_store(valid_label_store()) == []

    def test_dca_absent_passes(self):
        # No dca field — label stores are not expected to have one
        data = valid_label_store()
        assert "dca" not in data
        assert errors(validate_label_store(data)) == []

    def test_dca_present_is_ignored(self):
        # extra="allow" — a dca key is silently accepted, not validated
        data = {
            **valid_label_store(),
            "dca": {
                "version": "0.2",
                "url": _SPEC_URL,
                "channels": [],
                "normalization_statistics": {},
            },
        }
        assert errors(validate_label_store(data)) == []

    def test_wrong_axes_errors(self):
        data = valid_label_store()
        data["axes"] = ["z", "y", "x", "t", "c"]
        assert len(errors(validate_label_store(data))) >= 1

    def test_uppercase_axes_passes(self):
        data = valid_label_store()
        data["axes"] = ["T", "C", "Z", "Y", "X"]
        assert validate_label_store(data) == []

    def test_mixed_case_axes_errors(self):
        data = valid_label_store()
        data["axes"] = ["T", "C", "Z", "y", "x"]
        errs = errors(validate_label_store(data))
        assert any("Axes must be" in e.message for e in errs)

    def test_all_levels_above_2048_fails(self):
        levels = [
            valid_label_level("0", shape=[1, 1, 8192, 8192, 8192]),
            valid_label_level("1", shape=[1, 1, 4096, 4096, 4096]),
            valid_label_level("2", shape=[1, 1, 2049, 2049, 2049]),
        ]
        data = {**valid_label_store(), "levels": levels}
        errs = errors(validate_label_store(data))
        assert any(str(COARSE_LEVEL_MAX_SPATIAL) in e.message for e in errs)

    def test_single_level_within_2048_passes(self):
        data = valid_label_store(n_levels=1)
        assert errors(validate_label_store(data)) == []

    def test_t_downsampled_errors(self):
        data = valid_label_store(n_levels=3)
        data["levels"][0]["shape"] = [2, 1, 256, 512, 512]
        data["levels"][1]["shape"] = [1, 1, 128, 256, 256]
        errs = errors(validate_label_store(data))
        assert any("'t'" in e.message for e in errs)

    def test_non_factor_2_spatial_warns(self):
        data = valid_label_store(n_levels=2)
        data["levels"][1]["shape"] = [1, 1, 200, 256, 256]
        warns = warnings(validate_label_store(data))
        assert any("'z'" in w.message for w in warns)

    def test_uint32_passes_no_dtype_warning(self):
        data = valid_label_store()
        assert not any(
            "uint32" in w.message for w in warnings(validate_label_store(data))
        )

    def test_uint16_levels_warn(self):
        levels = [
            valid_level(
                str(i), shape=[1, 1, 256 // (2**i), 512 // (2**i), 512 // (2**i)]
            )
            for i in range(3)
        ]
        data = {
            "spec_version": "0.2",
            "axes": ["t", "c", "z", "y", "x"],
            "levels": levels,
        }
        warns = warnings(validate_label_store(data))
        assert any("uint32" in w.message for w in warns)


# ---------------------------------------------------------------------------
# spec_version Literal["0.2"] pin
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpecVersionPin:
    def test_version_02_passes(self):
        assert validate_store(valid_store()) == []

    def test_version_01_errors(self):
        data = {**valid_store(), "spec_version": "0.1"}
        with pytest.raises(pydantic.ValidationError):
            DCAStoreSpecV0_2.model_validate(data)

    def test_version_03_errors(self):
        data = {**valid_store(), "spec_version": "0.3"}
        with pytest.raises(pydantic.ValidationError):
            DCAStoreSpecV0_2.model_validate(data)


# ---------------------------------------------------------------------------
# ome-zarr-metadata.rst § Dimensions — MUST (Gap #1)
# dimension_names in each array zarr.json must match axis names
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDimensionNamesMust:
    """
    dimension_names is MUST per ome-zarr-metadata.rst § Dimensions.
    The validator fires when dimension_names is explicitly provided (as it
    always is when data comes from _extract_level()).
    """

    def test_valid_dimension_names_passes(self):
        assert validate_level(valid_level()) == []

    def test_missing_dimension_names_errors(self):
        errs = errors(validate_level(valid_level(dimension_names=None)))
        assert len(errs) == 1
        assert "dimension_names" in errs[0].message

    def test_wrong_order_errors(self):
        errs = errors(
            validate_level(valid_level(dimension_names=["x", "y", "z", "c", "t"]))
        )
        assert len(errs) == 1
        assert "dimension_names" in errs[0].message

    def test_wrong_names_errors(self):
        errs = errors(
            validate_level(
                valid_level(dimension_names=["time", "channel", "z", "y", "x"])
            )
        )
        assert len(errs) == 1
        assert "dimension_names" in errs[0].message

    def test_uppercase_dimension_names_passes(self):
        assert (
            validate_level(valid_level(dimension_names=["T", "C", "Z", "Y", "X"])) == []
        )

    def test_mixed_case_dimension_names_errors(self):
        errs = errors(
            validate_level(valid_level(dimension_names=["T", "C", "Z", "y", "x"]))
        )
        assert len(errs) == 1
        assert "dimension_names" in errs[0].message

    def test_label_level_missing_dimension_names_errors(self):
        errs = errors(validate_label_level(valid_label_level(dimension_names=None)))
        assert len(errs) == 1
        assert "dimension_names" in errs[0].message

    def test_label_level_valid_dimension_names_passes(self):
        assert validate_label_level(valid_label_level()) == []

    def test_store_with_missing_dimension_names_errors(self):
        data = valid_store()
        data["levels"][0]["dimension_names"] = None
        errs = errors(validate_store(data))
        assert any("dimension_names" in e.message for e in errs)

    def test_null_entries_produce_must_error_not_type_error(self):
        # dimension_names with None entries (valid per zarr spec but invalid per DCA)
        # must produce the MUST message, not a generic Pydantic type coercion error
        errs = errors(
            validate_level(valid_level(dimension_names=[None, "c", "z", "y", "x"]))
        )
        assert len(errs) == 1
        assert "dimension_names" in errs[0].message


# ---------------------------------------------------------------------------
# ome-zarr-metadata.rst § Units — MUST (Gap #2)
# Spatial (z, y, x) and temporal (t) axes must declare valid units
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAxisUnitsMust:
    """
    axis_units MUST contain valid OME-NGFF v0.5 units for t, z, y, x.
    Channel axis 'c' has no unit requirement and is excluded from validation.
    """

    def test_valid_axis_units_passes(self):
        assert validate_store(valid_store()) == []

    def test_missing_spatial_unit_errors(self):
        for ax in ["z", "y", "x"]:
            units = {**VALID_AXIS_UNITS, ax: None}
            errs = errors(validate_store(valid_store(axis_units=units)))
            assert any(f"'{ax}'" in e.message for e in errs), ax

    def test_invalid_spatial_unit_errors(self):
        for ax in ["z", "y", "x"]:
            units = {**VALID_AXIS_UNITS, ax: "cubits"}
            errs = errors(validate_store(valid_store(axis_units=units)))
            assert any(
                f"'{ax}'" in e.message and "cubits" in e.message for e in errs
            ), ax

    def test_missing_temporal_unit_errors(self):
        units = {**VALID_AXIS_UNITS, "t": None}
        errs = errors(validate_store(valid_store(axis_units=units)))
        assert any("'t'" in e.message for e in errs)

    def test_invalid_temporal_unit_errors(self):
        units = {**VALID_AXIS_UNITS, "t": "megayear"}
        errs = errors(validate_store(valid_store(axis_units=units)))
        assert any("'t'" in e.message and "megayear" in e.message for e in errs)

    def test_channel_axis_no_unit_passes(self):
        units = {**VALID_AXIS_UNITS, "c": None}
        assert errors(validate_store(valid_store(axis_units=units))) == []

    def test_channel_axis_with_arbitrary_unit_passes(self):
        # c has no unit vocabulary constraint
        units = {**VALID_AXIS_UNITS, "c": "arbitrary_unit"}
        assert errors(validate_store(valid_store(axis_units=units))) == []

    def test_empty_axis_units_reports_all_four_missing(self):
        errs = errors(validate_store(valid_store(axis_units={})))
        axes_reported = {e.message for e in errs}
        assert any("'t'" in m for m in axes_reported)
        assert any("'z'" in m for m in axes_reported)
        assert any("'y'" in m for m in axes_reported)
        assert any("'x'" in m for m in axes_reported)

    @pytest.mark.parametrize("unit", sorted(SPATIAL_UNIT_VOCABULARY))
    def test_all_spatial_vocabulary_units_pass_for_z(self, unit):
        units = {**VALID_AXIS_UNITS, "z": unit}
        assert errors(validate_store(valid_store(axis_units=units))) == []

    @pytest.mark.parametrize("unit", sorted(TEMPORAL_UNIT_VOCABULARY))
    def test_all_temporal_vocabulary_units_pass_for_t(self, unit):
        units = {**VALID_AXIS_UNITS, "t": unit}
        assert errors(validate_store(valid_store(axis_units=units))) == []

    def test_uppercase_axis_units_keys_passes(self):
        units = {k.upper(): v for k, v in VALID_AXIS_UNITS.items()}
        assert errors(validate_store(valid_store(axis_units=units))) == []

    def test_label_store_has_no_axis_units_check(self):
        # DCALabelStoreSpecV0_2 does not validate axis_units — no errors expected
        data = valid_label_store()
        assert "axis_units" not in data
        assert errors(validate_label_store(data)) == []


# ---------------------------------------------------------------------------
# Variable time sampling
# ---------------------------------------------------------------------------


NO_T_UNIT = {**VALID_AXIS_UNITS, "t": None}


def variable_store(
    timestamps=(0.0, 7200.0, 14400.0),
    t=3,
    time_sampling="variable",
    t_scale=1.0,
    extra_acquisition=None,
):
    """A store declaring variable sampling, with T timepoints and no 't' unit."""
    store = valid_store(axis_units=NO_T_UNIT)
    for level in store["levels"]:
        level["shape"] = [t, *level["shape"][1:]]
    acquisition = {}
    if time_sampling is not None:
        acquisition["time_sampling"] = time_sampling
    if timestamps is not None:
        acquisition["timestamps"] = list(timestamps)
    if extra_acquisition:
        acquisition.update(extra_acquisition)
    store["dca"]["acquisition"] = acquisition
    store["axis_scales"] = {
        "t": t_scale,
        "c": 1.0,
        "z": 1.0,
        "y": 1.0,
        "x": 1.0,
    }
    return store


@pytest.mark.unit
class TestVariableTimeSampling:
    """
    Declaring acquisition.time_sampling = "variable" relaxes the temporal-unit MUST
    to a SHOULD and, in exchange, requires per-frame timestamps that match the array.
    Every rule here is inside the "variable" branch: a store that does not declare it
    is untouched.
    """

    def test_declaration_demotes_missing_t_unit_to_warning(self):
        issues = validate_store(variable_store())
        assert errors(issues) == []
        assert any("has no unit" in w.message for w in warnings(issues))

    def test_missing_t_unit_without_declaration_stays_an_error(self):
        errs = errors(validate_store(valid_store(axis_units=NO_T_UNIT)))
        assert any("Temporal axis 't' has no unit" in e.message for e in errs)

    def test_uniform_declaration_does_not_relax_the_unit_must(self):
        store = variable_store(time_sampling="uniform", timestamps=None)
        errs = errors(validate_store(store))
        assert any("Temporal axis 't' has no unit" in e.message for e in errs)

    def test_absent_acquisition_block_is_unaffected(self):
        # The common case for an existing store: no declaration, a real 't' unit,
        # so none of the rules in this class apply.
        assert validate_store(valid_store()) == []

    def test_variable_without_timestamps_is_an_error(self):
        errs = errors(validate_store(variable_store(timestamps=None)))
        assert any("'acquisition.timestamps' is absent" in e.message for e in errs)

    def test_should_and_must_are_reported_together(self):
        # Regression guard: the SHOULD and the timestamps MUSTs live in one
        # validator, because a second mode="after" validator would never run.
        issues = validate_store(variable_store(timestamps=None))
        assert any("has no unit" in w.message for w in warnings(issues))
        assert any("timestamps" in e.message for e in errors(issues))

    def test_timestamps_length_must_match_t(self):
        errs = errors(validate_store(variable_store(timestamps=(0.0, 1.0), t=3)))
        assert any("has 2 entries" in e.message and "T=3" in e.message for e in errs)

    def test_timestamps_must_be_non_decreasing(self):
        store = variable_store(timestamps=(0.0, 14400.0, 7200.0))
        errs = errors(validate_store(store))
        assert any("non-decreasing" in e.message for e in errs)

    def test_equal_timestamps_are_allowed(self):
        store = variable_store(timestamps=(0.0, 0.0, 7200.0))
        assert errors(validate_store(store)) == []

    def test_variable_requires_a_t_scale_of_one(self):
        errs = errors(validate_store(variable_store(t_scale=7200.0)))
        assert any("MUST be 1" in e.message for e in errs)

    def test_uppercase_t_axis_scale_is_still_checked(self):
        store = variable_store(t_scale=7200.0)
        store["axes"] = ["T", "C", "Z", "Y", "X"]
        store["axis_units"] = {k.upper(): v for k, v in store["axis_units"].items()}
        store["axis_scales"] = {k.upper(): v for k, v in store["axis_scales"].items()}
        errs = errors(validate_store(store))
        assert any("MUST be 1" in e.message for e in errs)

    def test_unreadable_t_scale_is_not_treated_as_a_violation(self):
        assert errors(validate_store(variable_store(t_scale=None))) == []

    def test_unknown_time_sampling_value_is_an_error(self):
        errs = errors(validate_store(variable_store(time_sampling="sometimes")))
        assert any("is not one of" in e.message for e in errs)

    def test_timestamps_are_checked_without_a_declaration(self):
        # A uniform or undeclared group may still carry per-frame times; a
        # mis-sized array is as wrong there as under `variable`.
        store = variable_store(timestamps=(0.0, 1.0), t=3, time_sampling=None)
        errs = errors(validate_store(store))
        assert any("has 2 entries" in e.message and "T=3" in e.message for e in errs)

    def test_timestamps_order_is_checked_under_uniform(self):
        store = variable_store(
            timestamps=(0.0, 14400.0, 7200.0), time_sampling="uniform"
        )
        assert any("non-decreasing" in e.message for e in errors(validate_store(store)))

    def test_well_formed_timestamps_under_uniform_pass(self):
        store = variable_store(
            timestamps=(0.0, 7200.0, 14400.0), time_sampling="uniform"
        )
        errs = errors(validate_store(store))
        # The only complaint is the absent 't' unit, which uniform does not excuse.
        # Nothing is reported against the array itself. (The unit error names
        # 'acquisition.timestamps' in its remedy text, so match on the array rules.)
        assert any("has no unit" in e.message for e in errs)
        assert not any(
            "entries but the" in e.message or "non-decreasing" in e.message
            for e in errs
        )

    def test_store_rules_do_not_re_validate_the_acquisition_block(self):
        # The store-structure model reads two fields and reports nothing else about
        # acquisition. Block-level validation belongs to
        # validate_dca_experimental_metadata_v0_2, which _run_dca_validation runs
        # alongside this model and whose issues are concatenated with these — so
        # validating the block here too would report each defect twice.
        # 'some_future_key' is an ERROR there (the generated model is extra="forbid");
        # what this asserts is only that it is not *also* an error from here.
        store = variable_store(
            extra_acquisition={
                "microscope": {"model": "ExampleScope 1"},
                "some_future_key": 1,
            }
        )
        assert errors(validate_store(store)) == []


# ---------------------------------------------------------------------------
# normalization-statistics.rst § Channel Normalization Fields — MUST
# timepoint_statistics, if present, covers every timepoint
# ---------------------------------------------------------------------------


def timepoint_store(n_timepoints, covered, channels=1):
    store = valid_store(c=channels)
    for level in store["levels"]:
        level["shape"][0] = n_timepoints
    stats = {"p1": 1.0, "p5": 5.0, "p95": 95.0, "p99": 99.0}
    for channel in store["dca"]["normalization_statistics"].values():
        channel["timepoint_statistics"] = {str(t): dict(stats) for t in covered}
    return store


@pytest.mark.unit
class TestTimepointStatisticsCompleteMust:
    def test_full_coverage_passes(self):
        store = timepoint_store(4, range(4))
        assert errors(validate_store(store)) == []

    def test_absent_timepoint_statistics_passes(self):
        store = timepoint_store(4, range(4))
        del store["dca"]["normalization_statistics"]["0"]["timepoint_statistics"]
        assert errors(validate_store(store)) == []

    def test_partial_coverage_errors(self):
        store = timepoint_store(100, range(50))
        errs = errors(validate_store(store))
        assert len(errs) == 1
        assert "missing 50 of 100 timepoints" in errs[0].message
        assert "first missing: '50'" in errs[0].message

    def test_gap_in_coverage_errors(self):
        store = timepoint_store(4, [0, 1, 3])
        errs = errors(validate_store(store))
        assert len(errs) == 1
        assert "first missing: '2'" in errs[0].message

    def test_empty_timepoint_statistics_errors(self):
        store = timepoint_store(3, [])
        errs = errors(validate_store(store))
        assert len(errs) == 1
        assert "missing 3 of 3 timepoints" in errs[0].message

    def test_each_incomplete_channel_is_reported(self):
        store = timepoint_store(3, [0], channels=2)
        errs = errors(validate_store(store))
        assert len(errs) == 2
        assert "['0']" in errs[0].message
        assert "['1']" in errs[1].message

    def test_single_timepoint_requires_entry_zero(self):
        store = timepoint_store(1, ["1"])
        messages = [e.message for e in errors(validate_store(store))]
        assert len(messages) == 2
        assert any("first missing: '0'" in m for m in messages)
        assert any("first unexpected: '1'" in m for m in messages)

    def test_malformed_timepoint_statistics_is_not_a_completeness_error(self):
        store = timepoint_store(3, range(3))
        store["dca"]["normalization_statistics"]["0"]["timepoint_statistics"] = []
        assert not any(
            "timepoints" in e.message and "MUST contain" in e.message
            for e in errors(validate_store(store))
        )


def with_extra_timepoint_key(store, key):
    stats = {"p1": 1.0, "p5": 5.0, "p95": 95.0, "p99": 99.0}
    store["dca"]["normalization_statistics"]["0"]["timepoint_statistics"][key] = stats
    return store


@pytest.mark.unit
class TestTimepointStatisticsNoExtraKeysMust:
    def test_index_equal_to_t_errors(self):
        store = with_extra_timepoint_key(timepoint_store(4, range(4)), "4")
        errs = errors(validate_store(store))
        assert len(errs) == 1
        assert "first unexpected: '4'" in errs[0].message
        assert "'0' to '3'" in errs[0].message

    def test_non_digit_key_errors(self):
        store = with_extra_timepoint_key(timepoint_store(4, range(4)), "foo")
        errs = errors(validate_store(store))
        assert len(errs) == 1
        assert "first unexpected: 'foo'" in errs[0].message

    def test_non_canonical_index_errors(self):
        store = with_extra_timepoint_key(timepoint_store(4, range(4)), "01")
        errs = errors(validate_store(store))
        assert len(errs) == 1
        assert "first unexpected: '01'" in errs[0].message

    def test_integer_keys_are_rejected(self):
        store = timepoint_store(2, [])
        store["dca"]["normalization_statistics"]["0"]["timepoint_statistics"] = {
            0: {"p1": 1.0, "p5": 5.0, "p95": 95.0, "p99": 99.0},
            1: {"p1": 1.0, "p5": 5.0, "p95": 95.0, "p99": 99.0},
        }
        messages = [e.message for e in errors(validate_store(store))]
        assert any("Input should be a valid string" in m for m in messages)

    def test_missing_and_unexpected_are_both_reported(self):
        store = with_extra_timepoint_key(timepoint_store(3, [0, 1]), "7")
        messages = [e.message for e in errors(validate_store(store))]
        assert len(messages) == 2
        assert any("missing 1 of 3 timepoints" in m for m in messages)
        assert any("first unexpected: '7'" in m for m in messages)
