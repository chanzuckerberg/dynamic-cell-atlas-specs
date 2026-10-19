"""
Unit tests for versions/v0_3/models.py: the dca.projection model and the
group-level projection rules (check_projection_v0_3).
"""

import pydantic
import pytest

from dca_helpers.validation.versions.v0_3.models import (
    DCAGroupMetadataV0_3,
    DCAProjectionV0_3,
    check_projection_v0_3,
)

MIP = "maximum intensity projection"
AXES = ["t", "c", "z", "y", "x"]


def _projection(**overrides):
    base = {
        "source_store": "plate.ome.zarr",
        "method": "max",
        "z_fraction": 1.0,
        "z_start": 0,
        "z_end": 16,
        "z_dim_source": 16,
    }
    return {k: v for k, v in {**base, **overrides}.items() if v is not None}


def _node(z_sizes=(1, 1), z_scale=1.0):
    return {
        "axes": AXES,
        "array_shapes": [[1, 3, z, 64 >> i, 64 >> i] for i, z in enumerate(z_sizes)],
        "axis_scales": {"t": 1.0, "c": 1.0, "z": z_scale, "y": 0.5, "x": 0.5},
    }


def _dca(steps=("deconvolved", MIP), projection="default"):
    dca = {"image_data": {"processing_variant": list(steps)}}
    if projection == "default":
        dca["projection"] = _projection()
    elif projection is not None:
        dca["projection"] = projection
    return dca


def _sev(issues):
    return sorted(i.severity.value for i in issues)


@pytest.mark.unit
class TestProjectionModel:
    def test_full_block_passes(self):
        DCAProjectionV0_3.model_validate(_projection())

    def test_method_and_source_store_only_passes(self):
        DCAProjectionV0_3.model_validate(
            {"method": "max", "source_store": "a.ome.zarr"}
        )

    def test_full_url_source_store_passes(self):
        DCAProjectionV0_3.model_validate(
            _projection(source_store="s3://bucket/datasets/plate.ome.zarr")
        )

    @pytest.mark.parametrize("missing", ["method", "source_store"])
    def test_required_field_missing_errors(self, missing):
        data = _projection()
        del data[missing]
        with pytest.raises(pydantic.ValidationError):
            DCAProjectionV0_3.model_validate(data)

    def test_method_other_than_max_errors(self):
        with pytest.raises(pydantic.ValidationError, match="max"):
            DCAProjectionV0_3.model_validate(_projection(method="mean"))

    def test_partial_z_range_errors(self):
        data = {"method": "max", "source_store": "a", "z_start": 0}
        with pytest.raises(pydantic.ValidationError, match="together"):
            DCAProjectionV0_3.model_validate(data)

    @pytest.mark.parametrize(
        "z_start,z_end,z_dim", [(5, 5, 10), (6, 5, 10), (0, 11, 10)]
    )
    def test_bad_z_range_errors(self, z_start, z_end, z_dim):
        with pytest.raises(pydantic.ValidationError, match="range is invalid"):
            DCAProjectionV0_3.model_validate(
                _projection(z_start=z_start, z_end=z_end, z_dim_source=z_dim)
            )

    def test_group_metadata_requires_version_0_3(self):
        assert DCAGroupMetadataV0_3.model_fields["version"].annotation.__args__ == (
            "0.3",
        )


@pytest.mark.unit
class TestCheckProjection:
    def test_valid_projection_is_clean(self):
        assert check_projection_v0_3(_node(), _dca(), {}) == []

    def test_plain_image_is_clean(self):
        dca = _dca(steps=("deconvolved",), projection=None)
        assert check_projection_v0_3(_node(z_sizes=(16, 8)), dca, {}) == []

    def test_mip_step_without_block_errors(self):
        issues = check_projection_v0_3(_node(), _dca(projection=None), {})
        assert _sev(issues) == ["error"]
        assert issues[0].loc == ("dca", "projection")

    def test_block_without_mip_step_errors(self):
        issues = check_projection_v0_3(_node(), _dca(steps=("deconvolved",)), {})
        assert _sev(issues) == ["error"]
        assert issues[0].loc == ("dca", "image_data", "processing_variant")

    def test_z_not_one_at_some_level_errors(self):
        issues = check_projection_v0_3(_node(z_sizes=(1, 2)), _dca(), {})
        assert _sev(issues) == ["error"]
        assert "(1, 2)" in issues[0].message

    def test_z_scale_not_one_warns(self):
        issues = check_projection_v0_3(_node(z_scale=2.0), _dca(), {})
        assert _sev(issues) == ["warning"]

    def test_legacy_top_level_mip_errors(self):
        issues = check_projection_v0_3(_node(), _dca(), {"mip": {"method": "max"}})
        assert _sev(issues) == ["error"]
        assert issues[0].loc == ("mip",)

    def test_z_fraction_disagreeing_with_range_warns(self):
        dca = _dca(projection=_projection(z_fraction=0.5))
        issues = check_projection_v0_3(_node(), dca, {})
        assert _sev(issues) == ["warning"]

    def test_central_slab_z_fraction_agrees(self):
        # 60% of 50 planes, as a central slab: planes 10..39.
        dca = _dca(
            projection=_projection(
                z_fraction=0.6, z_start=10, z_end=40, z_dim_source=50
            )
        )
        assert check_projection_v0_3(_node(), dca, {}) == []

    def test_no_z_range_is_clean(self):
        dca = _dca(projection={"method": "max", "source_store": "plate.ome.zarr"})
        assert check_projection_v0_3(_node(), dca, {}) == []

    def test_label_group_without_dca_is_clean(self):
        assert check_projection_v0_3(_node(z_sizes=(16,)), None, {}) == []
