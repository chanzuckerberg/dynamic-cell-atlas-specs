"""
Unit tests for versions/v0_3/models.py: the dca.projection model (uniform and
per-channel forms, plane ranges, focus provenance) and the group-level projection
rules (check_projection_v0_3).
"""

import pydantic
import pytest

from dca_helpers.validation.versions.v0_3.models import (
    DCAGroupMetadataV0_3,
    DCAProjectionV0_3,
    check_projection_v0_3,
)

MIP = "maximum intensity projection"
MEAN = "mean intensity projection"
SLICE = "z slice"
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


def _per_channel(**overrides):
    base = {
        "source_store": "plate.ome.zarr",
        "z_dim_source": 86,
        "channels": {
            "membrane": {"method": "max", "z_start": 0, "z_end": 86},
            "nuclei": {"method": "mean", "z_start": [0, 0, 0], "z_end": [59, 58, 60]},
            "mCherry": {"method": "slice", "z_start": 42, "z_end": 43},
        },
        "focus": {"channel": "Phase3D", "per": "timepoint", "z_per_t": [19, 18, 20]},
    }
    return {k: v for k, v in {**base, **overrides}.items() if v is not None}


def _node(z_sizes=(1, 1), t=3, c=3):
    return {
        "axes": AXES,
        "array_shapes": [[t, c, z, 64 >> i, 64 >> i] for i, z in enumerate(z_sizes)],
        "axis_scales": {"t": 1.0, "c": 1.0, "z": 2.0, "y": 0.5, "x": 0.5},
    }


def _dca(steps=("deconvolved", MIP), projection="default", channels=None):
    dca = {"image_data": {"processing_variant": list(steps)}}
    if projection == "default":
        dca["projection"] = _projection()
    elif projection is not None:
        dca["projection"] = projection
    if channels is not None:
        dca["channels"] = [{"name": n, "index": i} for i, n in enumerate(channels)]
    return dca


def _sev(issues):
    return sorted(i.severity.value for i in issues)


def _valid(data):
    DCAProjectionV0_3.model_validate(data)


def _invalid(data, match):
    with pytest.raises(pydantic.ValidationError, match=match):
        DCAProjectionV0_3.model_validate(data)


@pytest.mark.unit
class TestUniformForm:
    def test_full_block_passes(self):
        _valid(_projection())

    def test_method_and_source_store_only_passes(self):
        _valid({"method": "max", "source_store": "a.ome.zarr"})

    @pytest.mark.parametrize("method", ["max", "mean", "sum"])
    def test_every_reducing_method_passes(self, method):
        _valid(_projection(method=method))

    @pytest.mark.parametrize("method", ["min", "median"])
    def test_methods_outside_the_shared_four_error(self, method):
        _invalid(_projection(method=method), "max")

    def test_full_url_source_store_passes(self):
        _valid(_projection(source_store="s3://bucket/datasets/plate.ome.zarr"))

    def test_missing_source_store_passes_per_store(self):
        # Required only when the 3D source is in the DCA, which the cross-store pass
        # checks; on its own a block without it is a projection made before deposit.
        data = _projection()
        del data["source_store"]
        _valid(data)

    def test_z_start_without_z_end_errors(self):
        _invalid(_projection(z_end=None), "together")

    def test_range_without_z_dim_source_errors(self):
        _invalid(_projection(z_dim_source=None), "z_dim_source MUST be given")

    @pytest.mark.parametrize(
        "z_start,z_end,z_dim", [(5, 5, 10), (6, 5, 10), (0, 11, 10)]
    )
    def test_bad_z_range_errors(self, z_start, z_end, z_dim):
        _invalid(
            _projection(z_start=z_start, z_end=z_end, z_dim_source=z_dim),
            "invalid",
        )

    def test_z_step_passes(self):
        _valid(_projection(z_step=2, z_fraction=None))

    def test_z_step_without_range_errors(self):
        _invalid({"method": "max", "source_store": "a", "z_step": 2}, "z_step")

    def test_slice_of_one_plane_passes(self):
        _valid(_projection(method="slice", z_start=7, z_end=8, z_fraction=None))

    def test_slice_of_several_planes_errors(self):
        _invalid(_projection(method="slice", z_fraction=None), "MUST be 1")

    def test_slice_without_planes_errors(self):
        _invalid({"method": "slice", "source_store": "a"}, "MUST give z_start")

    def test_group_metadata_requires_version_0_3(self):
        assert DCAGroupMetadataV0_3.model_fields["version"].annotation.__args__ == (
            "0.3",
        )


@pytest.mark.unit
class TestPerChannelForm:
    def test_resolved_per_channel_block_passes(self):
        _valid(_per_channel())

    def test_without_focus_passes(self):
        # focus is provenance; the planes are in z_start / z_end.
        _valid(_per_channel(focus=None))

    def test_channels_without_planes_pass(self):
        _valid({"source_store": "a", "channels": {"GFP": {"method": "max"}}})

    def test_method_and_channels_together_errors(self):
        _invalid(_per_channel(method="max"), "exactly one")

    def test_neither_method_nor_channels_errors(self):
        _invalid({"source_store": "a"}, "exactly one")

    @pytest.mark.parametrize("field,value", [("z_start", 0), ("z_fraction", 1.0)])
    def test_uniform_fields_in_per_channel_form_error(self, field, value):
        _invalid(_per_channel(**{field: value}), "uniform form")

    def test_channel_range_without_z_dim_source_errors(self):
        _invalid(_per_channel(z_dim_source=None), "z_dim_source MUST be given")

    def test_channel_range_past_the_stack_errors(self):
        channels = {"a": {"method": "max", "z_start": 80, "z_end": 90}}
        _invalid(_per_channel(channels=channels), "invalid")

    def test_mixed_int_and_list_errors(self):
        channels = {"a": {"method": "mean", "z_start": 0, "z_end": [5, 6]}}
        _invalid(_per_channel(channels=channels), "both be integers or both be lists")

    def test_list_lengths_must_agree(self):
        channels = {"a": {"method": "mean", "z_start": [0, 0], "z_end": [5, 6, 7]}}
        _invalid(_per_channel(channels=channels), "one entry per timepoint")

    def test_per_timepoint_slice_passes(self):
        channels = {"a": {"method": "slice", "z_start": [19, 18], "z_end": [20, 19]}}
        _valid(_per_channel(channels=channels))

    def test_slice_without_planes_errors(self):
        channels = {"a": {"method": "slice"}}
        _invalid(_per_channel(channels=channels), "MUST give z_start")

    def test_focus_per_fov_needs_its_plane(self):
        _invalid(_per_channel(focus={"channel": "Phase3D", "per": "fov"}), "focus.z")

    def test_focus_per_timepoint_needs_its_planes(self):
        _invalid(
            _per_channel(focus={"channel": "Phase3D", "per": "timepoint"}), "z_per_t"
        )


@pytest.mark.unit
class TestCheckProjection:
    def test_valid_projection_is_clean(self):
        assert check_projection_v0_3(_node(), _dca(), {}) == []

    def test_plain_image_is_clean(self):
        dca = _dca(steps=("deconvolved",), projection=None)
        assert check_projection_v0_3(_node(z_sizes=(16, 8)), dca, {}) == []

    def test_projection_step_without_block_warns(self):
        # Fine for an image projected before deposit; an error only when the 3D
        # source is in the dataset (checked across stores).
        issues = check_projection_v0_3(_node(), _dca(projection=None), {})
        assert _sev(issues) == ["warning"]
        assert issues[0].loc == ("dca", "projection")

    def test_block_without_projection_step_errors(self):
        issues = check_projection_v0_3(_node(), _dca(steps=("deconvolved",)), {})
        assert _sev(issues) == ["error"]
        assert issues[0].loc == ("dca", "image_data", "processing_variant")

    def test_step_must_match_method(self):
        dca = _dca(projection=_projection(method="mean"))  # lists the MIP step
        issues = check_projection_v0_3(_node(), dca, {})
        assert _sev(issues) == ["error"]
        assert "mean intensity projection" in issues[0].message

    def test_mixed_methods_list_every_step(self):
        dca = _dca(
            steps=(MIP, MEAN, SLICE),
            projection=_per_channel(),
            channels=["membrane", "nuclei", "mCherry"],
        )
        assert check_projection_v0_3(_node(), dca, {}) == []

    def test_mixed_methods_missing_a_step_errors(self):
        dca = _dca(
            steps=(MIP, MEAN),
            projection=_per_channel(),
            channels=["membrane", "nuclei", "mCherry"],
        )
        assert _sev(check_projection_v0_3(_node(), dca, {})) == ["error"]

    def test_per_channel_key_that_is_not_a_channel_errors(self):
        dca = _dca(
            steps=(MIP, MEAN, SLICE),
            projection=_per_channel(),
            channels=["membrane", "nuclei", "RFP"],
        )
        issues = check_projection_v0_3(_node(), dca, {})
        assert _sev(issues) == ["error"]
        assert "['mCherry']" in issues[0].message

    def test_per_channel_keys_may_be_a_subset(self):
        # Phase2D (a reconstruction) is a channel but was not projected along z.
        projection = {
            "z_dim_source": 7,
            "channels": {"GFP": {"method": "max", "z_start": 0, "z_end": 7}},
        }
        dca = _dca(steps=(MIP,), projection=projection, channels=["Phase2D", "GFP"])
        assert check_projection_v0_3(_node(c=2), dca, {}) == []

    def test_per_timepoint_planes_must_cover_every_timepoint(self):
        dca = _dca(
            steps=(MIP, MEAN, SLICE),
            projection=_per_channel(focus=None),
            channels=["membrane", "nuclei", "mCherry"],
        )
        issues = check_projection_v0_3(_node(t=4), dca, {})
        assert _sev(issues) == ["error", "error"]  # z_start and z_end
        assert all("T=4" in i.message for i in issues)

    def test_focus_z_per_t_must_cover_every_timepoint(self):
        projection = _per_channel(
            channels={"membrane": {"method": "max"}},
            focus={"channel": "Phase3D", "per": "timepoint", "z_per_t": [19, 18]},
        )
        dca = _dca(steps=(MIP,), projection=projection, channels=["membrane"])
        issues = check_projection_v0_3(_node(c=1), dca, {})
        assert _sev(issues) == ["error"]
        assert "z_per_t has 2 entries" in issues[0].message

    def test_z_not_one_at_some_level_errors(self):
        issues = check_projection_v0_3(_node(z_sizes=(1, 2)), _dca(), {})
        assert _sev(issues) == ["error"]
        assert "(1, 2)" in issues[0].message

    def test_any_z_scale_is_accepted(self):
        # _node uses z scale 2.0 (copied from a source): not a warning.
        assert check_projection_v0_3(_node(), _dca(), {}) == []

    def test_legacy_top_level_mip_errors(self):
        issues = check_projection_v0_3(_node(), _dca(), {"mip": {"method": "max"}})
        assert _sev(issues) == ["error"]
        assert issues[0].loc == ("mip",)

    def test_z_fraction_disagreeing_with_range_warns(self):
        dca = _dca(projection=_projection(z_fraction=0.5))
        assert _sev(check_projection_v0_3(_node(), dca, {})) == ["warning"]

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
