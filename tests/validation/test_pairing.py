"""
Functional tests for the cross-store projection checks (validation/pairing.py)
and --require-projection, against v0.3 source / projection pairs written to
tmp_path by projection_fixtures.make_pair (uniform form) and to_per_channel.
"""

import shutil

import pytest
from click.testing import CliRunner

from dca_helpers.validation.cli import main
from dca_helpers.validation.result import Severity
from dca_helpers.validation.runner import validate

from .projection_fixtures import (
    FIELD,
    MIP_STEP,
    edit_field_attrs,
    make_pair,
    read_json,
    to_per_channel,
    write_json,
)

# The v0.2 fixture this builds on has no provenance records, so every run also
# carries provenance-coverage warnings; the assertions below look only at the
# issues these checks add.
PAIRING_LOC_ROOTS = {"projection", "dca", "mip", "levels", "axis_scales"}


def _run(path, **kw):
    return validate(str(path), spec_version="0.3", show_progress=False, **kw)


def _issues(run, severity=None):
    out = []
    for r in run:
        for i in r.issues:
            if (
                i.loc
                and i.loc[0] in PAIRING_LOC_ROOTS
                and i.loc[:2] != ("dca", "provenance")
            ):
                if severity is None or i.severity == severity:
                    out.append(i)
    return out


def _messages(run, severity=Severity.ERROR):
    return [i.message for i in _issues(run, severity)]


@pytest.mark.functional
class TestValidPair:
    def test_pair_passes(self, tmp_path):
        make_pair(tmp_path)
        run = _run(tmp_path)
        assert run.summary.stores_failed == 0
        assert _issues(run) == []

    def test_pair_passes_with_require_projection(self, tmp_path):
        make_pair(tmp_path)
        run = _run(tmp_path, require_projection=True)
        assert run.summary.stores_failed == 0
        assert _issues(run) == []

    def test_source_store_as_absolute_path_passes(self, tmp_path):
        source = tmp_path / "plate.ome.zarr"

        def absolute(mip):
            edit_field_attrs(
                mip,
                lambda a: a["dca"]["projection"].update(source_store=str(source)),
            )

        make_pair(tmp_path, edit_mip=absolute)
        run = _run(tmp_path, require_projection=True)
        assert _issues(run) == []

    def test_source_store_url_matching_a_sibling_by_name_passes(self, tmp_path):
        # A local mirror of a remote dataset: the URL is not in the run, but the
        # same-named sibling is.
        def remote(mip):
            edit_field_attrs(
                mip,
                lambda a: a["dca"]["projection"].update(
                    source_store="s3://bucket/release/plate.ome.zarr"
                ),
            )

        make_pair(tmp_path, edit_mip=remote)
        run = _run(tmp_path, require_projection=True)
        assert _issues(run) == []


@pytest.mark.functional
class TestSourceResolution:
    def test_missing_source_is_error(self, tmp_path):
        _, mip = make_pair(tmp_path)
        shutil.rmtree(tmp_path / "plate.ome.zarr")
        run = _run(tmp_path)
        assert any("does not exist" in m for m in _messages(run))

    def test_source_outside_the_run_is_warning(self, tmp_path):
        _, mip = make_pair(tmp_path)
        run = _run(mip)  # only the projection store is validated
        warnings = _messages(run, Severity.WARNING)
        assert any("was not part of this run" in m for m in warnings)
        assert _messages(run) == []

    def test_store_name_without_suffix_warns(self, tmp_path):
        make_pair(tmp_path, mip_name="plate_projection.ome.zarr")
        run = _run(tmp_path)
        assert any("_proj.ome.zarr" in m for m in _messages(run, Severity.WARNING))

    def test_proj_suffix_passes(self, tmp_path):
        make_pair(tmp_path, mip_name="plate_proj.ome.zarr")
        assert _issues(_run(tmp_path)) == []

    def test_mip_suffix_on_a_mean_projection_warns(self, tmp_path):
        def to_mean(mip):
            def edit(a):
                a["dca"]["projection"]["method"] = "mean"
                a["dca"]["image_data"]["processing_variant"] = [
                    "deconvolved",
                    "mean intensity projection",
                ]

            edit_field_attrs(mip, edit)

        make_pair(tmp_path, edit_mip=to_mean)
        warnings = _messages(_run(tmp_path), Severity.WARNING)
        assert any("_proj.ome.zarr" in m for m in warnings)


@pytest.mark.functional
class TestPairMismatches:
    def test_extra_source_field_is_hierarchy_error(self, tmp_path):
        def add_field(source):
            shutil.copytree(source / FIELD, source / "A/1/1")
            well = read_json(source / "A/1/zarr.json")
            well["attributes"]["ome"]["well"]["images"].append(
                {"path": "1", "acquisition": 0}
            )
            write_json(source / "A/1/zarr.json", well)

        make_pair(tmp_path, edit_source=add_field)
        run = _run(tmp_path)
        assert any("missing from the projection: A/1/1" in m for m in _messages(run))

    def test_wrong_z_dim_source_is_error(self, tmp_path):
        def wrong(mip):
            edit_field_attrs(
                mip,
                lambda a: a["dca"]["projection"].update(z_dim_source=20, z_end=20),
            )

        make_pair(tmp_path, edit_mip=wrong)
        run = _run(tmp_path)
        assert any("z_dim_source is 20" in m for m in _messages(run))

    def test_dtype_mismatch_is_error(self, tmp_path):
        def to_uint8(mip):
            for level in ("0", "1", "2"):
                p = mip / FIELD / level / "zarr.json"
                arr = read_json(p)
                arr["data_type"] = "uint8"
                write_json(p, arr)

        make_pair(tmp_path, edit_mip=to_uint8)
        run = _run(tmp_path)
        assert any("dtype uint8 differs" in m for m in _messages(run))

    def test_yx_scale_mismatch_is_error(self, tmp_path):
        def rescale(mip):
            def edit(a):
                ds = a["ome"]["multiscales"][0]["datasets"][0]
                ds["coordinateTransformations"][0]["scale"][4] = 2.0

            edit_field_attrs(mip, edit)

        make_pair(tmp_path, edit_mip=rescale)
        run = _run(tmp_path)
        assert any("'x' scale" in m for m in _messages(run))


@pytest.mark.functional
class TestGroupRulesEndToEnd:
    def test_legacy_top_level_mip_is_error(self, tmp_path):
        def legacy(mip):
            edit_field_attrs(mip, lambda a: a.update(mip={"method": "max"}))

        make_pair(tmp_path, edit_mip=legacy)
        run = _run(tmp_path)
        assert any("Top-level 'mip'" in m for m in _messages(run))

    def test_projection_block_without_projection_step_is_error(self, tmp_path):
        def drop_step(mip):
            edit_field_attrs(
                mip,
                lambda a: a["dca"]["image_data"].update(
                    processing_variant=["deconvolved"]
                ),
            )

        make_pair(tmp_path, edit_mip=drop_step)
        run = _run(tmp_path)
        assert run.summary.stores_failed > 0
        assert any("lists no projection step" in m for m in _messages(run))


@pytest.mark.functional
class TestRequireProjection:
    def test_3d_store_without_projection_is_error(self, tmp_path):
        make_pair(tmp_path, with_mip=False)
        run = _run(tmp_path, require_projection=True)
        assert run.summary.stores_failed == 1
        assert any("--require-projection" in m for m in _messages(run))

    def test_without_flag_a_lone_3d_store_passes(self, tmp_path):
        make_pair(tmp_path, with_mip=False)
        run = _run(tmp_path)
        assert run.summary.stores_failed == 0

    def test_every_3d_variant_needs_its_own_projection(self, tmp_path):
        make_pair(tmp_path)
        make_pair(tmp_path, source_name="plate_trimmed.ome.zarr", with_mip=False)
        run = _run(tmp_path, require_projection=True)
        failed = [r.node_path for r in run if not r.passed]
        assert [p.rsplit("/", 1)[-1] for p in failed] == ["plate_trimmed.ome.zarr"]

    def test_requires_a_projection_spec_version(self, tmp_path):
        make_pair(tmp_path, with_mip=False)
        with pytest.raises(ValueError, match="require_projection"):
            validate(
                str(tmp_path),
                spec_version="0.2",
                show_progress=False,
                require_projection=True,
            )

    def test_cli_flag(self, tmp_path):
        make_pair(tmp_path, with_mip=False)
        result = CliRunner().invoke(
            main, [str(tmp_path), "--spec-version", "0.3", "--require-projection"]
        )
        assert result.exit_code == 1
        assert "--require-projection" in result.output

    def test_cli_require_mip_is_an_alias(self, tmp_path):
        make_pair(tmp_path, with_mip=False)
        result = CliRunner().invoke(
            main, [str(tmp_path), "--spec-version", "0.3", "--require-mip"]
        )
        assert result.exit_code == 1

    def test_cli_flag_with_v0_2_is_usage_error(self, tmp_path):
        make_pair(tmp_path, with_mip=False)
        result = CliRunner().invoke(
            main, [str(tmp_path), "--spec-version", "0.2", "--require-projection"]
        )
        assert result.exit_code == 2


MEAN_STEP = "mean intensity projection"
SLICE_STEP = "z slice"


def _per_channel(channels, steps, **kw):
    return lambda mip: to_per_channel(mip, channels, steps, **kw)


@pytest.mark.functional
class TestPerChannel:
    """Per-channel projections: a channel subset with mixed methods and resolved planes."""

    MIXED = {
        "GFP": {"method": "max"},
        "DAPI": {"method": "slice", "z_start": 8, "z_end": 9},
    }

    def test_subset_with_mixed_methods_passes(self, tmp_path):
        make_pair(
            tmp_path,
            mip_name="plate_proj.ome.zarr",
            edit_mip=_per_channel(self.MIXED, ["deconvolved", MIP_STEP, SLICE_STEP]),
        )
        run = _run(tmp_path, require_projection=True)
        assert _messages(run) == []
        assert run.summary.stores_failed == 0

    def test_mean_in_float32_passes(self, tmp_path):
        channels = {"GFP": {"method": "mean", "z_start": 4, "z_end": 12}}
        make_pair(
            tmp_path,
            mip_name="plate_proj.ome.zarr",
            edit_mip=_per_channel(channels, [MEAN_STEP], dtype="float32"),
        )
        assert _messages(_run(tmp_path)) == []

    def test_max_only_in_float32_is_dtype_error(self, tmp_path):
        make_pair(
            tmp_path,
            mip_name="plate_proj.ome.zarr",
            edit_mip=_per_channel(
                {"GFP": {"method": "max"}}, [MIP_STEP], dtype="float32"
            ),
        )
        assert any("MUST keep the source dtype" in m for m in _messages(_run(tmp_path)))

    def test_renamed_channel_with_source_channel_passes(self, tmp_path):
        channels = {"nuclei": {"method": "max", "source_channel": "DAPI"}}
        make_pair(
            tmp_path,
            mip_name="plate_proj.ome.zarr",
            edit_mip=_per_channel(channels, [MIP_STEP]),
        )
        assert _messages(_run(tmp_path)) == []

    def test_unknown_source_channel_is_error(self, tmp_path):
        def rename(mip):
            to_per_channel(mip, {"GFP": {"method": "max"}}, [MIP_STEP])

            def edit(a):
                a["dca"]["channels"][0]["name"] = "RFP"
                a["dca"]["projection"]["channels"] = {"RFP": {"method": "max"}}

            edit_field_attrs(mip, edit)

        make_pair(tmp_path, mip_name="plate_proj.ome.zarr", edit_mip=rename)
        assert any("['RFP']" in m for m in _messages(_run(tmp_path)))

    def test_focus_channel_must_be_a_source_channel(self, tmp_path):
        channels = {"GFP": {"method": "mean", "z_start": [6], "z_end": [11]}}
        focus = {"channel": "Phase3D", "per": "fov", "z": 8}
        make_pair(
            tmp_path,
            mip_name="plate_proj.ome.zarr",
            edit_mip=_per_channel(channels, [MEAN_STEP], dtype="float32", focus=focus),
        )
        assert any("focus.channel 'Phase3D'" in m for m in _messages(_run(tmp_path)))

    def test_per_timepoint_planes_with_focus_provenance_pass(self, tmp_path):
        channels = {"GFP": {"method": "mean", "z_start": [6], "z_end": [11]}}
        focus = {"channel": "DAPI", "per": "timepoint", "z_per_t": [8]}
        make_pair(
            tmp_path,
            mip_name="plate_proj.ome.zarr",
            edit_mip=_per_channel(channels, [MEAN_STEP], dtype="float32", focus=focus),
        )
        assert _messages(_run(tmp_path)) == []

    def test_uniform_form_with_fewer_channels_is_error(self, tmp_path):
        def drop_channel(mip):
            to_per_channel(mip, {"GFP": {"method": "max"}}, ["deconvolved", MIP_STEP])

            def back_to_uniform(a):
                a["dca"]["projection"] = {
                    "source_store": "plate.ome.zarr",
                    "method": "max",
                }

            edit_field_attrs(mip, back_to_uniform)

        make_pair(tmp_path, edit_mip=drop_channel)
        assert any("keeps every source channel" in m for m in _messages(_run(tmp_path)))


def _drop_projection(mip):
    edit_field_attrs(mip, lambda a: a["dca"].pop("projection"))


def _drop_source_store(mip):
    edit_field_attrs(mip, lambda a: a["dca"]["projection"].pop("source_store"))


@pytest.mark.functional
class TestSourceInDataset:
    """dca.projection is required only when the 3D source is in the DCA."""

    def test_projection_of_a_sibling_without_block_is_error(self, tmp_path):
        make_pair(tmp_path, edit_mip=_drop_projection)
        run = _run(tmp_path)
        assert any(
            "is a projection of 'plate.ome.zarr'" in m and "MUST carry" in m
            for m in _messages(run)
        )
        assert run.summary.stores_failed > 0

    def test_projection_of_a_sibling_without_source_store_is_error(self, tmp_path):
        make_pair(tmp_path, edit_mip=_drop_source_store)
        run = _run(tmp_path)
        assert any("MUST name it as source_store" in m for m in _messages(run))

    def test_projected_before_deposit_without_block_warns_only(self, tmp_path):
        _, mip = make_pair(tmp_path, edit_mip=_drop_projection)
        shutil.rmtree(tmp_path / "plate.ome.zarr")  # the stack is not shared
        run = _run(tmp_path)
        assert _messages(run) == []
        assert any(
            "projected before deposit" in m for m in _messages(run, Severity.WARNING)
        )
        assert run.summary.stores_failed == 0

    def test_projected_before_deposit_with_known_parameters_passes(self, tmp_path):
        def known_parameters(mip):
            def edit(a):
                a["dca"]["projection"] = {
                    "z_dim_source": 16,
                    "channels": {"GFP": {"method": "max", "z_start": 0, "z_end": 16}},
                }
                a["dca"]["image_data"]["processing_variant"] = [MIP_STEP]

            edit_field_attrs(mip, edit)

        make_pair(tmp_path, mip_name="plate_2d.ome.zarr", edit_mip=known_parameters)
        shutil.rmtree(tmp_path / "plate.ome.zarr")
        run = _run(tmp_path)
        assert _issues(run) == []
        assert run.summary.stores_failed == 0
