"""
Functional tests for the cross-store projection checks (validation/pairing.py)
and --require-mip, against v0.3 source / projection pairs written to tmp_path
by projection_fixtures.make_pair.
"""

import shutil

import pytest
from click.testing import CliRunner

from dca_helpers.validation.cli import main
from dca_helpers.validation.result import Severity
from dca_helpers.validation.runner import validate

from .projection_fixtures import (
    FIELD,
    edit_field_attrs,
    make_pair,
    read_json,
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

    def test_pair_passes_with_require_mip(self, tmp_path):
        make_pair(tmp_path)
        run = _run(tmp_path, require_mip=True)
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
        run = _run(tmp_path, require_mip=True)
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
        run = _run(tmp_path, require_mip=True)
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

    def test_store_name_without_mip_suffix_warns(self, tmp_path):
        make_pair(tmp_path, mip_name="plate_projection.ome.zarr")
        run = _run(tmp_path)
        assert any("_mip.ome.zarr" in m for m in _messages(run, Severity.WARNING))


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

    def test_projection_block_without_mip_step_is_error(self, tmp_path):
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
        assert any("does not list" in m for m in _messages(run))


@pytest.mark.functional
class TestRequireMip:
    def test_3d_store_without_projection_is_error(self, tmp_path):
        make_pair(tmp_path, with_mip=False)
        run = _run(tmp_path, require_mip=True)
        assert run.summary.stores_failed == 1
        assert any("--require-mip" in m for m in _messages(run))

    def test_without_flag_a_lone_3d_store_passes(self, tmp_path):
        make_pair(tmp_path, with_mip=False)
        run = _run(tmp_path)
        assert run.summary.stores_failed == 0

    def test_every_3d_variant_needs_its_own_projection(self, tmp_path):
        make_pair(tmp_path)
        make_pair(tmp_path, source_name="plate_trimmed.ome.zarr", with_mip=False)
        run = _run(tmp_path, require_mip=True)
        failed = [r.node_path for r in run if not r.passed]
        assert [p.rsplit("/", 1)[-1] for p in failed] == ["plate_trimmed.ome.zarr"]

    def test_requires_a_projection_spec_version(self, tmp_path):
        make_pair(tmp_path, with_mip=False)
        with pytest.raises(ValueError, match="require_mip"):
            validate(
                str(tmp_path), spec_version="0.2", show_progress=False, require_mip=True
            )

    def test_cli_flag(self, tmp_path):
        make_pair(tmp_path, with_mip=False)
        result = CliRunner().invoke(
            main, [str(tmp_path), "--spec-version", "0.3", "--require-mip"]
        )
        assert result.exit_code == 1
        assert "--require-mip" in result.output

    def test_cli_flag_with_v0_2_is_usage_error(self, tmp_path):
        make_pair(tmp_path, with_mip=False)
        result = CliRunner().invoke(
            main, [str(tmp_path), "--spec-version", "0.2", "--require-mip"]
        )
        assert result.exit_code == 2
