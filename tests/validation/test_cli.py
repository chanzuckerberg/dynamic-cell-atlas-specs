"""
Unit/functional tests for the dca-validate CLI.

Covers the --output parquet / --output-directory report-writing mode and
well/field subsetting options (--wells, --field-offset, --field-limit,
--list-wells).
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from dca_helpers.validation.cli import main
from dca_helpers.validation.result import (
    Issue,
    PlateSubsetInfo,
    Severity,
    ValidationRun,
    ValidationSummary,
    ZarrNodeValidationResult,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
VALID_STORE = str(FIXTURES / "v0_2" / "valid_two_level_pyramid.ome.zarr")
MULTIWELL_PLATE = str(FIXTURES / "v0_2/valid_plate_multiwell.ome.zarr")


@pytest.mark.functional
class TestOutputParquet:
    def test_writes_report_to_output_directory(self, tmp_path):
        result = CliRunner().invoke(
            main,
            [VALID_STORE, "--output", "parquet", "--output-directory", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert list(tmp_path.glob("*.parquet"))

    def test_defaults_to_current_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(main, [VALID_STORE, "--output", "parquet"])
        assert result.exit_code == 0
        assert list(tmp_path.glob("*.parquet"))

    def test_confirmation_on_stderr_not_stdout(self, tmp_path):
        result = CliRunner().invoke(
            main,
            [VALID_STORE, "--output", "parquet", "--output-directory", str(tmp_path)],
        )
        assert "Wrote validation report" not in result.stdout
        assert "Wrote validation report" in result.stderr

    def test_output_directory_without_parquet_is_a_usage_error(self, tmp_path):
        result = CliRunner().invoke(
            main, [VALID_STORE, "--output", "json", "--output-directory", str(tmp_path)]
        )
        assert result.exit_code != 0
        assert "--output-directory requires --output parquet" in result.output

    def test_list_wells_with_output_parquet_is_a_usage_error(self):
        result = CliRunner().invoke(
            main, [MULTIWELL_PLATE, "--list-wells", "--output", "parquet"]
        )
        assert result.exit_code != 0
        assert "cannot be combined with --output parquet" in result.output


def _make_partial_run(*, passed: bool) -> ValidationRun:
    """Build a ValidationRun as if a subset run had completed, for output/
    exit-code assertions without exercising real I/O."""
    subset = PlateSubsetInfo(
        wells_total=6, wells_selected=1, fields_total=124000, fields_selected=1
    )
    issues = (
        []
        if passed
        else [Issue(loc=("plate",), message="boom", severity=Severity.ERROR)]
    )
    plate_result = ZarrNodeValidationResult(
        node_path="s3://bucket/plate.ome.zarr",
        spec_version="0.2",
        passed=passed,
        issues=issues,
        subset_coverage=subset,
    )
    field_result = ZarrNodeValidationResult(
        node_path="s3://bucket/plate.ome.zarr/A/1/0",
        parent_path="s3://bucket/plate.ome.zarr",
        spec_version="0.2",
        passed=True,
    )
    results = [plate_result, field_result]
    summary = ValidationSummary.from_results(results, duration_seconds=0.1)
    return ValidationRun(
        path="s3://bucket/plate.ome.zarr",
        results=results,
        summary=summary,
        spec_version="0.2",
    )


@pytest.mark.functional
class TestListWells:
    def test_list_wells_output_against_real_fixture(self):
        runner = CliRunner()
        result = runner.invoke(
            main, [MULTIWELL_PLATE, "--spec-version", "0.2", "--list-wells"]
        )
        assert result.exit_code == 0
        assert "A/1  2 fields" in result.output
        assert "B/1  2 fields" in result.output
        assert "2 wells, 4 fields total" in result.output

    def test_list_wells_against_non_plate_path_errors_cleanly(self):
        runner = CliRunner()
        non_plate = str(FIXTURES / "v0_2/valid_two_level_pyramid.ome.zarr")
        result = runner.invoke(
            main, [non_plate, "--spec-version", "0.2", "--list-wells"]
        )
        assert result.exit_code != 0
        assert "not an HCS plate" in result.output

    def test_list_wells_with_output_json(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                MULTIWELL_PLATE,
                "--spec-version",
                "0.2",
                "--list-wells",
                "--output",
                "json",
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["wells_total"] == 2
        assert payload["fields_total"] == 4
        assert {"well": "A/1", "fields": 2} in payload["wells"]
        assert {"well": "B/1", "fields": 2} in payload["wells"]

    def test_list_wells_combined_with_wells_is_a_usage_error(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                MULTIWELL_PLATE,
                "--spec-version",
                "0.2",
                "--list-wells",
                "--wells",
                "A/1",
            ],
        )
        assert result.exit_code != 0
        assert "cannot be combined" in result.output

    def test_list_wells_combined_with_field_limit_is_a_usage_error(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                MULTIWELL_PLATE,
                "--spec-version",
                "0.2",
                "--list-wells",
                "--field-limit",
                "1",
            ],
        )
        assert result.exit_code != 0
        assert "cannot be combined" in result.output


@pytest.mark.unit
class TestListPlateWellsUnit:
    def test_dropped_well_is_logged(self, monkeypatch, caplog):
        from unittest.mock import MagicMock

        from dca_helpers.validation.runner import list_plate_wells

        plate_mock = MagicMock()
        plate_mock.attrs = {
            "ome": {
                "plate": {"wells": [{"path": "A/1"}, {"path": "B/2"}]},
                "version": "0.5",
            }
        }
        well_mock = MagicMock()
        well_mock.attrs = {"ome": {"well": {"images": [{"path": "0"}]}}}

        def _open(path, *a, **kw):
            if path == "/fake/plate/B/2":
                raise Exception("simulated S3 error")
            return well_mock if path == "/fake/plate/A/1" else plate_mock

        monkeypatch.setattr(
            "dca_helpers.validation.runner.is_zarr_store", lambda p: True
        )
        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _open)
        with caplog.at_level("WARNING"):
            wells = list_plate_wells("/fake/plate")
        assert wells == [("A/1", 1)]
        assert "B/2" in caplog.text

    def test_s3_error_opening_well_becomes_value_error(self, monkeypatch):
        from unittest.mock import MagicMock

        import obstore.exceptions as obs_exc

        from dca_helpers.validation.runner import list_plate_wells

        plate_mock = MagicMock()
        plate_mock.attrs = {
            "ome": {
                "plate": {"wells": [{"path": "A/1"}, {"path": "B/2"}]},
                "version": "0.5",
            }
        }
        well_mock = MagicMock()
        well_mock.attrs = {"ome": {"well": {"images": [{"path": "0"}]}}}

        def _open(path, *a, **kw):
            if path == "/fake/plate/B/2":
                raise obs_exc.GenericError("slow down")
            return well_mock if path == "/fake/plate/A/1" else plate_mock

        monkeypatch.setattr(
            "dca_helpers.validation.runner.is_zarr_store", lambda p: True
        )
        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _open)
        with pytest.raises(ValueError, match=r"S3 access error.*B/2"):
            list_plate_wells("/fake/plate")

    def test_plate_root_open_failure_becomes_value_error(self, monkeypatch):
        # An open failure on the plate root itself (e.g. S3/network/auth
        # error) must surface as ValueError — the CLI's --list-wells branch
        # only converts ValueError into a clean UsageError, so any other
        # exception type would leak a raw stack trace instead.
        from dca_helpers.validation.runner import list_plate_wells

        def _open(path, *a, **kw):
            raise Exception("simulated S3 error")

        monkeypatch.setattr(
            "dca_helpers.validation.runner.is_zarr_store", lambda p: True
        )
        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _open)
        with pytest.raises(ValueError, match="simulated S3 error"):
            list_plate_wells("/fake/plate")

    def test_blank_well_path_is_skipped(self, monkeypatch):
        from unittest.mock import MagicMock

        from dca_helpers.validation.runner import list_plate_wells

        plate_mock = MagicMock()
        plate_mock.attrs = {
            "ome": {"plate": {"wells": [{"path": "A/1"}, {}]}, "version": "0.5"}
        }
        well_mock = MagicMock()
        well_mock.attrs = {"ome": {"well": {"images": [{"path": "0"}]}}}

        def _open(path, *a, **kw):
            return well_mock if path == "/fake/plate/A/1" else plate_mock

        monkeypatch.setattr(
            "dca_helpers.validation.runner.is_zarr_store", lambda p: True
        )
        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _open)
        assert list_plate_wells("/fake/plate") == [("A/1", 1)]

    def test_null_images_reports_zero_fields_not_a_crash(self, monkeypatch):
        from unittest.mock import MagicMock

        from dca_helpers.validation.runner import list_plate_wells

        plate_mock = MagicMock()
        plate_mock.attrs = {
            "ome": {"plate": {"wells": [{"path": "A/1"}]}, "version": "0.5"}
        }
        well_mock = MagicMock()
        well_mock.attrs = {"ome": {"well": {"images": None}}}

        def _open(path, *a, **kw):
            return well_mock if path == "/fake/plate/A/1" else plate_mock

        monkeypatch.setattr(
            "dca_helpers.validation.runner.is_zarr_store", lambda p: True
        )
        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _open)
        assert list_plate_wells("/fake/plate") == [("A/1", 0)]

    def test_null_well_ome_reports_zero_fields_not_a_crash(self, monkeypatch):
        from unittest.mock import MagicMock

        from dca_helpers.validation.runner import list_plate_wells

        plate_mock = MagicMock()
        plate_mock.attrs = {
            "ome": {"plate": {"wells": [{"path": "A/1"}]}, "version": "0.5"}
        }
        well_mock = MagicMock()
        well_mock.attrs = {"ome": None}

        def _open(path, *a, **kw):
            return well_mock if path == "/fake/plate/A/1" else plate_mock

        monkeypatch.setattr(
            "dca_helpers.validation.runner.is_zarr_store", lambda p: True
        )
        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _open)
        assert list_plate_wells("/fake/plate") == [("A/1", 0)]

    def test_non_dict_well_entry_is_skipped(self, monkeypatch):
        from unittest.mock import MagicMock

        from dca_helpers.validation.runner import list_plate_wells

        plate_mock = MagicMock()
        plate_mock.attrs = {
            "ome": {
                "plate": {"wells": ["not-a-dict", {"path": "A/1"}]},
                "version": "0.5",
            }
        }
        well_mock = MagicMock()
        well_mock.attrs = {"ome": {"well": {"images": [{"path": "0"}]}}}

        def _open(path, *a, **kw):
            return well_mock if path == "/fake/plate/A/1" else plate_mock

        monkeypatch.setattr(
            "dca_helpers.validation.runner.is_zarr_store", lambda p: True
        )
        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _open)
        assert list_plate_wells("/fake/plate") == [("A/1", 1)]

    def test_non_string_well_path_is_skipped_not_a_crash(self, monkeypatch):
        # A well entry with a non-string "path" (e.g. an int) must not crash
        # the final sorted(well_paths) call — mixed str/int values raise
        # TypeError on comparison.
        from unittest.mock import MagicMock

        from dca_helpers.validation.runner import list_plate_wells

        plate_mock = MagicMock()
        plate_mock.attrs = {
            "ome": {
                "plate": {"wells": [{"path": 123}, {"path": "A/1"}]},
                "version": "0.5",
            }
        }
        well_mock = MagicMock()
        well_mock.attrs = {"ome": {"well": {"images": [{"path": "0"}]}}}

        def _open(path, *a, **kw):
            return well_mock if path == "/fake/plate/A/1" else plate_mock

        monkeypatch.setattr(
            "dca_helpers.validation.runner.is_zarr_store", lambda p: True
        )
        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _open)
        assert list_plate_wells("/fake/plate") == [("A/1", 1)]

    def test_null_wells_reports_empty_list_not_a_crash(self, monkeypatch):
        # ome.plate.wells: null (rather than a list) must not crash the
        # well_paths comprehension's iteration — it's simply no wells.
        from unittest.mock import MagicMock

        from dca_helpers.validation.runner import list_plate_wells

        plate_mock = MagicMock()
        plate_mock.attrs = {"ome": {"plate": {"wells": None}, "version": "0.5"}}

        monkeypatch.setattr(
            "dca_helpers.validation.runner.is_zarr_store", lambda p: True
        )
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: plate_mock,
        )
        assert list_plate_wells("/fake/plate") == []

    @pytest.mark.parametrize("plate_value", [None, "A/1", []])
    def test_non_dict_plate_value_raises_not_an_hcs_plate(
        self, monkeypatch, plate_value
    ):
        # ome.plate present but not a dict (e.g. null, a string, or a list)
        # must raise "not an HCS plate", matching validate_zarr_node's
        # is_hcs_plate check — not silently report "0 wells, 0 fields" as
        # though the store were a genuinely empty, valid plate.
        from unittest.mock import MagicMock

        from dca_helpers.validation.runner import list_plate_wells

        plate_mock = MagicMock()
        plate_mock.attrs = {"ome": {"plate": plate_value, "version": "0.5"}}

        monkeypatch.setattr(
            "dca_helpers.validation.runner.is_zarr_store", lambda p: True
        )
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: plate_mock,
        )
        with pytest.raises(ValueError, match="is not an HCS plate"):
            list_plate_wells("/fake/plate")


@pytest.mark.unit
class TestSubsetOptionForwarding:
    def test_wells_and_field_offset_limit_forwarded_to_validate(self, monkeypatch):
        captured = {}

        def fake_validate(path, **kwargs):
            captured.update(kwargs)
            return _make_partial_run(passed=True)

        monkeypatch.setattr("dca_helpers.validation.cli.validate", fake_validate)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "s3://bucket/plate.ome.zarr",
                "--wells",
                "A/1,B/2",
                "--field-offset",
                "10",
                "--field-limit",
                "5",
            ],
        )
        assert result.exit_code == 0
        assert captured["wells"] == "A/1,B/2"
        assert captured["field_offset"] == 10
        assert captured["field_limit"] == 5

    def test_negative_field_offset_is_a_usage_error(self):
        runner = CliRunner()
        result = runner.invoke(
            main, ["s3://bucket/plate.ome.zarr", "--field-offset", "-1"]
        )
        assert result.exit_code != 0

    def test_value_error_from_validate_becomes_usage_error(self, monkeypatch):
        def fake_validate(path, **kwargs):
            raise ValueError("wells must name at least one well path")

        monkeypatch.setattr("dca_helpers.validation.cli.validate", fake_validate)
        runner = CliRunner()
        result = runner.invoke(main, ["s3://bucket/plate.ome.zarr", "--wells", "  "])
        assert result.exit_code != 0
        assert "wells must name at least one well path" in result.output


@pytest.mark.unit
class TestPartialMarkerAndExitCodes:
    def test_partial_marker_in_text_output(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.cli.validate",
            lambda path, **kw: _make_partial_run(passed=True),
        )
        runner = CliRunner()
        result = runner.invoke(main, ["s3://bucket/plate.ome.zarr", "--output", "text"])
        assert "PARTIAL: 1/6 wells, 1/124000 fields selected" in result.output

    def test_partial_marker_in_json_output(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.cli.validate",
            lambda path, **kw: _make_partial_run(passed=True),
        )
        runner = CliRunner()
        result = runner.invoke(main, ["s3://bucket/plate.ome.zarr", "--output", "json"])
        assert '"wells_total": 6' in result.output
        assert '"fields_selected": 1' in result.output
        assert '"partial": true' in result.output

    def test_partial_run_with_error_exits_1(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.cli.validate",
            lambda path, **kw: _make_partial_run(passed=False),
        )
        runner = CliRunner()
        result = runner.invoke(main, ["s3://bucket/plate.ome.zarr"])
        assert result.exit_code == 1

    def test_partial_run_with_no_errors_exits_0(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.cli.validate",
            lambda path, **kw: _make_partial_run(passed=True),
        )
        runner = CliRunner()
        result = runner.invoke(main, ["s3://bucket/plate.ome.zarr"])
        assert result.exit_code == 0
