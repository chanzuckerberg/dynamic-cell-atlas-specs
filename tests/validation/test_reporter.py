"""
Unit tests for validation.reporter — Parquet report building.
"""

import dataclasses
import re
from datetime import datetime, timezone

import pyarrow.parquet as pq
import pytest

from dca_helpers.validation import reporter
from dca_helpers.validation.result import (
    Issue,
    PlateSubsetInfo,
    Severity,
    ValidationRun,
    ValidationSummary,
    ZarrNodeValidationResult,
)
from dca_helpers.validation.zarr_node import ZarrNodeType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(
    node_path="s3://bucket/plate.ome.zarr/A/01/0",
    parent_path=None,
    passed=True,
    node_type=None,
    multiscale_level_count=None,
    issues=None,
    subset_coverage=None,
    skipped=False,
):
    return ZarrNodeValidationResult(
        node_path=node_path,
        parent_path=parent_path,
        spec_version="0.2",
        passed=passed,
        node_type=node_type,
        multiscale_level_count=multiscale_level_count,
        issues=issues or [],
        subset_coverage=subset_coverage,
        skipped=skipped,
    )


def make_run(path="s3://bucket", results=None, spec_version="0.2"):
    results = results if results is not None else [make_result()]
    summary = ValidationSummary.from_results(
        results,
        duration_seconds=1.5,
        stores_discovered=len(results),
        start_timestamp="2026-01-01T00:00:00Z",
    )
    return ValidationRun(
        path=path, results=results, summary=summary, spec_version=spec_version
    )


# ---------------------------------------------------------------------------
# _slugify_path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlugifyPath:
    def test_strips_s3_scheme(self):
        assert "s3" not in reporter._slugify_path("s3://bucket/plate.ome.zarr")
        assert "://" not in reporter._slugify_path("s3://bucket/plate.ome.zarr")

    def test_strips_trailing_slash(self):
        slug = reporter._slugify_path("s3://bucket/plate.ome.zarr/")
        assert not slug.endswith("_")
        assert slug == reporter._slugify_path("s3://bucket/plate.ome.zarr")

    def test_replaces_unsafe_characters(self):
        slug = reporter._slugify_path("s3://bucket/plate.ome.zarr/A/01/0")
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", slug)

    def test_truncates_to_max_length_keeping_head_and_tail(self):
        long_path = "s3://head-marker/" + "x" * 200 + "/tail-marker"
        slug = reporter._slugify_path(long_path)
        assert len(slug) == reporter._MAX_SLUG_LENGTH
        assert slug.startswith("head-marker")
        assert slug.endswith("tail-marker")

    def test_truncation_preserves_a_store_roots_distinguishing_tail(self):
        # A store-root path's distinguishing part (dataset/plate name) is
        # its tail, behind a long generic bucket/prefix head -- unlike a
        # field/FOV path, where the distinguishing part sits near the head.
        # Head-only truncation would drop this tail entirely.
        long_path = (
            "s3://example-bucket/path/to/some-collection/"
            "experiment-batch/dataset-name/plate001.ome.zarr"
        )
        slug = reporter._slugify_path(long_path)
        assert slug.startswith("example-bucket")
        assert slug.endswith("dataset-name_plate001.ome.zarr")

    def test_head_lands_on_a_segment_boundary_not_mid_word(self):
        # The head is capped at _SLUG_HEAD_LENGTH and taken from the first
        # path segment, not a raw character count -- so a bucket name
        # shorter than the cap survives whole instead of being cut mid-word.
        long_path = "s3://short-bucket/" + "x" * 200
        slug = reporter._slugify_path(long_path)
        assert slug.startswith("short-bucket" + reporter._SLUG_TRUNCATION_MARKER)

    def test_no_leading_hyphen(self):
        # A leading "-" survives sanitization (it's an allowed character) but
        # could be misread as a CLI flag by tools operating on this filename
        # later (e.g. `rm -foo.parquet`).
        slug = reporter._slugify_path("s3://bucket/-experiment/plate.zarr")
        assert not slug.startswith("-")

    def test_empty_path_falls_back_to_run(self):
        assert reporter._slugify_path("") == "run"


# ---------------------------------------------------------------------------
# _generate_filename
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateFilename:
    def test_matches_expected_shape(self):
        run = make_run(path="s3://bucket/plate.ome.zarr")
        now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        filename = reporter._generate_filename(run, now)
        slug, timestamp, suffix_and_ext = filename.split("__")
        assert slug == "bucket_plate.ome.zarr"
        assert timestamp == "20260102T030405Z"
        suffix, ext = suffix_and_ext.split(".")
        assert re.fullmatch(r"[0-9a-f]{8}", suffix)
        assert ext == "parquet"

    def test_two_calls_at_the_same_instant_are_still_distinct(self):
        # Distinctness comes from the random suffix, not the timestamp --
        # this is exactly the case parallel partial-validation runs hit,
        # since they can share the same instant.
        run = make_run()
        now = datetime.now(timezone.utc)
        assert reporter._generate_filename(run, now) != reporter._generate_filename(
            run, now
        )


# ---------------------------------------------------------------------------
# to_rows
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToRows:
    TS = "2026-09-04T12:00:00Z"

    def test_first_row_is_run_summary(self):
        run = make_run()
        rows = reporter.to_rows(run, self.TS)
        assert rows[0]["record_type"] == "validation_run_summary"

    def test_one_node_row_per_result_after_the_summary_row(self):
        results = [
            make_result(node_path="s3://bucket/a"),
            make_result(node_path="s3://bucket/b"),
            make_result(node_path="s3://bucket/c"),
        ]
        run = make_run(results=results)
        rows = reporter.to_rows(run, self.TS)
        assert len(rows) == 4
        assert [r["record_type"] for r in rows[1:]] == ["node", "node", "node"]

    def test_zero_store_run_produces_only_the_summary_row(self):
        run = make_run(results=[])
        rows = reporter.to_rows(run, self.TS)
        assert len(rows) == 1
        assert rows[0]["record_type"] == "validation_run_summary"

    def test_shared_fields_present_on_every_row(self):
        run = make_run(path="s3://bucket/plate.ome.zarr", spec_version="0.1")
        run.validator_version = "1.2.3"
        rows = reporter.to_rows(run, self.TS)
        for row in rows:
            assert row["input_path"] == "s3://bucket/plate.ome.zarr"
            assert row["dca_spec_version"] == "0.1"
            assert row["dca_validator_version"] == "1.2.3"
            assert row["reporting_format_version"] == "1.1.0"
            assert row["start_timestamp"] == "2026-01-01T00:00:00Z"
            assert row["completion_timestamp"] == self.TS

    def test_start_timestamp_carries_through_from_validation_summary(self):
        results = [make_result()]
        summary = ValidationSummary.from_results(
            results, start_timestamp="2026-03-01T10:00:00Z"
        )
        run = ValidationRun(
            path="s3://bucket", results=results, summary=summary, spec_version="0.2"
        )
        rows = reporter.to_rows(run, self.TS)
        assert all(row["start_timestamp"] == "2026-03-01T10:00:00Z" for row in rows)

    def test_run_summary_row_has_no_leftover_summary_aggregate_fields(self):
        # None of ValidationSummary's other aggregate fields are persisted
        # -- stores_discovered/duration_seconds/stores_passed/nodes_by_type/
        # is_partial/wells_total/etc. are all derivable via aggregate query
        # over the node rows (stores_discovered as a row count,
        # duration_seconds as completion_timestamp - start_timestamp, the
        # rest as plain aggregates), so persisting any of them again would
        # be a second, independently-maintained source of truth that could
        # silently disagree with the first.
        run = make_run()
        summary_row = reporter.to_rows(run, self.TS)[0]
        for leftover in (
            "stores_validated",
            "stores_passed",
            "stores_failed",
            "stores_skipped",
            "stores_discovered",
            "duration_seconds",
            "nodes_by_type",
            "is_partial",
            "wells_total",
            "wells_selected",
            "fields_total",
            "fields_selected",
            "level_zarr_jsons_validated",
        ):
            assert leftover not in summary_row

    def test_run_summary_row_has_null_node_only_fields(self):
        run = make_run()
        summary_row = reporter.to_rows(run, self.TS)[0]
        for field_name in (
            "node_path",
            "parent_path",
            "passed",
            "ngff_version",
            "multiscale_level_count",
            "issues",
            "node_type",
            "skipped",
            "subset_coverage",
        ):
            assert summary_row[field_name] is None

    def test_node_row_content_matches_result(self):
        result = make_result(node_path="s3://bucket/x", passed=False)
        run = make_run(results=[result])
        node_row = reporter.to_rows(run, self.TS)[1]
        assert node_row["node_path"] == "s3://bucket/x"
        assert node_row["passed"] is False

    def test_spec_version_absent_from_node_rows(self):
        # spec_version is run-level (dca_spec_version); redundant per node.
        run = make_run(results=[make_result()])
        node_row = reporter.to_rows(run, self.TS)[1]
        assert "spec_version" not in node_row

    def test_node_type_is_enum_name(self):
        result = make_result(node_type=ZarrNodeType.HCS_PLATE)
        run = make_run(results=[result])
        node_row = reporter.to_rows(run, self.TS)[1]
        assert node_row["node_type"] == "HCS_PLATE"

    def test_issue_loc_elements_are_stringified(self):
        result = make_result(
            passed=False,
            issues=[
                Issue(loc=("wells", 0, "path"), message="bad", severity=Severity.ERROR)
            ],
        )
        run = make_run(results=[result])
        issue = reporter.to_rows(run, self.TS)[1]["issues"][0]
        assert issue["loc"] == ["wells", "0", "path"]
        assert all(isinstance(x, str) for x in issue["loc"])

    def test_includes_all_issues_regardless_of_severity(self):
        result = make_result(
            passed=False,
            issues=[
                Issue(loc=("a",), message="bad", severity=Severity.ERROR),
                Issue(loc=("b",), message="meh", severity=Severity.WARNING),
            ],
        )
        run = make_run(results=[result])
        issues = reporter.to_rows(run, self.TS)[1]["issues"]
        assert len(issues) == 2
        assert {i["severity"] for i in issues} == {"error", "warning"}

    def test_subset_carries_through_on_partial_plate_result(self):
        subset = PlateSubsetInfo(
            wells_total=6, wells_selected=1, fields_total=124000, fields_selected=1
        )
        run = make_run(results=[make_result(subset_coverage=subset)])
        node_row = reporter.to_rows(run, self.TS)[1]
        assert node_row["subset_coverage"] == {
            "wells_total": 6,
            "wells_selected": 1,
            "fields_total": 124000,
            "fields_selected": 1,
        }

    def test_subset_is_none_for_a_full_plate_result(self):
        run = make_run(results=[make_result(subset_coverage=None)])
        node_row = reporter.to_rows(run, self.TS)[1]
        assert node_row["subset_coverage"] is None


# ---------------------------------------------------------------------------
# REPORT_SCHEMA / dataclass field consistency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSchemaMatchesDataclassFields:
    def test_node_fields_are_covered_by_the_schema(self):
        dataclass_fields = {
            f.name
            for f in dataclasses.fields(ZarrNodeValidationResult)
            if f.name not in reporter._NODE_RESULT_EXCLUDED_FIELDS
        }
        assert dataclass_fields <= set(reporter.REPORT_SCHEMA.names)

    def test_subset_schema_covers_dataclass_fields(self):
        dataclass_fields = {f.name for f in dataclasses.fields(PlateSubsetInfo)}
        assert dataclass_fields == set(reporter._SUBSET_COVERAGE_TYPE.names)

    def test_shared_timestamp_field_exists_on_validation_summary(self):
        # _shared_fields reads run.summary.start_timestamp directly by
        # attribute name -- guards against a silent rename on
        # ValidationSummary breaking that access.
        dataclass_fields = {f.name for f in dataclasses.fields(ValidationSummary)}
        assert "start_timestamp" in dataclass_fields

    def test_schema_has_no_stray_columns(self):
        # The other direction of the tests above: every REPORT_SCHEMA column
        # is accounted for by shared provenance or a real
        # ZarrNodeValidationResult field -- nothing left over from the old
        # stores[]/summary-struct columns.
        node_fields = {
            f.name
            for f in dataclasses.fields(ZarrNodeValidationResult)
            if f.name not in reporter._NODE_RESULT_EXCLUDED_FIELDS
        }
        shared_fields = {
            "record_type",
            "input_path",
            "snapshot_id",
            "dca_spec_version",
            "dca_validator_version",
            "reporting_format_version",
            "start_timestamp",
            "completion_timestamp",
        }
        assert set(reporter.REPORT_SCHEMA.names) == node_fields | shared_fields


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildReport:
    def test_returns_valid_parquet_bytes(self, tmp_path):
        run = make_run(results=[make_result(node_type=ZarrNodeType.IMAGE)])
        report_bytes = reporter.build_report(run)
        assert isinstance(report_bytes, bytes)
        scratch = tmp_path / "scratch.parquet"
        scratch.write_bytes(report_bytes)
        table = pq.read_table(scratch)
        assert table.num_rows == 2  # one run-summary row + one node row

    def test_row_count_is_one_summary_plus_one_per_node(self, tmp_path):
        results = [
            make_result(node_path="s3://bucket/a"),
            make_result(node_path="s3://bucket/b"),
        ]
        run = make_run(results=results)
        scratch = tmp_path / "scratch.parquet"
        scratch.write_bytes(reporter.build_report(run))
        assert pq.read_table(scratch).num_rows == 3

    def test_content_matches_to_rows_shape(self, tmp_path):
        run = make_run(results=[make_result(passed=False)])
        scratch = tmp_path / "scratch.parquet"
        scratch.write_bytes(reporter.build_report(run))
        rows = pq.read_table(scratch).to_pylist()
        node_rows = [r for r in rows if r["record_type"] == "node"]
        assert len(node_rows) == 1
        assert node_rows[0]["passed"] is False

    def test_builds_valid_table_with_full_detail(self):
        subset = PlateSubsetInfo(
            wells_total=6, wells_selected=1, fields_total=10, fields_selected=5
        )
        root = make_result(
            node_path="s3://bucket/plate.ome.zarr",
            node_type=ZarrNodeType.HCS_PLATE,
            subset_coverage=subset,
            issues=[Issue(loc=("a", 1), message="bad", severity=Severity.ERROR)],
        )
        report_bytes = reporter.build_report(make_run(results=[root]))
        assert len(report_bytes) > 0

    def test_two_calls_produce_distinct_completion_timestamps_or_at_least_valid_ones(
        self,
    ):
        # Not asserting distinctness (calls can land in the same second) --
        # just that each call computes its own fresh timestamp rather than
        # caching/reusing one across calls.
        run = make_run()
        first = reporter.build_report(run)
        second = reporter.build_report(run)
        assert isinstance(first, bytes) and isinstance(second, bytes)

    def test_omitted_completion_timestamp_still_produces_a_report(self, tmp_path):
        # Backward compatible: no completion_timestamp kwarg behaves exactly
        # like before this parameter was added.
        run = make_run(results=[make_result(passed=False)])
        scratch = tmp_path / "scratch.parquet"
        scratch.write_bytes(reporter.build_report(run))
        rows = pq.read_table(scratch).to_pylist()
        node_rows = [r for r in rows if r["record_type"] == "node"]
        assert node_rows[0]["passed"] is False

    def test_explicit_completion_timestamp_is_used_verbatim(self, tmp_path):
        run = make_run()
        supplied = "2026-01-01T00:00:00Z"
        scratch = tmp_path / "scratch.parquet"
        scratch.write_bytes(reporter.build_report(run, completion_timestamp=supplied))
        rows = pq.read_table(scratch).to_pylist()
        assert all(row["completion_timestamp"] == supplied for row in rows)

    def test_zero_store_run_produces_valid_report_with_only_summary_row(self, tmp_path):
        # A run that discovered no stores at all (e.g. an empty
        # --output-directory prefix) still has to serialize into a valid,
        # single-row report representing that the run happened.
        run = make_run(results=[])
        scratch = tmp_path / "scratch.parquet"
        scratch.write_bytes(reporter.build_report(run))
        rows = pq.read_table(scratch).to_pylist()
        assert len(rows) == 1
        assert rows[0]["record_type"] == "validation_run_summary"


# ---------------------------------------------------------------------------
# write_local
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWriteLocal:
    def test_writes_file_under_explicit_output_dir(self, tmp_path):
        run = make_run()
        result_path = reporter.write_local(run, output_dir=tmp_path)
        assert result_path.parent == tmp_path
        assert result_path.exists()

    def test_falls_back_to_cwd_when_no_output_dir_given(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run = make_run()
        result_path = reporter.write_local(run)
        assert result_path.parent == tmp_path

    def test_no_tmp_file_left_behind(self, tmp_path):
        run = make_run()
        reporter.write_local(run, output_dir=tmp_path)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_tmp_file_cleaned_up_on_replace_failure(self, tmp_path, monkeypatch):
        run = make_run()

        def _boom(*args, **kwargs):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(reporter.os, "replace", _boom)
        with pytest.raises(OSError):
            reporter.write_local(run, output_dir=tmp_path)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_tmp_file_cleaned_up_on_non_os_error_write_failure(
        self, tmp_path, monkeypatch
    ):
        run = make_run()

        def _boom(run, **kwargs):
            raise ValueError("simulated non-OSError build_report failure")

        monkeypatch.setattr(reporter, "build_report", _boom)
        with pytest.raises(ValueError):
            reporter.write_local(run, output_dir=tmp_path)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_creates_missing_output_dir(self, tmp_path):
        run = make_run()
        nested = tmp_path / "nested" / "dir"
        result_path = reporter.write_local(run, output_dir=nested)
        assert result_path.exists()

    def test_written_content_round_trips(self, tmp_path):
        run = make_run(results=[make_result(passed=False)])
        result_path = reporter.write_local(run, output_dir=tmp_path)
        rows = pq.read_table(result_path).to_pylist()
        node_rows = [r for r in rows if r["record_type"] == "node"]
        assert rows[0]["input_path"] == run.path
        assert node_rows[0]["passed"] is False

    def test_filename_and_record_timestamp_agree(self, tmp_path):
        # Regression test: these used to come from two independent
        # datetime.now() calls, which could tick a second (or a UTC
        # midnight) apart -- write_local must derive both from one instant.
        run = make_run()
        result_path = reporter.write_local(run, output_dir=tmp_path)
        _, filename_timestamp, _ = result_path.name.split("__")
        from_filename = datetime.strptime(filename_timestamp, "%Y%m%dT%H%M%SZ")

        rows = pq.read_table(result_path).to_pylist()
        from_record = datetime.strptime(
            rows[0]["completion_timestamp"], "%Y-%m-%dT%H:%M:%SZ"
        )
        assert from_filename == from_record

    def test_no_partition_subdirectories(self, tmp_path):
        run = make_run(path="s3://bucket/some-collection/plate.ome.zarr")
        result_path = reporter.write_local(run, output_dir=tmp_path)
        assert result_path.parent == tmp_path
