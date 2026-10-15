"""
Unit tests for validation.result — ValidationSummary and ValidationRun.
"""

import pytest
from click.testing import CliRunner

from dca_helpers import CURRENT_SPEC_VERSION, __version__
from dca_helpers.validation.cli import _print_text
from dca_helpers.validation.result import (
    PlateSubsetInfo,
    ValidationRun,
    ValidationSummary,
    ZarrNodeValidationResult,
)
from dca_helpers.validation.zarr_node import ZarrNodeType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(
    passed: bool,
    node_type: ZarrNodeType | None = None,
    multiscale_level_count: int | None = None,
    skipped: bool = False,
    subset_coverage: PlateSubsetInfo | None = None,
) -> ZarrNodeValidationResult:
    return ZarrNodeValidationResult(
        node_path="/fake/store",
        spec_version="0.1",
        passed=passed,
        node_type=node_type,
        multiscale_level_count=multiscale_level_count,
        skipped=skipped,
        subset_coverage=subset_coverage,
    )


# ---------------------------------------------------------------------------
# ValidationSummary.from_results
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidationSummaryFromResults:
    def test_empty_results(self):
        summary = ValidationSummary.from_results([])
        assert summary.stores_validated == 0
        assert summary.stores_passed == 0
        assert summary.stores_failed == 0
        assert summary.level_zarr_jsons_validated == 0
        assert summary.nodes_by_type == {}
        assert summary.stores_discovered == 0
        assert summary.duration_seconds == 0.0

    def test_pass_fail_counts(self):
        results = [make_result(True), make_result(False), make_result(True)]
        summary = ValidationSummary.from_results(results)
        assert summary.stores_validated == 3
        assert summary.stores_passed == 2
        assert summary.stores_failed == 1

    def test_nodes_by_type_grouped(self):
        results = [
            make_result(True, node_type=ZarrNodeType.IMAGE),
            make_result(True, node_type=ZarrNodeType.IMAGE),
            make_result(False, node_type=ZarrNodeType.HCS_PLATE),
        ]
        summary = ValidationSummary.from_results(results)
        assert summary.nodes_by_type == {
            ZarrNodeType.IMAGE: 2,
            ZarrNodeType.HCS_PLATE: 1,
        }

    def test_nodes_by_type_excludes_none(self):
        results = [
            make_result(True, node_type=None),
            make_result(True, node_type=ZarrNodeType.IMAGE),
        ]
        summary = ValidationSummary.from_results(results)
        assert summary.nodes_by_type == {ZarrNodeType.IMAGE: 1}

    def test_level_zarr_jsons_summed(self):
        results = [
            make_result(True, multiscale_level_count=3),
            make_result(True, multiscale_level_count=2),
            make_result(True, multiscale_level_count=None),
        ]
        summary = ValidationSummary.from_results(results)
        assert summary.level_zarr_jsons_validated == 5

    def test_stores_discovered_and_duration_forwarded(self):
        results = [make_result(True)]
        summary = ValidationSummary.from_results(
            results,
            duration_seconds=4.2,
            stores_discovered=10,
            start_timestamp="2026-03-01T10:00:00Z",
        )
        assert summary.stores_discovered == 10
        assert summary.duration_seconds == pytest.approx(4.2)
        assert summary.start_timestamp == "2026-03-01T10:00:00Z"

    def test_start_timestamp_defaults_to_empty_string(self):
        summary = ValidationSummary.from_results([make_result(True)])
        assert summary.start_timestamp == ""

    def test_skipped_zero_by_default(self):
        summary = ValidationSummary.from_results(
            [make_result(True), make_result(False)]
        )
        assert summary.stores_skipped == 0

    def test_skipped_excluded_from_stores_validated(self):
        results = [
            make_result(True),
            make_result(True, node_type=ZarrNodeType.GEFF, skipped=True),
        ]
        summary = ValidationSummary.from_results(results)
        assert summary.stores_validated == 1
        assert summary.stores_passed == 1
        assert summary.stores_failed == 0
        assert summary.stores_skipped == 1

    def test_skipped_invariant_holds(self):
        # stores_validated == stores_passed + stores_failed always
        results = [
            make_result(True),
            make_result(False),
            make_result(True, node_type=ZarrNodeType.GEFF, skipped=True),
        ]
        summary = ValidationSummary.from_results(results)
        assert summary.stores_validated == summary.stores_passed + summary.stores_failed

    def test_skipped_appear_in_nodes_by_type(self):
        results = [
            make_result(True, node_type=ZarrNodeType.IMAGE),
            make_result(True, node_type=ZarrNodeType.GEFF, skipped=True),
        ]
        summary = ValidationSummary.from_results(results)
        assert ZarrNodeType.GEFF in summary.nodes_by_type
        assert summary.nodes_by_type[ZarrNodeType.GEFF] == 1

    def test_skipped_level_zarr_jsons_excluded(self):
        results = [
            make_result(True, multiscale_level_count=3),
            make_result(True, multiscale_level_count=5, skipped=True),
        ]
        summary = ValidationSummary.from_results(results)
        assert summary.level_zarr_jsons_validated == 3


# ---------------------------------------------------------------------------
# ValidationSummary.from_results — well/field subset aggregation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidationSummarySubset:
    def test_is_partial_false_without_any_subset(self):
        summary = ValidationSummary.from_results([make_result(True)])
        assert summary.is_partial is False
        assert summary.wells_total == 0
        assert summary.wells_selected == 0
        assert summary.fields_total == 0
        assert summary.fields_selected == 0

    def test_is_partial_true_and_counts_forwarded(self):
        subset = PlateSubsetInfo(
            wells_total=6,
            wells_selected=1,
            fields_total=124000,
            fields_selected=5000,
        )
        results = [make_result(True, subset_coverage=subset), make_result(True)]
        summary = ValidationSummary.from_results(results)
        assert summary.is_partial is True
        assert summary.wells_total == 6
        assert summary.wells_selected == 1
        assert summary.fields_total == 124000
        assert summary.fields_selected == 5000

    def test_subset_counts_sum_across_multiple_plate_roots(self):
        subset_a = PlateSubsetInfo(
            wells_total=2, wells_selected=1, fields_total=10, fields_selected=5
        )
        subset_b = PlateSubsetInfo(
            wells_total=3, wells_selected=2, fields_total=20, fields_selected=8
        )
        results = [
            make_result(True, subset_coverage=subset_a),
            make_result(True, subset_coverage=subset_b),
        ]
        summary = ValidationSummary.from_results(results)
        assert summary.is_partial is True
        assert summary.wells_total == 5
        assert summary.wells_selected == 3
        assert summary.fields_total == 30
        assert summary.fields_selected == 13


# ---------------------------------------------------------------------------
# ValidationRun
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidationRun:
    def _make_run(self, results):
        summary = ValidationSummary.from_results(
            results, duration_seconds=1.0, stores_discovered=len(results)
        )
        return ValidationRun(
            path="/fake/root", results=results, summary=summary, spec_version="0.1"
        )

    def test_len(self):
        run = self._make_run([make_result(True), make_result(False)])
        assert len(run) == 2

    def test_iter(self):
        results = [make_result(True), make_result(False)]
        run = self._make_run(results)
        assert list(run) == results

    def test_summary_attached(self):
        run = self._make_run([make_result(True)])
        assert run.summary.stores_passed == 1

    def test_path_preserved(self):
        run = self._make_run([])
        assert run.path == "/fake/root"

    def test_spec_version_and_validator_version_have_defaults(self):
        # Regression test: spec_version used to be a required field with no
        # default, so this constructor call raised TypeError for any
        # external caller that didn't pass it.
        summary = ValidationSummary.from_results([])
        run = ValidationRun(path="/fake/root", results=[], summary=summary)
        assert run.spec_version == CURRENT_SPEC_VERSION
        assert run.validator_version == __version__


# ---------------------------------------------------------------------------
# _print_text — skipped-store output
# ---------------------------------------------------------------------------


def _make_run(results):
    summary = ValidationSummary.from_results(
        results, duration_seconds=0.1, stores_discovered=len(results)
    )
    return ValidationRun(
        path="/fake/root", results=results, summary=summary, spec_version="0.2"
    )


@pytest.mark.unit
class TestPrintTextSkipped:
    def _capture(self, run, show_warnings=False):
        """Run _print_text under Click's test runner and return captured output."""
        runner = CliRunner()
        import click

        @click.command()
        def _cmd():
            _print_text(run, show_warnings=show_warnings)

        return runner.invoke(_cmd).output

    def test_skipped_store_not_printed_in_issue_block(self):
        geff = make_result(True, node_type=ZarrNodeType.GEFF, skipped=True)
        geff.node_path = "/fake/store.tracks.geff"
        run = _make_run([geff])
        output = self._capture(run)
        assert "store.tracks.geff" not in output

    def test_skipped_line_appears_in_summary_when_nonzero(self):
        geff = make_result(True, node_type=ZarrNodeType.GEFF, skipped=True)
        run = _make_run([geff])
        output = self._capture(run)
        assert "1 skipped (GEFF)" in output

    def test_partial_marker_absent_on_full_run(self):
        run = _make_run([make_result(True)])
        output = self._capture(run)
        assert "PARTIAL" not in output

    def test_partial_marker_present_with_subset_counts(self):
        subset = PlateSubsetInfo(
            wells_total=6, wells_selected=1, fields_total=124000, fields_selected=5000
        )
        run = _make_run([make_result(True, subset_coverage=subset)])
        output = self._capture(run)
        assert "PARTIAL: 1/6 wells, 5000/124000 fields selected" in output

    def test_skipped_line_absent_when_zero(self):
        run = _make_run([make_result(True)])
        output = self._capture(run)
        assert "skipped" not in output

    def test_validated_count_excludes_skipped(self):
        geff = make_result(True, node_type=ZarrNodeType.GEFF, skipped=True)
        run = _make_run([make_result(True), geff])
        output = self._capture(run)
        assert "Validated 1 stores" in output
