"""
Unit tests for validation.result — ValidationSummary and ValidationRun.
"""

import pytest
from click.testing import CliRunner

from dca_helpers.validation.cli import _print_text
from dca_helpers.validation.result import (
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
) -> ZarrNodeValidationResult:
    return ZarrNodeValidationResult(
        node_path="/fake/store",
        spec_version="0.1",
        passed=passed,
        node_type=node_type,
        multiscale_level_count=multiscale_level_count,
        skipped=skipped,
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
            results, duration_seconds=4.2, stores_discovered=10
        )
        assert summary.stores_discovered == 10
        assert summary.duration_seconds == pytest.approx(4.2)

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
# ValidationRun
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidationRun:
    def _make_run(self, results):
        summary = ValidationSummary.from_results(
            results, duration_seconds=1.0, stores_discovered=len(results)
        )
        return ValidationRun(path="/fake/root", results=results, summary=summary)

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


# ---------------------------------------------------------------------------
# _print_text — skipped-store output
# ---------------------------------------------------------------------------


def _make_run(results):
    summary = ValidationSummary.from_results(
        results, duration_seconds=0.1, stores_discovered=len(results)
    )
    return ValidationRun(path="/fake/root", results=results, summary=summary)


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

    def test_skipped_line_absent_when_zero(self):
        run = _make_run([make_result(True)])
        output = self._capture(run)
        assert "skipped" not in output

    def test_validated_count_excludes_skipped(self):
        geff = make_result(True, node_type=ZarrNodeType.GEFF, skipped=True)
        run = _make_run([make_result(True), geff])
        output = self._capture(run)
        assert "Validated 1 stores" in output
