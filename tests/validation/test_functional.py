"""
Integration tests for the DCA validator.

Exercises the full validate() → runner → validator → discovery stack against
minimal local fixture stores (zarr.json files only, no chunk data).

Run with: pytest -m functional
"""

from pathlib import Path

import pytest

from dca_helpers.validation.result import Severity
from dca_helpers.validation.runner import validate

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _all_issues(run):
    return [i for r in run.results for i in r.issues]


@pytest.mark.functional
class TestValidStores:
    def test_valid_v01_plate_passes(self):
        run = validate(
            str(FIXTURES / "v0_1/valid_plate.ome.zarr"),
            spec_version="0.1",
            show_progress=False,
        )
        assert run.summary.stores_failed == 0

    def test_valid_v01_single_image_passes(self):
        run = validate(
            str(FIXTURES / "v0_1/valid_single_image.ome.zarr"),
            spec_version="0.1",
            show_progress=False,
        )
        assert run.summary.stores_failed == 0

    def test_valid_plate_with_should_violations_passes(self):
        run = validate(
            str(FIXTURES / "v0_1/valid_plate_warnings_only.ome.zarr"),
            spec_version="0.1",
            show_progress=False,
        )
        assert run.summary.stores_failed == 0
        assert any(i.severity == Severity.WARNING for i in _all_issues(run))


@pytest.mark.functional
class TestInvalidPlates:
    def test_missing_well_is_error(self):
        run = validate(
            str(FIXTURES / "v0_1/invalid_missing_well.ome.zarr"),
            spec_version="0.1",
            show_progress=False,
        )
        assert run.summary.stores_failed > 0
        assert any("no group on disk" in i.message for i in _all_issues(run))

    def test_missing_field_is_error(self):
        run = validate(
            str(FIXTURES / "v0_1/invalid_missing_field.ome.zarr"),
            spec_version="0.1",
            show_progress=False,
        )
        assert run.summary.stores_failed > 0
        assert any("no group on disk" in i.message for i in _all_issues(run))

    def test_orphan_field_is_error(self):
        run = validate(
            str(FIXTURES / "v0_1/invalid_orphan_field.ome.zarr"),
            spec_version="0.1",
            show_progress=False,
        )
        assert run.summary.stores_failed > 0
        assert any(
            "not declared in ome.well.images" in i.message for i in _all_issues(run)
        )

    def test_bad_acquisition_is_error(self):
        run = validate(
            str(FIXTURES / "v0_1/invalid_bad_acquisition.ome.zarr"),
            spec_version="0.1",
            show_progress=False,
        )
        assert run.summary.stores_failed > 0
        assert any(
            "Acquisition ID" in i.message and "not in" in i.message
            for i in _all_issues(run)
        )

    def test_bad_row_name_is_error(self):
        run = validate(
            str(FIXTURES / "v0_1/invalid_bad_row_name.ome.zarr"),
            spec_version="0.1",
            show_progress=False,
        )
        assert run.summary.stores_failed > 0
        assert any("alphanumeric" in i.message for i in _all_issues(run))


@pytest.mark.functional
class TestInvalidImages:
    def test_bad_codec_is_error(self):
        run = validate(
            str(FIXTURES / "v0_1/invalid_single_image_bad_codec.ome.zarr"),
            spec_version="0.1",
            show_progress=False,
        )
        assert run.summary.stores_failed > 0
        assert any("not permitted" in i.message for i in _all_issues(run))
