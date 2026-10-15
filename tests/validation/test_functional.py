"""
Integration tests for the DCA validator.

Exercises the full validate() → runner → validator → discovery stack against
minimal local fixture stores (zarr.json files only, no chunk data).

Run with: pytest -m functional
"""

from pathlib import Path

import pytest

from dca_helpers.validation.result import Severity
from dca_helpers.validation.runner import list_plate_wells, validate

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


@pytest.mark.functional
class TestValidStoresV02:
    def test_valid_v02_plate_passes(self):
        run = validate(
            str(FIXTURES / "v0_2/valid_plate.ome.zarr"),
            spec_version="0.2",
            show_progress=False,
        )
        assert run.summary.stores_failed == 0

    def test_valid_v02_two_level_pyramid_passes(self):
        """Two scale levels satisfies the v0.2 coarse-level rule; v0.1 required ≥ 3."""
        run = validate(
            str(FIXTURES / "v0_2/valid_two_level_pyramid.ome.zarr"),
            spec_version="0.2",
            show_progress=False,
        )
        assert run.summary.stores_failed == 0

    def test_valid_v02_plate_with_provenance_passes(self):
        run = validate(
            str(FIXTURES / "v0_2/valid_plate_with_provenance.ome.zarr"),
            spec_version="0.2",
            show_progress=False,
        )
        assert run.summary.stores_failed == 0
        assert not any("no provenance record" in i.message for i in _all_issues(run))


@pytest.mark.functional
class TestInvalidStoresV02:
    def test_missing_dca_is_error(self):
        run = validate(
            str(FIXTURES / "v0_2/invalid_missing_dca.ome.zarr"),
            spec_version="0.2",
            show_progress=False,
        )
        assert run.summary.stores_failed > 0
        assert any("'dca' key" in i.message for i in _all_issues(run))

    def test_missing_channel_field_is_error(self):
        run = validate(
            str(FIXTURES / "v0_2/invalid_missing_channel_field.ome.zarr"),
            spec_version="0.2",
            show_progress=False,
        )
        assert run.summary.stores_failed > 0
        assert any("description" in i.loc for i in _all_issues(run))

    def test_no_coarse_level_is_error(self):
        run = validate(
            str(FIXTURES / "v0_2/invalid_no_coarse_level.ome.zarr"),
            spec_version="0.2",
            show_progress=False,
        )
        assert run.summary.stores_failed > 0
        assert any("spatial dimensions" in i.message for i in _all_issues(run))


@pytest.mark.functional
class TestWellFieldSubsetting:
    """End-to-end coverage of --wells/--field-offset/--field-limit
    against valid_plate_multiwell.ome.zarr: 2 wells (A/1, B/1), 2 fields each.
    """

    PLATE = str(FIXTURES / "v0_2/valid_plate_multiwell.ome.zarr")

    def _field_node_paths(self, run):
        return sorted(
            r.node_path.rsplit(self.PLATE + "/", 1)[-1] for r in run if r.parent_path
        )

    def test_wells_narrows_field_results(self):
        run = validate(self.PLATE, spec_version="0.2", show_progress=False, wells="A/1")
        assert self._field_node_paths(run) == ["A/1/0", "A/1/1"]
        assert run.summary.is_partial is True
        assert run.summary.wells_selected == 1
        assert run.summary.wells_total == 2

    def test_field_offset_and_limit_narrow_fields_within_a_well(self):
        run = validate(
            self.PLATE,
            spec_version="0.2",
            show_progress=False,
            wells="A/1",
            field_offset=1,
            field_limit=1,
        )
        assert self._field_node_paths(run) == ["A/1/1"]
        assert run.summary.fields_selected == 1
        assert run.summary.fields_total == 2

    def test_full_run_has_all_four_fields_and_is_not_partial(self):
        run = validate(self.PLATE, spec_version="0.2", show_progress=False)
        assert self._field_node_paths(run) == ["A/1/0", "A/1/1", "B/1/0", "B/1/1"]
        assert run.summary.is_partial is False

    def test_subset_flags_with_discovery_root_raises(self):
        with pytest.raises(ValueError, match="single Zarr store"):
            validate(
                str(FIXTURES / "v0_2"),
                spec_version="0.2",
                show_progress=False,
                field_limit=1,
            )

    def test_subset_flags_against_non_plate_store_does_not_raise_but_errors_result(
        self,
    ):
        # validate() itself only raises for the discovery-root ambiguity;
        # a non-plate *single* store is a clean ERROR result, not an exception
        # (see TestValidateZarrNodeSubsetting for the unit-level equivalent).
        run = validate(
            str(FIXTURES / "v0_2/valid_two_level_pyramid.ome.zarr"),
            spec_version="0.2",
            show_progress=False,
            field_limit=1,
        )
        assert run.summary.stores_failed > 0
        assert any("only apply to HCS plates" in i.message for i in _all_issues(run))

    def test_empty_wells_string_raises(self):
        with pytest.raises(ValueError, match="at least one well path"):
            validate(self.PLATE, spec_version="0.2", show_progress=False, wells="  ")

    def test_negative_field_offset_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            validate(
                self.PLATE, spec_version="0.2", show_progress=False, field_offset=-1
            )

    def test_negative_field_limit_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            validate(
                self.PLATE, spec_version="0.2", show_progress=False, field_limit=-1
            )

    def test_list_plate_wells_reports_declared_counts_not_on_disk_counts(self):
        # invalid_missing_field.ome.zarr declares one field in well A/1 that
        # does not exist on disk. list_plate_wells reports the declared
        # count (1) rather than the on-disk count (0), and performs no
        # validation, so it doesn't raise despite the missing field being an
        # error under a real validate() run.
        wells = list_plate_wells(str(FIXTURES / "v0_1/invalid_missing_field.ome.zarr"))
        assert wells == [("A/1", 1)]
