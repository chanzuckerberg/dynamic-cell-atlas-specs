"""
Unit tests for validation.validator helpers.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import obstore.exceptions as _obs_exc
import pytest

from dca_helpers.validation.result import Severity
from dca_helpers.validation.validator import (
    _check_plate_schema,
    _check_well_perturbation_agreement,
    _extract_level,
    _gather_plate_fields,
    _get_compression,
    _level0_axis_scales,
    _validate_hcs_plate,
    validate_zarr_node,
)
from dca_helpers.validation.zarr_node import ZarrNodeType
from tests.validation.json_schema_oracle import SchemaNotFoundError, _find_schema_path


def _experimental_schema_on_path() -> bool:
    """The committed JSON Schema lives at the repo root (outside the package),
    so it is only reachable from an editable / source checkout. Skip the
    schema-dependent tests when it cannot be located (e.g. a non-editable
    install), mirroring how the drift-guard tests skip when linkml is absent."""
    try:
        _find_schema_path()
        return True
    except SchemaNotFoundError:
        return False


_needs_schema = pytest.mark.skipif(
    not _experimental_schema_on_path(),
    reason="generated experimental-metadata schema not on path (non-editable install)",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_well_group(images, keys):
    """Build a minimal well zarr group mock for _gather_plate_fields tests."""
    mock = MagicMock()
    mock.attrs = {"ome": {"well": {"images": [{"path": p} for p in images]}}}
    mock.keys.return_value = keys
    return mock


def make_well_group_with_perturbation(well_perturbation):
    """Well group mock carrying a dca.well_perturbation block, one field on disk."""
    mock = MagicMock()
    mock.attrs = {
        "ome": {"well": {"images": [{"path": "0"}]}},
        "dca": {"well_perturbation": well_perturbation},
    }
    mock.keys.return_value = ["0"]
    return mock


_UNSET = object()


def make_array_spec(shape, chunk_shape, codecs, dtype="uint16", dimension_names=_UNSET):
    """Build a minimal ArraySpec-like mock."""
    spec = MagicMock()
    spec.shape = shape
    spec.data_type = dtype
    spec.chunk_grid = {"configuration": {"chunk_shape": list(chunk_shape)}}
    spec.codecs = codecs
    spec.dimension_names = (
        ("t", "c", "z", "y", "x") if dimension_names is _UNSET else dimension_names
    )
    return spec


# ---------------------------------------------------------------------------
# _get_compression — codec dict chain parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetCompression:
    def test_zstd_codec(self):
        codecs = [{"name": "zstd", "configuration": {"level": 3}}]
        assert _get_compression(codecs) == ("zstd", 3)

    def test_blosc_codec(self):
        codecs = [{"name": "blosc", "configuration": {"clevel": 5}}]
        assert _get_compression(codecs) == ("blosc", 5)

    def test_lz4_codec(self):
        codecs = [{"name": "lz4", "configuration": {"acceleration": 1}}]
        assert _get_compression(codecs) == ("lz4", 1)

    def test_gzip_codec(self):
        codecs = [{"name": "gzip", "configuration": {"level": 6}}]
        assert _get_compression(codecs) == ("gzip", 6)

    def test_no_compression_codec(self):
        codecs = [{"name": "bytes", "configuration": {}}]
        assert _get_compression(codecs) == (None, None)

    def test_empty_codec_list(self):
        assert _get_compression([]) == (None, None)

    def test_sharding_with_inner_zstd(self):
        codecs = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [128, 128, 128],
                    "codecs": [
                        {"name": "bytes", "configuration": {}},
                        {"name": "zstd", "configuration": {"level": 1}},
                    ],
                },
            }
        ]
        assert _get_compression(codecs) == ("zstd", 1)

    def test_sharding_with_inner_blosc(self):
        codecs = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [64, 64, 64],
                    "codecs": [
                        {"name": "blosc", "configuration": {"clevel": 3}},
                    ],
                },
            }
        ]
        assert _get_compression(codecs) == ("blosc", 3)

    def test_bytes_then_zstd(self):
        codecs = [
            {"name": "bytes", "configuration": {}},
            {"name": "zstd", "configuration": {"level": 2}},
        ]
        assert _get_compression(codecs) == ("zstd", 2)


# ---------------------------------------------------------------------------
# _extract_level — without sharding
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractLevelNoShard:
    def test_basic_non_sharded(self):
        codecs = [{"name": "zstd", "configuration": {"level": 3}}]
        spec = make_array_spec(
            shape=[1, 1, 256, 512, 512],
            chunk_shape=[1, 1, 128, 128, 128],
            codecs=codecs,
        )
        result = _extract_level(0, "0", spec)
        assert result["level_index"] == 0
        assert result["path"] == "0"
        assert result["dtype"] == "uint16"
        assert result["shape"] == [1, 1, 256, 512, 512]
        assert result["chunk_shape"] == [1, 1, 128, 128, 128]
        assert result["shard_shape"] is None
        assert result["codec_id"] == "zstd"
        assert result["codec_level"] == 3
        assert result["dimension_names"] == ["t", "c", "z", "y", "x"]

    def test_no_compression(self):
        codecs = [{"name": "bytes", "configuration": {}}]
        spec = make_array_spec(
            shape=[1, 1, 64, 64, 64],
            chunk_shape=[1, 1, 64, 64, 64],
            codecs=codecs,
        )
        result = _extract_level(1, "1", spec)
        assert result["level_index"] == 1
        assert result["codec_id"] == "none"
        assert result["shard_shape"] is None
        assert result["dimension_names"] == ["t", "c", "z", "y", "x"]

    def test_missing_dimension_names(self):
        codecs = [{"name": "zstd", "configuration": {"level": 3}}]
        spec = make_array_spec(
            shape=[1, 1, 256, 512, 512],
            chunk_shape=[1, 1, 128, 128, 128],
            codecs=codecs,
            dimension_names=None,
        )
        result = _extract_level(0, "0", spec)
        assert result["dimension_names"] is None


# ---------------------------------------------------------------------------
# _extract_level — with sharding
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractLevelSharded:
    def test_sharded_extracts_inner_chunk(self):
        codecs = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [1, 1, 64, 64, 64],
                    "codecs": [
                        {"name": "bytes", "configuration": {}},
                        {"name": "zstd", "configuration": {"level": 1}},
                    ],
                },
            }
        ]
        spec = make_array_spec(
            shape=[1, 1, 512, 1024, 1024],
            chunk_shape=[1, 1, 256, 512, 512],  # shard shape (outer)
            codecs=codecs,
        )
        result = _extract_level(0, "0", spec)
        assert result["level_index"] == 0
        assert result["shard_shape"] == [1, 1, 256, 512, 512]
        assert result["chunk_shape"] == [1, 1, 64, 64, 64]
        assert result["codec_id"] == "zstd"
        assert result["codec_level"] == 1


# ---------------------------------------------------------------------------
# _gather_plate_fields — well traversal and MUST checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGatherPlateFields:
    def test_no_issues_all_present(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0"], ["0"]),
        )
        issues, field_paths, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        assert issues == []
        assert field_paths == ["A/1/0"]

    def test_multiple_wells(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0"], ["0"]),
        )
        issues, field_paths, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}, {"path": "B/2"}], spec_version="0.2"
        )
        assert issues == []
        assert field_paths == ["A/1/0", "B/2/0"]

    def test_missing_well_is_error(self, monkeypatch):
        def _raise(*a, **kw):
            raise Exception("not found")

        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _raise)
        issues, field_paths, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR
        assert issues[0].loc == ("plate", "wells", "A/1")
        assert field_paths == []

    def test_missing_well_via_file_not_found_is_still_missing(self, monkeypatch):
        def _raise(*a, **kw):
            raise FileNotFoundError("no such key")

        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _raise)
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        assert [i.loc for i in issues] == [("plate", "wells", "A/1")]

    def test_s3_error_opening_well_is_s3_io_not_missing(self, monkeypatch):
        def _raise(*a, **kw):
            raise _obs_exc.GenericError("slow down")

        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _raise)
        issues, field_paths, _, perturbations = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        assert [i.loc for i in issues] == [("s3_io",)]
        assert "A/1" in issues[0].message
        assert "no group on disk" not in issues[0].message
        assert issues[0].severity == Severity.ERROR
        assert field_paths == []
        assert perturbations == {}

    def test_s3_error_listing_well_is_s3_io_not_missing_fields(self, monkeypatch):
        group = make_well_group(["0", "1"], [])
        group.keys.side_effect = _obs_exc.GenericError("slow down")
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: group,
        )
        issues, field_paths, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        assert [i.loc for i in issues] == [("s3_io",)]
        assert "A/1" in issues[0].message
        assert field_paths == []

    def test_missing_field_is_error(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0", "missing"], ["0"]),
        )
        issues, field_paths, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR
        assert issues[0].loc == ("plate", "wells", "A/1", "missing")

    def test_missing_field_not_in_field_paths(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0", "missing"], ["0"]),
        )
        _, field_paths, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        assert "A/1/0" in field_paths
        assert "A/1/missing" not in field_paths

    def test_multiple_missing_wells_each_produce_error(self, monkeypatch):
        def _raise(*a, **kw):
            raise Exception("not found")

        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _raise)
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}, {"path": "B/2"}], spec_version="0.2"
        )
        assert len(issues) == 2
        assert all(i.severity == Severity.ERROR for i in issues)

    def test_no_acquisition_check_when_plate_acquisitions_is_none(self, monkeypatch):
        mock = MagicMock()
        mock.attrs = {"ome": {"well": {"images": [{"path": "0", "acquisition": 99}]}}}
        mock.keys.return_value = ["0"]
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock,
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], None, spec_version="0.2"
        )
        assert issues == []

    def test_valid_acquisition_id_no_issue(self, monkeypatch):
        mock = MagicMock()
        mock.attrs = {"ome": {"well": {"images": [{"path": "0", "acquisition": 1}]}}}
        mock.keys.return_value = ["0"]
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock,
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], [{"id": 1}, {"id": 2}], spec_version="0.2"
        )
        assert issues == []

    def test_invalid_acquisition_id_is_error(self, monkeypatch):
        mock = MagicMock()
        mock.attrs = {"ome": {"well": {"images": [{"path": "0", "acquisition": 99}]}}}
        mock.keys.return_value = ["0"]
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock,
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], [{"id": 1}, {"id": 2}], spec_version="0.2"
        )
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR
        assert issues[0].loc == ("plate", "wells", "A/1", "0", "acquisition")
        assert "99" in issues[0].message

    def test_non_dict_acquisition_entry_excluded_from_valid_set(self, monkeypatch):
        # A non-dict entry in plate.acquisitions must not crash the
        # valid_aq_ids comprehension's aq.get("id") access, and must not
        # itself count as a valid acquisition id.
        mock = MagicMock()
        mock.attrs = {"ome": {"well": {"images": [{"path": "0", "acquisition": 1}]}}}
        mock.keys.return_value = ["0"]
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock,
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], ["not-a-dict", {"id": 1}], spec_version="0.2"
        )
        assert issues == []

    def test_non_dict_well_entry_in_plate_wells_is_skipped_not_a_crash(
        self, monkeypatch
    ):
        # A non-dict entry in plate.wells (e.g. a bare string) must not crash
        # well_meta.get("path", "") — it's not a real well to open.
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0"], ["0"]),
        )
        issues, field_paths, _, _ = _gather_plate_fields(
            "/p", ["not-a-dict", {"path": "A/1"}], spec_version="0.2"
        )
        assert issues == []
        assert field_paths == ["A/1/0"]

    def test_non_dict_image_meta_entry_is_skipped_not_a_crash(self, monkeypatch):
        # A non-dict entry in well.images (e.g. a bare string) must not crash
        # img_meta.get("path", "") — it's not a real field to validate.
        mock = MagicMock()
        mock.attrs = {"ome": {"well": {"images": ["not-a-dict", {"path": "0"}]}}}
        mock.keys.return_value = ["0"]
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock,
        )
        issues, field_paths, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        assert issues == []
        assert field_paths == ["A/1/0"]

    def test_malformed_acquisition_excluded_from_valid_set(self, monkeypatch):
        # Acquisition with null id is excluded from valid_aq_ids; a field
        # referencing a real integer id should still get a cross-reference error.
        mock = MagicMock()
        mock.attrs = {"ome": {"well": {"images": [{"path": "0", "acquisition": 0}]}}}
        mock.keys.return_value = ["0"]
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock,
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], [{"id": None}], spec_version="0.2"
        )
        assert any(
            i.severity == Severity.ERROR and "acquisition" in i.loc for i in issues
        )

    def test_null_acquisition_field_skipped(self, monkeypatch):
        mock = MagicMock()
        mock.attrs = {"ome": {"well": {"images": [{"path": "0", "acquisition": None}]}}}
        mock.keys.return_value = ["0"]
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock,
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], [{"id": 1}], spec_version="0.2"
        )
        assert issues == []


# ---------------------------------------------------------------------------
# _gather_plate_fields — well/field subset selection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGatherPlateFieldsSubsetting:
    def test_malformed_well_missing_path_is_skipped_not_a_crash(self, monkeypatch):
        # A {} well entry (missing "path") must not raise KeyError — it's
        # already reported by _check_plate_schema's well-path syntax check,
        # so _gather_plate_fields just skips it.
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0"], ["0"]),
        )
        issues, field_paths, selected_field_paths, _ = _gather_plate_fields(
            "/p", [{}, {"path": "A/1"}], spec_version="0.2"
        )
        assert issues == []
        assert field_paths == ["A/1/0"]
        assert selected_field_paths == ["A/1/0"]

    def test_well_filter_excludes_non_selected_well(self, monkeypatch):
        opened_paths = []

        def _open(path, *a, **kw):
            opened_paths.append(path)
            return make_well_group(["0"], ["0"])

        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _open)
        issues, field_paths, selected_field_paths, well_perturbation_by_well = (
            _gather_plate_fields(
                "/p",
                [{"path": "A/1"}, {"path": "B/2"}],
                spec_version="0.2",
                well_filter=frozenset({"A/1"}),
            )
        )
        assert issues == []
        assert field_paths == ["A/1/0"]
        assert selected_field_paths == ["A/1/0"]
        assert well_perturbation_by_well == {"A/1": None}
        assert opened_paths == ["/p/A/1"]  # B/2 was never opened

    def test_well_filter_none_opens_every_well(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0"], ["0"]),
        )
        _, field_paths, selected_field_paths, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}, {"path": "B/2"}], spec_version="0.2"
        )
        assert field_paths == ["A/1/0", "B/2/0"]
        assert selected_field_paths == field_paths

    def test_field_limit_slices_selected_but_not_declared(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0", "1", "2"], ["0", "1", "2"]),
        )
        issues, field_paths, selected_field_paths, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2", field_limit=2
        )
        assert issues == []
        assert field_paths == ["A/1/0", "A/1/1", "A/1/2"]
        assert selected_field_paths == ["A/1/0", "A/1/1"]

    def test_field_offset_and_limit_slice_per_well(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0", "1", "2"], ["0", "1", "2"]),
        )
        _, field_paths, selected_field_paths, _ = _gather_plate_fields(
            "/p",
            [{"path": "A/1"}, {"path": "B/2"}],
            spec_version="0.2",
            field_offset=1,
            field_limit=1,
        )
        assert field_paths == ["A/1/0", "A/1/1", "A/1/2", "B/2/0", "B/2/1", "B/2/2"]
        # offset/limit applied independently to each well, not to the flattened list
        assert selected_field_paths == ["A/1/1", "B/2/1"]

    def test_out_of_range_offset_yields_empty_selection_without_error(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0"], ["0"]),
        )
        issues, field_paths, selected_field_paths, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2", field_offset=100
        )
        assert issues == []
        assert field_paths == ["A/1/0"]
        assert selected_field_paths == []

    def test_field_limit_does_not_suppress_structural_checks(self, monkeypatch):
        # A field beyond the selected slice that is missing on disk still
        # produces the usual MUST error — slicing only narrows which fields
        # get deep-validated later, not the metadata checks in this function.
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0", "1"], ["0"]),
        )
        issues, field_paths, selected_field_paths, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2", field_limit=1
        )
        assert len(issues) == 1
        assert issues[0].loc == ("plate", "wells", "A/1", "1")
        assert field_paths == ["A/1/0"]
        assert selected_field_paths == ["A/1/0"]

    def test_null_images_treated_as_no_declared_fields_not_a_crash(self, monkeypatch):
        # A well with "images": null (rather than a list) must not raise
        # TypeError iterating None — treat it as declaring no fields.
        mock = MagicMock()
        mock.attrs = {"ome": {"well": {"images": None}}}
        mock.keys.return_value = []
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock,
        )
        issues, field_paths, selected_field_paths, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        assert issues == []
        assert field_paths == []
        assert selected_field_paths == []


# ---------------------------------------------------------------------------
# _gather_plate_fields — dca.well_perturbation dispatch
#
# Dispatches to validate_well_perturbation_v0_2 rather than the image-group
# entry point. WellPerturbation's own content rules (non-empty ⇒ definition,
# modality ⇒ typed-block, plate/well requiredness) are already pinned against
# validate_well_perturbation_v0_2 directly in
# tests/validation/versions/v0_2/test_experimental_metadata.py, so these focus
# on what's unique to the dispatch: the loc prefix, the absent/valid no-op
# paths, non-dict input safety, and the spec_version gate.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGatherPlateFieldsWellPerturbation:
    def test_absent_well_perturbation_no_issue(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0"], ["0"]),
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], None, spec_version="0.2"
        )
        assert issues == []

    def test_valid_well_perturbation_no_issue(self, monkeypatch):
        valid = {"plate": "P1", "well": "A1", "control_class": "empty"}
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group_with_perturbation(valid),
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], None, spec_version="0.2"
        )
        assert issues == []

    def test_compound_without_chemical_is_error(self, monkeypatch):
        bad = {
            "plate": "P1",
            "well": "A1",
            "control_class": "treatment",
            "definition": {"perturbation_id": "X", "modality": "compound"},
        }
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group_with_perturbation(bad),
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], None, spec_version="0.2"
        )
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR
        assert issues[0].loc[:5] == (
            "plate",
            "wells",
            "A/1",
            "dca",
            "well_perturbation",
        )
        assert "chemical" in issues[0].message

    def test_non_dict_well_perturbation_is_error_not_crash(self, monkeypatch):
        # A hand-edited store could put anything under the key. model_validate
        # rejects a non-dict with a normal pydantic type error rather than
        # raising an unhandled exception from _gather_plate_fields itself.
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group_with_perturbation("not-an-object"),
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], None, spec_version="0.2"
        )
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR

    def test_not_checked_when_spec_version_is_not_experimental(self, monkeypatch):
        # v0.1 predates experimental metadata; gated the same way as the
        # image-level check (_EXPERIMENTAL_SPEC_VERSIONS).
        bad = {
            "plate": "P1",
            "well": "A1",
            "control_class": "treatment",
            "definition": {"perturbation_id": "X", "modality": "compound"},
        }
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group_with_perturbation(bad),
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], None, spec_version="0.1"
        )
        assert issues == []

    def test_spec_version_is_required(self):
        # spec_version has no default: an omitted argument is a caller bug and
        # must raise immediately rather than silently disabling the check.
        with pytest.raises(TypeError):
            _gather_plate_fields("/p", [{"path": "A/1"}])

    def test_prerename_definition_key_is_rejected(self, monkeypatch):
        # A regression fixture for the slot rename: the old `perturbation`
        # key (rather than the renamed `definition`) must be rejected as an
        # extra field. Current writers emit `definition`, so this only pins the
        # rename going forward.
        prerename = {
            "plate": "P1",
            "well": "A1",
            "control_class": "treatment",
            "perturbation": {
                "perturbation_id": "X",
                "modality": "biological",
                "name": "n",
                "description": "d",
            },
        }
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group_with_perturbation(prerename),
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], None, spec_version="0.2"
        )
        assert any(
            i.severity == Severity.ERROR and "extra" in i.message.lower()
            for i in issues
        )

    def test_postrename_definition_key_is_clean(self, monkeypatch):
        # Matches the shape current writers produce: the renamed `definition`
        # key, plus fields (solvent, concentration, provenance) that writers
        # emit but the other fixtures in this class omit.
        postrename = {
            "plate": "P1",
            "well": "A1",
            "control_class": "treatment",
            "solvent": "DMSO",
            "concentration": {"value": 10.0, "unit": "uM"},
            "provenance": {"source_kind": "original_data"},
            "definition": {
                "perturbation_id": "X",
                "modality": "biological",
                "name": "n",
                "description": "d",
            },
        }
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group_with_perturbation(postrename),
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], None, spec_version="0.2"
        )
        assert issues == []


# ---------------------------------------------------------------------------
# _validate_hcs_plate — real spec_version threading into _gather_plate_fields
# (Every test above calls _gather_plate_fields directly with a literal
# spec_version, so none of them exercise the call site in _validate_hcs_plate.
# These do not mock _gather_plate_fields away — they drive the real entry point
# so a dropped/misplaced argument would surface.)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateHcsPlateWellPerturbationWiring:
    def _run(self, monkeypatch, well_perturbation, spec_version):
        node_path = "/fake/plate"
        raw_plate_attrs = {"wells": [{"path": "A/1"}]}

        # zarr.open_group is patched to return this same mock for every path
        # opened (well and field alike), so the field's dca.perturbation is
        # given the well row's own assignment fields (sans plate/well) to
        # keep it out of disagreement with the well_perturbation row. This
        # test is about the content-check dispatch, not cross-placement
        # agreement — TestCheckWellPerturbationAgreement covers that.
        assignment_only = {
            k: v for k, v in well_perturbation.items() if k not in ("plate", "well")
        }
        shared_mock = MagicMock()
        shared_mock.attrs = {
            "ome": {"well": {"images": [{"path": "0"}]}},
            "dca": {
                "well_perturbation": well_perturbation,
                "perturbation": assignment_only,
            },
        }
        shared_mock.keys.return_value = ["0"]

        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: shared_mock,
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator.open_ome_zarr",
            lambda *a, **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator._build_node_dict",
            lambda *a, **kw: _make_node_dict(
                ("t", "c", "z", "y", "x"), [1, 1, 1, 1, 1]
            ),
        )
        results = _validate_hcs_plate(
            node_path, raw_plate_attrs, "0.5", spec_version, MagicMock()
        )
        plate_result = results[0]
        assert plate_result.node_path == node_path
        return plate_result.issues

    def test_bad_well_block_surfaces_through_real_entry_point(self, monkeypatch):
        bad = {
            "plate": "P1",
            "well": "A1",
            "control_class": "treatment",
            "definition": {"perturbation_id": "X", "modality": "compound"},
        }
        issues = self._run(monkeypatch, bad, spec_version="0.2")
        wp_issues = [
            i
            for i in issues
            if i.severity == Severity.ERROR
            and i.loc[:5]
            == (
                "plate",
                "wells",
                "A/1",
                "dca",
                "well_perturbation",
            )
        ]
        assert len(wp_issues) == 1
        assert "chemical" in wp_issues[0].message

    def test_same_bad_block_produces_no_issue_on_non_experimental_spec_version(
        self, monkeypatch
    ):
        # Control for the case above: identical bad block, but spec_version
        # doesn't reach _EXPERIMENTAL_SPEC_VERSIONS, so the gate must suppress
        # it — through the real _validate_hcs_plate call site, not a mock.
        bad = {
            "plate": "P1",
            "well": "A1",
            "control_class": "treatment",
            "definition": {"perturbation_id": "X", "modality": "compound"},
        }
        issues = self._run(monkeypatch, bad, spec_version="0.1")
        wp_issues = [i for i in issues if "well_perturbation" in i.loc]
        assert wp_issues == []


# ---------------------------------------------------------------------------
# _check_well_perturbation_agreement — cross-placement agreement
#
# Pure function: field-of-view perturbation blocks and well_perturbation rows
# are passed in directly rather than mocking zarr groups. Per well: fields
# must agree with each other regardless of row presence; only once they agree
# is the (optional) row compared against the derived consensus.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckWellPerturbationAgreement:
    def test_agreeing_fields_and_matching_row_no_issue(self):
        pert = {"control_class": "empty"}
        row = {"plate": "P1", "well": "A1", "control_class": "empty"}
        issues = _check_well_perturbation_agreement(
            ["A/1/0", "A/1/1"],
            {"A/1/0": pert, "A/1/1": pert},
            {"A/1": row},
        )
        assert issues == []

    def test_agreeing_fields_no_well_row_no_issue(self):
        pert = {"control_class": "empty"}
        issues = _check_well_perturbation_agreement(
            ["A/1/0", "A/1/1"],
            {"A/1/0": pert, "A/1/1": pert},
            {"A/1": None},
        )
        assert issues == []

    def test_no_perturbation_anywhere_no_issue(self):
        # A dataset with no perturbations at all: every field's perturbation
        # is absent (None), and there is no well row either.
        issues = _check_well_perturbation_agreement(
            ["A/1/0", "A/1/1"],
            {"A/1/0": None, "A/1/1": None},
            {"A/1": None},
        )
        assert issues == []

    def test_disagreeing_fields_is_error_even_without_well_row(self):
        issues = _check_well_perturbation_agreement(
            ["A/1/0", "A/1/1"],
            {
                "A/1/0": {"control_class": "empty"},
                "A/1/1": {"control_class": "treatment"},
            },
            {"A/1": None},
        )
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR
        assert issues[0].loc == ("plate", "wells", "A/1", "dca", "perturbation")
        assert "not identical" in issues[0].message

    def test_agreeing_fields_but_row_mismatch_is_error(self):
        pert = {"control_class": "empty"}
        row = {"plate": "P1", "well": "A1", "control_class": "treatment"}
        issues = _check_well_perturbation_agreement(
            ["A/1/0", "A/1/1"],
            {"A/1/0": pert, "A/1/1": pert},
            {"A/1": row},
        )
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR
        assert issues[0].loc == ("plate", "wells", "A/1", "dca", "well_perturbation")
        assert "disagrees" in issues[0].message

    def test_malformed_well_perturbation_is_mismatch_not_crash(self):
        # A hand-edited store could put anything under well_perturbation.
        # validate_well_perturbation_v0_2 (in _gather_plate_fields) already
        # reports the shape defect; this only needs to not raise, and a
        # non-dict can never equal a real assignment, so it surfaces as an
        # (accurate, if redundant) mismatch rather than being silently skipped.
        issues = _check_well_perturbation_agreement(
            ["A/1/0"],
            {"A/1/0": {"control_class": "empty"}},
            {"A/1": "not-an-object"},
        )
        assert len(issues) == 1
        assert issues[0].loc == ("plate", "wells", "A/1", "dca", "well_perturbation")

    def test_wells_checked_independently(self):
        # Well A/1 agrees and matches; well A/2 disagrees internally. Only
        # A/2 should produce an issue.
        pert = {"control_class": "empty"}
        row = {"plate": "P1", "well": "A1", "control_class": "empty"}
        issues = _check_well_perturbation_agreement(
            ["A/1/0", "A/1/1", "A/2/0", "A/2/1"],
            {
                "A/1/0": pert,
                "A/1/1": pert,
                "A/2/0": {"control_class": "empty"},
                "A/2/1": {"control_class": "treatment"},
            },
            {"A/1": row, "A/2": None},
        )
        assert len(issues) == 1
        assert issues[0].loc[2] == "A/2"

    def test_well_with_no_fields_not_checked(self):
        # well_perturbation_by_well can carry a well that has no entries in
        # field_paths (e.g. a well group with zero declared images) — there
        # is no image-group data to derive a consensus from, so it is skipped
        # rather than flagged.
        issues = _check_well_perturbation_agreement(
            [],
            {},
            {"A/1": {"plate": "P1", "well": "A1", "control_class": "treatment"}},
        )
        assert issues == []

    def test_field_null_vs_omitted_optional_field_not_flagged(self):
        # One field writes an optional field explicitly as null; the other
        # omits it entirely. Both mean "no value" and must not be flagged as
        # a field-vs-field disagreement.
        issues = _check_well_perturbation_agreement(
            ["A/1/0", "A/1/1"],
            {
                "A/1/0": {"control_class": "empty", "solvent": None},
                "A/1/1": {"control_class": "empty"},
            },
            {"A/1": None},
        )
        assert issues == []

    def test_well_row_null_vs_omitted_optional_field_not_flagged(self):
        # The well row writes an optional field explicitly as null; the
        # field-of-view consensus omits it. Both mean "no value" and must
        # not be flagged as a row-vs-consensus mismatch.
        pert = {"control_class": "empty"}
        row = {
            "plate": "P1",
            "well": "A1",
            "control_class": "empty",
            "solvent": None,
        }
        issues = _check_well_perturbation_agreement(
            ["A/1/0", "A/1/1"],
            {"A/1/0": pert, "A/1/1": pert},
            {"A/1": row},
        )
        assert issues == []


# ---------------------------------------------------------------------------
# _validate_hcs_plate — cross-placement agreement wiring
#
# Drives the real entry point end-to-end (real _gather_plate_fields and
# _validate_field, mocked only at the zarr/ome-zarr-models I/O boundary) so a
# dropped or misplaced argument in the wiring would surface, mirroring
# TestValidateHcsPlateWellPerturbationWiring's rationale.
# ---------------------------------------------------------------------------


def _make_field_mock(perturbation):
    mock = MagicMock()
    mock.attrs = {"dca": {"perturbation": perturbation} if perturbation else {}}
    return mock


def _make_well_mock(well_perturbation, image_paths):
    mock = MagicMock()
    mock.attrs = {
        "ome": {"well": {"images": [{"path": p} for p in image_paths]}},
        "dca": {"well_perturbation": well_perturbation} if well_perturbation else {},
    }
    mock.keys.return_value = image_paths
    return mock


@pytest.mark.unit
class TestValidateHcsPlateCrossPlacementAgreementWiring:
    def _run(self, monkeypatch, well_perturbation, field_perturbations):
        node_path = "/fake/plate"
        raw_plate_attrs = {"wells": [{"path": "A/1"}]}
        image_paths = [str(i) for i in range(len(field_perturbations))]

        mocks_by_path = {
            f"{node_path}/A/1": _make_well_mock(well_perturbation, image_paths),
        }
        for image_path, perturbation in zip(image_paths, field_perturbations):
            mocks_by_path[f"{node_path}/A/1/{image_path}"] = _make_field_mock(
                perturbation
            )

        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda path, *a, **kw: mocks_by_path[path],
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator.open_ome_zarr",
            lambda *a, **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator._build_node_dict",
            lambda *a, **kw: _make_node_dict(
                ("t", "c", "z", "y", "x"), [1, 1, 1, 1, 1]
            ),
        )
        results = _validate_hcs_plate(
            node_path, raw_plate_attrs, "0.5", "0.2", MagicMock()
        )
        return results[0].issues

    def test_agreeing_fields_and_matching_row_no_issue(self, monkeypatch):
        pert = {"control_class": "empty"}
        row = {"plate": "P1", "well": "A1", "control_class": "empty"}
        issues = self._run(monkeypatch, row, [pert, pert])
        cross_check = [
            i
            for i in issues
            if i.loc[:4] == ("plate", "wells", "A/1", "dca")
            and i.loc[4] in ("perturbation", "well_perturbation")
        ]
        assert cross_check == []

    def test_disagreeing_fields_surfaces_through_real_entry_point(self, monkeypatch):
        issues = self._run(
            monkeypatch,
            None,
            [{"control_class": "empty"}, {"control_class": "treatment"}],
        )
        errors = [
            i
            for i in issues
            if i.severity == Severity.ERROR
            and i.loc == ("plate", "wells", "A/1", "dca", "perturbation")
        ]
        assert len(errors) == 1

    def test_row_mismatch_surfaces_through_real_entry_point(self, monkeypatch):
        pert = {"control_class": "empty"}
        # Internally valid on its own (non-empty control_class with a
        # definition) so the only well_perturbation-loc'd issue is the new
        # cross-check, not the pre-existing content check.
        row = {
            "plate": "P1",
            "well": "A1",
            "control_class": "treatment",
            "definition": {"perturbation_id": "X", "modality": "biological"},
        }
        issues = self._run(monkeypatch, row, [pert, pert])
        errors = [
            i
            for i in issues
            if i.severity == Severity.ERROR
            and i.loc == ("plate", "wells", "A/1", "dca", "well_perturbation")
        ]
        assert len(errors) == 1
        assert "disagrees" in errors[0].message


# ---------------------------------------------------------------------------
# _validate_hcs_plate — well/field subset selection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateHcsPlateSubsetting:
    def _run(
        self, monkeypatch, field_perturbations, extra_well_paths=(), **subset_kwargs
    ):
        node_path = "/fake/plate"
        raw_plate_attrs = {
            "wells": [{"path": "A/1"}] + [{"path": p} for p in extra_well_paths]
        }
        image_paths = [str(i) for i in range(len(field_perturbations))]

        mocks_by_path = {
            f"{node_path}/A/1": _make_well_mock(None, image_paths),
        }
        for image_path, perturbation in zip(image_paths, field_perturbations):
            mocks_by_path[f"{node_path}/A/1/{image_path}"] = _make_field_mock(
                perturbation
            )

        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda path, *a, **kw: mocks_by_path[path],
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator.open_ome_zarr",
            lambda *a, **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator._build_node_dict",
            lambda *a, **kw: _make_node_dict(
                ("t", "c", "z", "y", "x"), [1, 1, 1, 1, 1]
            ),
        )
        return _validate_hcs_plate(
            node_path, raw_plate_attrs, "0.5", "0.2", MagicMock(), **subset_kwargs
        )

    def test_field_limit_restricts_deep_validated_field_count(self, monkeypatch):
        pert = {"control_class": "empty"}
        results = self._run(monkeypatch, [pert, pert, pert], field_limit=1)
        field_results = [r for r in results if r.parent_path is not None]
        assert len(field_results) == 1
        assert field_results[0].node_path == "/fake/plate/A/1/0"

    def test_plate_root_always_returned_even_when_zero_fields_selected(
        self, monkeypatch
    ):
        # A chunk that selects zero fields (field_offset/field_limit slicing
        # past every declared field) must still return the plate root --
        # a downstream query grouping node rows by parent_path relies on
        # every plate's results always including exactly one
        # parent_path-is-None root row, even for a chunk this narrow.
        pert = {"control_class": "empty"}
        results = self._run(monkeypatch, [pert], field_offset=100, field_limit=1)
        assert len(results) == 1
        assert results[0].parent_path is None

    def test_disagreement_outside_slice_is_not_flagged(self, monkeypatch):
        # Field "1" would disagree with field "0"; excluding it via
        # field_limit=1 must not surface the disagreement — only fields
        # actually deep-validated participate in the consensus check.
        results = self._run(
            monkeypatch,
            [{"control_class": "empty"}, {"control_class": "treatment"}],
            field_limit=1,
        )
        cross_check = [
            i
            for i in results[0].issues
            if i.loc == ("plate", "wells", "A/1", "dca", "perturbation")
        ]
        assert cross_check == []

    def test_full_run_still_flags_the_same_disagreement(self, monkeypatch):
        # Sanity check for the test above: without field_limit the same
        # disagreement IS flagged, proving its absence above is due to
        # slicing rather than a broken agreement check.
        results = self._run(
            monkeypatch,
            [{"control_class": "empty"}, {"control_class": "treatment"}],
        )
        cross_check = [
            i
            for i in results[0].issues
            if i.loc == ("plate", "wells", "A/1", "dca", "perturbation")
        ]
        assert len(cross_check) == 1

    def test_subset_info_absent_on_a_full_run(self, monkeypatch):
        results = self._run(monkeypatch, [{"control_class": "empty"}])
        assert results[0].subset_coverage is None

    def test_subset_info_attached_and_correct_when_field_limited(self, monkeypatch):
        # Two fields declared, field_limit=1 selects only one — a genuine
        # narrowing, unlike field_limit >= the well's own field count.
        pert = {"control_class": "empty"}
        results = self._run(monkeypatch, [pert, pert], field_limit=1)
        subset_coverage = results[0].subset_coverage
        assert subset_coverage is not None
        assert subset_coverage.wells_total == 1
        assert subset_coverage.wells_selected == 1
        assert subset_coverage.fields_total == 2
        assert subset_coverage.fields_selected == 1

    def test_subset_info_attached_when_well_filter_set(self, monkeypatch):
        # Plate declares two wells; well_filter selects only one — a genuine
        # narrowing, unlike a well_filter naming every declared well.
        pert = {"control_class": "empty"}
        results = self._run(
            monkeypatch,
            [pert],
            extra_well_paths=["B/2"],
            well_filter=frozenset({"A/1"}),
        )
        subset_coverage = results[0].subset_coverage
        assert subset_coverage is not None
        assert subset_coverage.wells_total == 2
        assert subset_coverage.wells_selected == 1

    def test_subset_not_attached_when_wells_filter_names_every_well(self, monkeypatch):
        pert = {"control_class": "empty"}
        results = self._run(monkeypatch, [pert], well_filter=frozenset({"A/1"}))
        assert results[0].subset_coverage is None

    def test_subset_not_attached_when_field_limit_exceeds_field_count(
        self, monkeypatch
    ):
        pert = {"control_class": "empty"}
        results = self._run(monkeypatch, [pert], field_limit=100)
        assert results[0].subset_coverage is None

    def test_wells_total_excludes_malformed_well_entries(self, monkeypatch):
        # A malformed well (blank "path") must not count toward wells_total —
        # selecting every valid well via well_filter must not look "partial"
        # just because of an uncountable malformed entry alongside it.
        pert = {"control_class": "empty"}
        results = self._run(
            monkeypatch, [pert], extra_well_paths=[""], well_filter=frozenset({"A/1"})
        )
        assert results[0].subset_coverage is None

    def test_wells_total_excludes_non_dict_well_entries(self, monkeypatch):
        # A non-dict entry in plate.wells must not crash wells_total's
        # w.get("path", "") check, nor count toward wells_total — selecting
        # every valid well via well_filter must not look "partial" just
        # because of an uncountable non-dict entry.
        node_path = "/fake/plate"
        raw_plate_attrs = {"wells": [{"path": "A/1"}, "not-a-dict"]}
        mocks_by_path = {
            f"{node_path}/A/1": _make_well_mock(None, ["0"]),
            f"{node_path}/A/1/0": _make_field_mock({"control_class": "empty"}),
        }
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda path, *a, **kw: mocks_by_path[path],
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator.open_ome_zarr",
            lambda *a, **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator._build_node_dict",
            lambda *a, **kw: _make_node_dict(
                ("t", "c", "z", "y", "x"), [1, 1, 1, 1, 1]
            ),
        )
        results = _validate_hcs_plate(
            node_path,
            raw_plate_attrs,
            "0.5",
            "0.2",
            MagicMock(),
            well_filter=frozenset({"A/1"}),
        )
        assert results[0].subset_coverage is None

    def test_field_progress_hook_receives_selected_fields_only(self, monkeypatch):
        pert = {"control_class": "empty"}
        received = []

        def hook(field_paths):
            received.append(list(field_paths))
            return {}

        self._run(
            monkeypatch, [pert, pert, pert], field_progress_hook=hook, field_limit=2
        )
        assert received == [["A/1/0", "A/1/1"]]


# ---------------------------------------------------------------------------
# validate_zarr_node — well/field subset guard rails
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateZarrNodeSubsetting:
    def test_negative_field_offset_is_error_not_a_slice_footgun(self, monkeypatch):
        # A direct validate_zarr_node() caller (bypassing runner.validate()'s
        # own guard) must still get a clean error, not silent negative-slice
        # semantics (e.g. declared[-1:] selecting from the end).
        zarr_group_mock = MagicMock()
        zarr_group_mock.attrs = {
            "ome": {"plate": {"wells": [{"path": "A/1"}]}, "version": "0.5"}
        }
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: zarr_group_mock,
        )
        results = validate_zarr_node("/fake/plate", spec_version="0.2", field_offset=-1)
        assert len(results) == 1
        assert results[0].passed is False
        assert "must be >= 0" in results[0].issues[0].message

    def test_subset_flags_on_non_hcs_plate_is_error(self, monkeypatch):
        zarr_group_mock = MagicMock()
        zarr_group_mock.attrs = {
            "ome": {"multiscales": [{"axes": [], "datasets": []}], "version": "0.5"}
        }
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: zarr_group_mock,
        )
        results = validate_zarr_node("/fake/image", spec_version="0.2", field_limit=1)
        assert len(results) == 1
        assert results[0].passed is False
        assert "only apply to HCS plates" in results[0].issues[0].message

    def test_unknown_well_name_is_error_before_any_well_is_opened(self, monkeypatch):
        # Only the plate root is ever opened here; if the unknown-well check
        # didn't short-circuit before _gather_plate_fields, opening a well
        # group would raise (the mock only knows how to open the plate root).
        zarr_group_mock = MagicMock()
        zarr_group_mock.attrs = {
            "ome": {
                "plate": {"wells": [{"path": "A/1"}, {"path": "B/2"}]},
                "version": "0.5",
            }
        }
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: zarr_group_mock,
        )
        results = validate_zarr_node(
            "/fake/plate",
            spec_version="0.2",
            wells=frozenset({"C/3", "D/4"}),
        )
        assert len(results) == 1
        message = results[0].issues[0].message
        assert "C/3" in message and "D/4" in message
        assert "A/1" in message and "B/2" in message

    def test_malformed_well_missing_path_does_not_crash_unknown_well_check(
        self, monkeypatch
    ):
        # A well entry missing "path" must not make the unknown-well check's
        # sorted() call raise TypeError (comparing None to str) — it should
        # still cleanly report the actually-unknown well.
        zarr_group_mock = MagicMock()
        zarr_group_mock.attrs = {
            "ome": {
                "plate": {"wells": [{"path": "A/1"}, {}]},
                "version": "0.5",
            }
        }
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: zarr_group_mock,
        )
        results = validate_zarr_node(
            "/fake/plate", spec_version="0.2", wells=frozenset({"Z/9"})
        )
        assert len(results) == 1
        assert "Z/9" in results[0].issues[0].message
        # The malformed entry's blank "path" must not leak into the
        # "declared wells" list as a confusing empty string.
        message = results[0].issues[0].message
        assert "''" not in message and '""' not in message

    def test_non_dict_well_entry_does_not_crash_unknown_well_check(self, monkeypatch):
        # A non-dict well entry (e.g. a bare string) must not crash the
        # unknown-well check's w.get("path", "") access.
        zarr_group_mock = MagicMock()
        zarr_group_mock.attrs = {
            "ome": {
                "plate": {"wells": [{"path": "A/1"}, "not-a-dict"]},
                "version": "0.5",
            }
        }
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: zarr_group_mock,
        )
        results = validate_zarr_node(
            "/fake/plate", spec_version="0.2", wells=frozenset({"Z/9"})
        )
        assert len(results) == 1
        assert "Z/9" in results[0].issues[0].message

    def test_non_string_well_path_does_not_crash_unknown_well_check(self, monkeypatch):
        # A well entry with a non-string "path" (e.g. an int) mixed with a
        # string path must not crash sorted(declared_well_paths) — mixed
        # str/int values raise TypeError on comparison.
        zarr_group_mock = MagicMock()
        zarr_group_mock.attrs = {
            "ome": {
                "plate": {"wells": [{"path": "A/1"}, {"path": 123}]},
                "version": "0.5",
            }
        }
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: zarr_group_mock,
        )
        results = validate_zarr_node(
            "/fake/plate", spec_version="0.2", wells=frozenset({"Z/9"})
        )
        assert len(results) == 1
        assert "Z/9" in results[0].issues[0].message

    def test_null_plate_wells_does_not_crash_unknown_well_check(self, monkeypatch):
        # ome.plate.wells: null (rather than a list) must not crash the
        # declared_well_paths comprehension's iteration.
        zarr_group_mock = MagicMock()
        zarr_group_mock.attrs = {"ome": {"plate": {"wells": None}, "version": "0.5"}}
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: zarr_group_mock,
        )
        results = validate_zarr_node(
            "/fake/plate", spec_version="0.2", wells=frozenset({"Z/9"})
        )
        assert len(results) == 1
        assert "Z/9" in results[0].issues[0].message

    def test_null_ome_attrs_does_not_crash_is_hcs_plate_check(self, monkeypatch):
        # zarr.json attrs with an explicit "ome": null (rather than a dict)
        # must not crash the "plate" in raw_ome_attrs check with a
        # TypeError — it's simply not an HCS plate.
        zarr_group_mock = MagicMock()
        zarr_group_mock.attrs = {"ome": None}
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: zarr_group_mock,
        )
        results = validate_zarr_node("/fake/node", spec_version="0.2", field_limit=1)
        assert len(results) == 1
        assert "only apply to HCS plates" in results[0].issues[0].message

    def test_non_dict_plate_value_does_not_crash(self, monkeypatch):
        # ome.plate present but not a dict (malformed) must not crash
        # _validate_hcs_plate's wells lookup.
        zarr_group_mock = MagicMock()
        zarr_group_mock.attrs = {"ome": {"plate": "not-a-dict", "version": "0.5"}}
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: zarr_group_mock,
        )
        results = validate_zarr_node("/fake/plate", spec_version="0.2")
        assert len(results) >= 1

    def test_non_dict_plate_value_does_not_take_hcs_fast_path(self, monkeypatch):
        # is_hcs_plate must require ome.plate to actually be a dict — a
        # malformed non-dict "plate" value previously still took the HCS
        # fast path (via _as_dict coercing it to {}), which silently
        # produced a *passing* empty-plate result instead of surfacing the
        # malformed metadata as a structural validation failure.
        zarr_group_mock = MagicMock()
        zarr_group_mock.attrs = {"ome": {"plate": "not-a-dict", "version": "0.5"}}
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: zarr_group_mock,
        )
        results = validate_zarr_node("/fake/plate", spec_version="0.2")
        assert all(not r.passed for r in results)


# ---------------------------------------------------------------------------
# _check_plate_schema — OME-NGFF plate metadata schema checks
# ---------------------------------------------------------------------------


def _plate(*, rows=None, columns=None, acquisitions=None, wells=None):
    """Build a minimal raw_plate_attrs dict for _check_plate_schema tests."""
    attrs = {}
    if rows is not None:
        attrs["rows"] = [{"name": r} for r in rows]
    if columns is not None:
        attrs["columns"] = [{"name": c} for c in columns]
    if acquisitions is not None:
        attrs["acquisitions"] = acquisitions
    if wells is not None:
        attrs["wells"] = [{"path": p} for p in wells]
    return attrs


@pytest.mark.unit
class TestCheckPlateSchema:
    def test_valid_plate_no_issues(self):
        attrs = _plate(
            rows=["A", "B"],
            columns=["1", "2"],
            acquisitions=[{"id": 0}, {"id": 1}],
            wells=["A/1", "B/2"],
        )
        assert _check_plate_schema(attrs) == []

    # rows
    def test_row_non_alphanumeric_is_error(self):
        issues = _check_plate_schema(_plate(rows=["A!", "B"]))
        assert any(i.severity == Severity.ERROR and "A!" in i.message for i in issues)

    def test_row_names_unique(self):
        issues = _check_plate_schema(_plate(rows=["A", "A", "B"]))
        assert any(
            i.severity == Severity.ERROR and "Duplicate row" in i.message
            for i in issues
        )

    def test_valid_row_names_no_issue(self):
        assert _check_plate_schema(_plate(rows=["A", "B", "C"])) == []

    # columns
    def test_column_non_alphanumeric_is_error(self):
        issues = _check_plate_schema(_plate(columns=["1", "col-2"]))
        assert any(
            i.severity == Severity.ERROR and "col-2" in i.message for i in issues
        )

    def test_column_names_unique(self):
        issues = _check_plate_schema(_plate(columns=["1", "1"]))
        assert any(
            i.severity == Severity.ERROR and "Duplicate column" in i.message
            for i in issues
        )

    # fieldCount
    def test_field_count_zero_is_error(self):
        issues = _check_plate_schema({"fieldCount": 0})
        assert any(
            i.severity == Severity.ERROR and "fieldCount" in i.message for i in issues
        )

    def test_field_count_negative_is_error(self):
        issues = _check_plate_schema({"fieldCount": -1})
        assert any(
            i.severity == Severity.ERROR and "fieldCount" in i.message for i in issues
        )

    def test_field_count_non_integer_is_error(self):
        issues = _check_plate_schema({"fieldCount": "two"})
        assert any(
            i.severity == Severity.ERROR and "fieldCount" in i.message for i in issues
        )

    def test_field_count_positive_is_valid(self):
        assert _check_plate_schema({"fieldCount": 1}) == []
        assert _check_plate_schema({"fieldCount": 16}) == []

    def test_field_count_absent_is_valid(self):
        assert _check_plate_schema({}) == []

    # acquisitions
    def test_acquisition_negative_id_is_error(self):
        issues = _check_plate_schema(_plate(acquisitions=[{"id": -1}]))
        assert any(i.severity == Severity.ERROR and "-1" in i.message for i in issues)

    def test_acquisition_non_integer_id_is_error(self):
        issues = _check_plate_schema(_plate(acquisitions=[{"id": "bad"}]))
        assert any(i.severity == Severity.ERROR for i in issues)

    def test_acquisition_zero_id_is_valid(self):
        assert _check_plate_schema(_plate(acquisitions=[{"id": 0}])) == []

    def test_acquisition_missing_id_key_is_error(self):
        issues = _check_plate_schema(_plate(acquisitions=[{"name": "acq1"}]))
        assert any(
            i.severity == Severity.ERROR and "missing" in i.message for i in issues
        )

    def test_acquisition_null_id_is_error(self):
        issues = _check_plate_schema(_plate(acquisitions=[{"id": None}]))
        assert any(i.severity == Severity.ERROR for i in issues)

    def test_duplicate_acquisition_id_is_error(self):
        issues = _check_plate_schema(_plate(acquisitions=[{"id": 0}, {"id": 0}]))
        assert any(
            i.severity == Severity.ERROR and "Duplicate acquisition id" in i.message
            for i in issues
        )

    def test_duplicate_acquisition_id_only_first_is_valid(self):
        # Only the duplicate entry is flagged; the first occurrence is fine
        issues = _check_plate_schema(_plate(acquisitions=[{"id": 0}, {"id": 0}]))
        dup_errors = [i for i in issues if "Duplicate acquisition id" in i.message]
        assert len(dup_errors) == 1

    # well paths — syntax
    def test_well_path_no_slash_is_error(self):
        issues = _check_plate_schema(_plate(wells=["A1"]))
        assert any(i.severity == Severity.ERROR and "A1" in i.message for i in issues)

    def test_well_path_multiple_slashes_is_error(self):
        issues = _check_plate_schema(_plate(wells=["A/1/extra"]))
        assert any(
            i.severity == Severity.ERROR and "A/1/extra" in i.message for i in issues
        )

    # non-dict entries — malformed metadata must not crash, just be treated
    # as if the entry declared no name/id/path
    def test_non_dict_row_entry_does_not_crash(self):
        issues = _check_plate_schema({"rows": ["not-a-dict", {"name": "A"}]})
        assert any(
            i.severity == Severity.ERROR and i.loc == ("plate", "rows", "")
            for i in issues
        )

    def test_non_dict_column_entry_does_not_crash(self):
        issues = _check_plate_schema({"columns": [123, {"name": "1"}]})
        assert any(
            i.severity == Severity.ERROR and i.loc == ("plate", "columns", "")
            for i in issues
        )

    def test_non_dict_acquisition_entry_does_not_crash(self):
        issues = _check_plate_schema({"acquisitions": ["not-a-dict"]})
        assert any(
            i.severity == Severity.ERROR and "missing" in i.message for i in issues
        )

    def test_non_dict_well_entry_does_not_crash(self):
        issues = _check_plate_schema({"wells": [{"path": "A/1"}, "not-a-dict"]})
        assert any(
            i.severity == Severity.ERROR and "must contain exactly one '/'" in i.message
            for i in issues
        )

    def test_null_rows_columns_acquisitions_wells_do_not_crash(self):
        # rows/columns/acquisitions/wells: null (rather than a list) must not
        # crash the "for x in raw_plate_attrs.get(key, [])" loops — treated
        # as nothing declared, same as the key being absent.
        assert (
            _check_plate_schema(
                {"rows": None, "columns": None, "acquisitions": None, "wells": None}
            )
            == []
        )

    def test_non_string_well_path_does_not_crash(self):
        # A well entry with a non-string "path" (e.g. an int) must not crash
        # well_path.split("/") — it's coerced to str, same as row/column names.
        issues = _check_plate_schema({"wells": [{"path": 123}, {"path": "A/1"}]})
        assert any(
            i.severity == Severity.ERROR and "must contain exactly one '/'" in i.message
            for i in issues
        )

    # well paths — row/column cross-reference
    def test_well_path_row_not_in_rows_is_error(self):
        attrs = _plate(rows=["A", "B"], columns=["1"], wells=["X/1"])
        issues = _check_plate_schema(attrs)
        assert any(
            i.severity == Severity.ERROR and "X" in i.message and "rows" in i.message
            for i in issues
        )

    def test_well_path_column_not_in_columns_is_error(self):
        attrs = _plate(rows=["A"], columns=["1", "2"], wells=["A/9"])
        issues = _check_plate_schema(attrs)
        assert any(
            i.severity == Severity.ERROR and "9" in i.message and "columns" in i.message
            for i in issues
        )

    def test_well_path_valid_row_and_column_no_issue(self):
        attrs = _plate(rows=["A"], columns=["1"], wells=["A/1"])
        assert _check_plate_schema(attrs) == []

    def test_no_rows_declared_skips_row_cross_reference(self):
        # When rows is empty or absent, no cross-reference check is possible
        attrs = _plate(wells=["X/1"])
        issues = _check_plate_schema(attrs)
        row_errors = [i for i in issues if "rows" in i.message]
        assert row_errors == []

    def test_empty_plate_no_issues(self):
        assert _check_plate_schema({}) == []

    def test_duplicate_well_path_is_error(self):
        attrs = _plate(rows=["A"], columns=["1"], wells=["A/1", "A/1"])
        issues = _check_plate_schema(attrs)
        assert any(
            i.severity == Severity.ERROR and "Duplicate well path" in i.message
            for i in issues
        )

    def test_duplicate_well_path_only_first_is_validated(self):
        # The duplicate entry produces exactly one duplicate error; no extra
        # format or cross-reference errors for the repeated path
        attrs = _plate(rows=["A"], columns=["1"], wells=["A/1", "A/1"])
        issues = _check_plate_schema(attrs)
        dup_errors = [i for i in issues if "Duplicate well path" in i.message]
        assert len(dup_errors) == 1
        other_errors = [
            i
            for i in issues
            if i.severity == Severity.ERROR and "Duplicate well path" not in i.message
        ]
        assert other_errors == []


# ---------------------------------------------------------------------------
# _gather_plate_fields — well image path format and uniqueness checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGatherPlateFieldsImagePaths:
    def test_non_alphanumeric_image_path_is_error(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["field_0"], ["field_0"]),
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        assert any(
            i.severity == Severity.ERROR and "field_0" in i.message for i in issues
        )

    def test_alphanumeric_image_path_no_error(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0"], ["0"]),
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        path_errors = [
            i
            for i in issues
            if i.severity == Severity.ERROR and "alphanumeric" in i.message
        ]
        assert path_errors == []

    def test_duplicate_image_path_in_well_is_error(self, monkeypatch):
        mock = MagicMock()
        mock.attrs = {"ome": {"well": {"images": [{"path": "0"}, {"path": "0"}]}}}
        mock.keys.return_value = ["0"]
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock,
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        assert any(
            i.severity == Severity.ERROR and "Duplicate" in i.message for i in issues
        )

    def test_duplicate_image_path_only_added_once_to_field_paths(self, monkeypatch):
        mock = MagicMock()
        mock.attrs = {"ome": {"well": {"images": [{"path": "0"}, {"path": "0"}]}}}
        mock.keys.return_value = ["0"]
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock,
        )
        _, field_paths, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        assert field_paths.count("A/1/0") == 1

    def test_unique_image_paths_no_duplicate_error(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0", "1"], ["0", "1"]),
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        dup_errors = [i for i in issues if "Duplicate" in i.message]
        assert dup_errors == []

    def test_missing_field_does_not_produce_acquisition_id_error(self, monkeypatch):
        """Field absent from disk should not produce an acquisition ID error."""
        mock = MagicMock()
        # well declares image "0" with acquisition 99, but "0" is not on disk
        mock.attrs = {"ome": {"well": {"images": [{"path": "0", "acquisition": 99}]}}}
        mock.keys.return_value = []  # no groups on disk
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock,
        )
        plate_acquisitions = [{"id": 1}]  # acquisition 99 is not listed
        issues, field_paths, _, _ = _gather_plate_fields(
            "/p",
            [{"path": "A/1"}],
            plate_acquisitions=plate_acquisitions,
            spec_version="0.2",
        )
        assert any("no group on disk" in i.message for i in issues)
        aq_errors = [i for i in issues if "acquisitions" in i.message]
        assert aq_errors == []
        assert field_paths == []

    def test_orphan_alphanumeric_field_on_disk_is_error(self, monkeypatch):
        """A well sub-group on disk not declared in ome.well.images should be an error."""
        mock = MagicMock()
        mock.attrs = {"ome": {"well": {"images": [{"path": "0"}]}}}
        mock.keys.return_value = ["0", "1"]  # "1" is on disk but not declared
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock,
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        assert any("A/1/1" in i.message and "not declared" in i.message for i in issues)

    def test_labels_sub_group_is_not_flagged_as_orphan(self, monkeypatch):
        """'labels' is an OME-NGFF-specified non-field sub-group and must not be reported."""
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0"], ["0", "labels"]),
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        orphan_errors = [i for i in issues if "not declared" in i.message]
        assert orphan_errors == []

    def test_no_orphan_when_all_disk_fields_declared(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0", "1"], ["0", "1"]),
        )
        issues, _, _, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], spec_version="0.2"
        )
        orphan_errors = [i for i in issues if "not declared" in i.message]
        assert orphan_errors == []


# ---------------------------------------------------------------------------
# validate_zarr_node — HCS metadata extraction error loc
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateNodeHcsMetadataExtractionLoc:
    def test_loc_includes_field_path_on_metadata_extraction_failure(self):
        from dca_helpers.validation.validator import validate_zarr_node

        field_path = "A/1/0"
        node_path = "/fake/plate"

        zarr_group_mock = MagicMock()
        zarr_group_mock.attrs = {
            "ome": {"plate": {"wells": [{"path": "A/1"}]}, "version": "0.5"}
        }

        with (
            patch(
                "dca_helpers.validation.discovery.zarr.open_group",
                return_value=zarr_group_mock,
            ),
            patch(
                "dca_helpers.validation.validator._gather_plate_fields",
                return_value=([], [field_path], [field_path], {}),
            ),
            patch(
                "dca_helpers.validation.validator.open_ome_zarr",
                return_value=MagicMock(),
            ),
            patch(
                "dca_helpers.validation.validator._build_node_dict",
                side_effect=RuntimeError("parse error"),
            ),
        ):
            results = validate_zarr_node(node_path, spec_version="0.1")

        field_results = [
            r for r in results if r.node_path == f"{node_path}/{field_path}"
        ]
        assert len(field_results) == 1
        assert len(field_results[0].issues) == 1
        issue = field_results[0].issues[0]
        assert issue.loc == ("plate", field_path, "metadata_extraction")
        assert issue.severity == Severity.ERROR


# ---------------------------------------------------------------------------
# _validate_hcs_plate — single-pass uniformity SHOULD checks
# ---------------------------------------------------------------------------


def _make_node_dict(axes, chunk_shape):
    return {
        "axes": axes,
        "levels": [{"chunk_shape": chunk_shape}],
        "multiscale_level_count": 1,
        "array_shapes": [[1, 1, 32, 64, 64]],
    }


@pytest.mark.unit
class TestValidateHcsPlateUniformity:
    def _run(self, monkeypatch, field_node_dicts):
        """
        Run _validate_hcs_plate with N fields whose metadata is given by
        field_node_dicts (list of dicts returned by _build_node_dict).
        """
        field_paths = [f"A/1/{i}" for i in range(len(field_node_dicts))]
        node_path = "/fake/plate"

        field_zarr_mock = MagicMock()
        field_zarr_mock.attrs = {}

        nd_iter = iter(field_node_dicts)
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: field_zarr_mock,
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator.open_ome_zarr",
            lambda *a, **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator._gather_plate_fields",
            lambda *a, **kw: ([], field_paths, field_paths, {}),
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator._build_node_dict",
            lambda *a, **kw: next(nd_iter),
        )
        MockModel = MagicMock()
        return _validate_hcs_plate(node_path, {}, "0.5", "0.1", MockModel)

    def test_plate_result_is_first(self, monkeypatch):
        results = self._run(
            monkeypatch,
            [_make_node_dict(["t", "c", "z", "y", "x"], [1, 1, 64, 64, 64])],
        )
        assert results[0].node_path == "/fake/plate"

    def test_field_results_follow(self, monkeypatch):
        dicts = [_make_node_dict(["t", "c", "z", "y", "x"], [1, 1, 64, 64, 64])] * 3
        results = self._run(monkeypatch, dicts)
        assert len(results) == 4  # 1 plate + 3 fields

    def test_uniform_axes_no_warning(self, monkeypatch):
        dicts = [_make_node_dict(["t", "c", "z", "y", "x"], [1, 1, 64, 64, 64])] * 3
        results = self._run(monkeypatch, dicts)
        plate = results[0]
        axes_warnings = [i for i in plate.issues if "axes" in i.loc]
        assert axes_warnings == []

    def test_non_uniform_axes_warning(self, monkeypatch):
        dicts = [
            _make_node_dict(["t", "c", "z", "y", "x"], [1, 1, 64, 64, 64]),
            _make_node_dict(["c", "z", "y", "x"], [1, 64, 64, 64]),
        ]
        results = self._run(monkeypatch, dicts)
        plate = results[0]
        axes_issues = [i for i in plate.issues if "axes" in i.loc]
        assert len(axes_issues) == 1
        assert axes_issues[0].severity == Severity.WARNING
        assert axes_issues[0].loc == ("plate", "axes")

    def test_uniform_chunks_no_warning(self, monkeypatch):
        dicts = [_make_node_dict(["t", "c", "z", "y", "x"], [1, 1, 64, 64, 64])] * 3
        results = self._run(monkeypatch, dicts)
        plate = results[0]
        chunk_warnings = [i for i in plate.issues if "chunk_shape" in i.loc]
        assert chunk_warnings == []

    def test_non_uniform_chunks_warning(self, monkeypatch):
        dicts = [
            _make_node_dict(["t", "c", "z", "y", "x"], [1, 1, 64, 64, 64]),
            _make_node_dict(["t", "c", "z", "y", "x"], [1, 1, 32, 32, 32]),
        ]
        results = self._run(monkeypatch, dicts)
        plate = results[0]
        chunk_issues = [i for i in plate.issues if "chunk_shape" in i.loc]
        assert len(chunk_issues) == 1
        assert chunk_issues[0].severity == Severity.WARNING
        assert chunk_issues[0].loc == ("plate", "chunk_shape")


# ---------------------------------------------------------------------------
# HCS fast path — field results must be IMAGE, not HCS_PLATE
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHcsFastPathFieldNodeType:
    def _run(self, monkeypatch):
        node_path = "/fake/plate"
        plate_root_mock = MagicMock()
        plate_root_mock.attrs = {
            "ome": {"plate": {"wells": [{"path": "A/1"}]}, "version": "0.5"}
        }
        mocks_by_path = {
            node_path: plate_root_mock,
            f"{node_path}/A/1": _make_well_mock(None, ["0", "1"]),
            f"{node_path}/A/1/0": _make_field_mock(None),
            f"{node_path}/A/1/1": _make_field_mock(None),
        }
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda path, *a, **kw: mocks_by_path[path],
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator.open_ome_zarr",
            lambda *a, **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator._build_node_dict",
            lambda *a, **kw: _make_node_dict(
                ("t", "c", "z", "y", "x"), [1, 1, 1, 1, 1]
            ),
        )
        return validate_zarr_node(node_path, spec_version="0.2", ModelClass=MagicMock())

    def test_plate_root_is_hcs_plate(self, monkeypatch):
        results = self._run(monkeypatch)
        plate_root = next(r for r in results if r.parent_path is None)
        assert plate_root.node_type == ZarrNodeType.HCS_PLATE

    def test_field_results_are_image(self, monkeypatch):
        results = self._run(monkeypatch)
        fields = [r for r in results if r.parent_path is not None]
        assert len(fields) == 2
        assert all(r.node_type == ZarrNodeType.IMAGE for r in fields)


# ---------------------------------------------------------------------------
# dca attrs injection — IMAGE path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDcaInjectionImage:
    """validate_zarr_node passes dca attrs from zarr_group into node_dict."""

    def _run(self, monkeypatch, zarr_attrs, *, spec_version="0.2"):
        """Patch the full pipeline; return the node_dict seen by model_validate."""
        mock_group = MagicMock()
        mock_group.attrs = zarr_attrs
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock_group,
        )
        mock_ome = MagicMock()
        mock_ome.ome_zarr_version = "0.5"
        monkeypatch.setattr(
            "dca_helpers.validation.validator.open_ome_zarr",
            lambda *a, **kw: mock_ome,
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator.classify_group",
            lambda *a, **kw: ZarrNodeType.IMAGE,
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator._build_node_dict",
            lambda *a, **kw: {
                "axes": ["t", "c", "z", "y", "x"],
                "levels": [],
                "multiscale_level_count": 0,
                "array_shapes": [],
            },
        )
        captured = {}
        MockModel = MagicMock()
        MockModel.model_validate.side_effect = lambda d: captured.update(node_dict=d)
        validate_zarr_node(
            "/fake/path", spec_version=spec_version, ModelClass=MockModel
        )
        return captured["node_dict"]

    @_needs_schema
    def test_dca_present_in_attrs_is_injected(self, monkeypatch):
        payload = {"version": "0.2", "url": "https://example.com", "channels": []}
        nd = self._run(monkeypatch, {"ome": {}, "dca": payload})
        assert nd["dca"] == payload

    def test_dca_absent_from_attrs_injects_none(self, monkeypatch):
        nd = self._run(monkeypatch, {"ome": {}})
        assert nd["dca"] is None

    def test_spec_version_also_injected(self, monkeypatch):
        nd = self._run(monkeypatch, {"ome": {}}, spec_version="0.2")
        assert nd["spec_version"] == "0.2"


@pytest.mark.unit
class TestExperimentalMetadataEnforcement:
    """validate_zarr_node enforces the dca-block experimental floor (v0.2)."""

    def _run(self, monkeypatch, dca, *, spec_version="0.2"):
        """Validate one IMAGE store; ModelClass is mocked so only the
        experimental-metadata issues remain in the result."""
        mock_group = MagicMock()
        mock_group.attrs = {"ome": {}, "dca": dca} if dca is not None else {"ome": {}}
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock_group,
        )
        mock_ome = MagicMock()
        mock_ome.ome_zarr_version = "0.5"
        monkeypatch.setattr(
            "dca_helpers.validation.validator.open_ome_zarr",
            lambda *a, **kw: mock_ome,
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator.classify_group",
            lambda *a, **kw: ZarrNodeType.IMAGE,
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator._build_node_dict",
            lambda *a, **kw: {
                "axes": ["t", "c", "z", "y", "x"],
                "levels": [],
                "multiscale_level_count": 0,
                "array_shapes": [],
            },
        )
        return validate_zarr_node(
            "/fake/img", spec_version=spec_version, ModelClass=MagicMock()
        )

    @_needs_schema
    def test_channel_only_dca_fails_with_dca_prefixed_errors(self, monkeypatch):
        dca = {"version": "0.2", "url": "https://example.com", "channels": []}
        results = self._run(monkeypatch, dca, spec_version="0.2")
        (result,) = results
        assert not result.passed
        experimental = [i for i in result.issues if i.loc and i.loc[0] == "dca"]
        assert experimental
        assert all(i.severity == Severity.ERROR for i in experimental)

    def test_v0_1_does_not_enforce_experimental_metadata(self, monkeypatch):
        dca = {"version": "0.1", "channels": []}
        results = self._run(monkeypatch, dca, spec_version="0.1")
        (result,) = results
        assert not any(i.loc and i.loc[0] == "dca" for i in result.issues)


# ---------------------------------------------------------------------------
# node_type stamping — validate_zarr_node sets node_type on all results
# ---------------------------------------------------------------------------


def _patch_for_node_type(monkeypatch, zarr_node_type, node_dict=None):
    """Shared monkeypatching for node_type stamping tests."""
    mock_group = MagicMock()
    mock_group.attrs = {"ome": {}}
    monkeypatch.setattr(
        "dca_helpers.validation.discovery.zarr.open_group",
        lambda *a, **kw: mock_group,
    )
    mock_ome = MagicMock()
    mock_ome.ome_zarr_version = "0.5"
    monkeypatch.setattr(
        "dca_helpers.validation.validator.open_ome_zarr",
        lambda *a, **kw: mock_ome,
    )
    monkeypatch.setattr(
        "dca_helpers.validation.validator.classify_group",
        lambda *a, **kw: zarr_node_type,
    )
    if node_dict is not None:
        monkeypatch.setattr(
            "dca_helpers.validation.validator._build_node_dict",
            lambda *a, **kw: node_dict,
        )


@pytest.mark.unit
class TestNodeTypeStamping:
    """validate_zarr_node stamps node_type on every returned result."""

    def test_image_node_type(self, monkeypatch):
        _patch_for_node_type(
            monkeypatch,
            ZarrNodeType.IMAGE,
            node_dict={
                "axes": ["t", "c", "z", "y", "x"],
                "levels": [],
                "multiscale_level_count": 0,
                "array_shapes": [],
            },
        )
        MockModel = MagicMock()
        results = validate_zarr_node(
            "/fake/img", spec_version="0.1", ModelClass=MockModel
        )
        assert all(r.node_type == ZarrNodeType.IMAGE for r in results)

    def test_image_label_node_type(self, monkeypatch):
        _patch_for_node_type(monkeypatch, ZarrNodeType.IMAGE_LABEL)
        MockModel = MagicMock()
        results = validate_zarr_node(
            "/fake/label", spec_version="0.1", ModelClass=MockModel
        )
        assert all(r.node_type == ZarrNodeType.IMAGE_LABEL for r in results)


# ---------------------------------------------------------------------------
# IMAGE_LABEL branch — no label model registered (v0.1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImageLabelNodeNoModel:
    """
    validate_zarr_node returns passed=True with a WARNING when an IMAGE_LABEL node
    is encountered and no DCALabelStoreSpec is registered for that spec version.
    v0.1 has no label model, so it exercises this fallback path.
    """

    def _run(self, monkeypatch, spec_version="0.1"):
        mock_group = MagicMock()
        mock_group.attrs = {"ome": {}}
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock_group,
        )
        mock_ome = MagicMock()
        mock_ome.ome_zarr_version = "0.5"
        monkeypatch.setattr(
            "dca_helpers.validation.validator.open_ome_zarr",
            lambda *a, **kw: mock_ome,
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator.classify_group",
            lambda *a, **kw: ZarrNodeType.IMAGE_LABEL,
        )
        MockModel = MagicMock()
        return validate_zarr_node(
            "/fake/label", spec_version=spec_version, ModelClass=MockModel
        )

    def test_returns_single_result(self, monkeypatch):
        results = self._run(monkeypatch)
        assert len(results) == 1

    def test_passed_is_true(self, monkeypatch):
        results = self._run(monkeypatch)
        assert results[0].passed is True

    def test_warning_issued(self, monkeypatch):
        results = self._run(monkeypatch)
        warnings = [i for i in results[0].issues if i.severity == Severity.WARNING]
        assert warnings

    def test_warning_mentions_label_validation(self, monkeypatch):
        results = self._run(monkeypatch)
        assert any("label" in i.message.lower() for i in results[0].issues)

    def test_node_type_stamped(self, monkeypatch):
        results = self._run(monkeypatch)
        assert all(r.node_type == ZarrNodeType.IMAGE_LABEL for r in results)


# ---------------------------------------------------------------------------
# IMAGE_LABEL branch — label model registered (v0.2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImageLabelNodeWithModel:
    """
    validate_zarr_node dispatches to DCALabelStoreSpecV0_2 when an IMAGE_LABEL
    node is encountered with spec_version="0.2".
    """

    def _run(self, monkeypatch, *, node_dict=None, model_raises=None):
        mock_group = MagicMock()
        mock_group.attrs = {"ome": {}}
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock_group,
        )
        mock_ome = MagicMock()
        mock_ome.ome_zarr_version = "0.5"
        monkeypatch.setattr(
            "dca_helpers.validation.validator.open_ome_zarr",
            lambda *a, **kw: mock_ome,
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator.classify_group",
            lambda *a, **kw: ZarrNodeType.IMAGE_LABEL,
        )
        base_dict = node_dict or {
            "axes": ["t", "c", "z", "y", "x"],
            "levels": [],
            "multiscale_level_count": 0,
            "array_shapes": [],
        }
        monkeypatch.setattr(
            "dca_helpers.validation.validator._build_node_dict",
            lambda *a, **kw: base_dict,
        )
        LabelModel = MagicMock()
        if model_raises is not None:
            LabelModel.model_validate.side_effect = model_raises
        monkeypatch.setattr(
            "dca_helpers.validation.validator.get_label_model",
            lambda version: LabelModel,
        )
        return validate_zarr_node("/fake/label", spec_version="0.2"), LabelModel

    def test_label_model_called(self, monkeypatch):
        _, LabelModel = self._run(monkeypatch)
        LabelModel.model_validate.assert_called_once()

    def test_dca_injected_as_none(self, monkeypatch):
        _, LabelModel = self._run(monkeypatch)
        call_args = LabelModel.model_validate.call_args[0][0]
        assert call_args["dca"] is None

    def test_returns_passed_on_success(self, monkeypatch):
        results, _ = self._run(monkeypatch)
        assert len(results) == 1
        assert results[0].passed is True

    def test_no_warning_issued_on_success(self, monkeypatch):
        results, _ = self._run(monkeypatch)
        assert results[0].issues == []

    def test_node_type_stamped(self, monkeypatch):
        results, _ = self._run(monkeypatch)
        assert all(r.node_type == ZarrNodeType.IMAGE_LABEL for r in results)


# ---------------------------------------------------------------------------
# GEFF skip — validate_zarr_node returns skipped result for GEFF stores
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGeffSkip:
    """validate_zarr_node skips stores with a top-level 'geff' key in attrs."""

    def _run(self, monkeypatch, attrs):
        mock_group = MagicMock()
        mock_group.attrs = attrs
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock_group,
        )
        return validate_zarr_node("/fake/store.tracks.geff", spec_version="0.2")

    def test_returns_single_result(self, monkeypatch):
        results = self._run(monkeypatch, {"geff": {"geff_version": "0.1"}})
        assert len(results) == 1

    def test_skipped_is_true(self, monkeypatch):
        results = self._run(monkeypatch, {"geff": {"geff_version": "0.1"}})
        assert results[0].skipped is True

    def test_passed_is_true(self, monkeypatch):
        results = self._run(monkeypatch, {"geff": {"geff_version": "0.1"}})
        assert results[0].passed is True

    def test_node_type_is_geff(self, monkeypatch):
        results = self._run(monkeypatch, {"geff": {"geff_version": "0.1"}})
        assert results[0].node_type == ZarrNodeType.GEFF

    def test_no_issues(self, monkeypatch):
        results = self._run(monkeypatch, {"geff": {"geff_version": "0.1"}})
        assert results[0].issues == []

    def test_geff_with_extra_keys_still_skipped(self, monkeypatch):
        # GEFF stores may carry additional metadata alongside the geff key
        results = self._run(monkeypatch, {"geff": {}, "extra": {"foo": 1}})
        assert results[0].skipped is True

    def test_no_geff_key_not_skipped(self, monkeypatch):
        # A store without 'geff' must not be skipped (OME-Zarr path should proceed)
        mock_group = MagicMock()
        mock_group.attrs = {"ome": {}}
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock_group,
        )
        mock_ome = MagicMock()
        mock_ome.ome_zarr_version = "0.5"
        monkeypatch.setattr(
            "dca_helpers.validation.validator.open_ome_zarr",
            lambda *a, **kw: mock_ome,
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator.classify_group",
            lambda *a, **kw: ZarrNodeType.IMAGE,
        )
        monkeypatch.setattr(
            "dca_helpers.validation.validator._build_node_dict",
            lambda *a, **kw: {
                "axes": ["t", "c", "z", "y", "x"],
                "levels": [],
                "multiscale_level_count": 0,
                "array_shapes": [],
            },
        )
        MockModel = MagicMock()
        results = validate_zarr_node(
            "/fake/store.ome.zarr", spec_version="0.2", ModelClass=MockModel
        )
        assert all(not r.skipped for r in results)


# ---------------------------------------------------------------------------
# Level-0 scale extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLevel0AxisScales:
    """
    _level0_axis_scales() reads level 0's scale transform, keyed by axis name.

    Store parsing only, shared by every spec version: v0.2's variable-time-sampling
    rule is the only reader today, and v0.1 ignores the key. Covered directly because
    the model tests inject `axis_scales`, so a break in this plumbing would leave
    every scale None and silently disable the rule that reads it.
    """

    @staticmethod
    def _multiscale(scale=None, axis_names=("t", "c", "z", "y", "x")):
        """A multiscale carrying one level-0 dataset.

        The typed ``Dataset`` is used when there is a scale to read, so the happy
        path exercises the real OME model and would catch a renamed field. It
        rejects an empty transform list (OME-NGFF allows 1-2), so the
        nothing-to-read case uses a plain stand-in to reach the helper's own
        defensive branch.
        """
        from ome_zarr_models.common.coordinate_transformations import VectorScale
        from ome_zarr_models.v05.multiscales import Dataset

        if scale:
            dataset = Dataset(
                path="0",
                coordinateTransformations=[
                    VectorScale(type="scale", scale=list(scale))
                ],
            )
        else:
            dataset = SimpleNamespace(path="0", coordinateTransformations=[])
        return SimpleNamespace(
            axes=[SimpleNamespace(name=name) for name in axis_names],
            datasets=[dataset],
        )

    def test_reads_the_scale_per_axis(self):
        ms = self._multiscale(scale=[600.0, 1.0, 2.0, 0.65, 0.65])
        assert _level0_axis_scales(ms) == {
            "t": 600.0,
            "c": 1.0,
            "z": 2.0,
            "y": 0.65,
            "x": 0.65,
        }

    def test_absent_transform_yields_none_not_one(self):
        # None means "could not be read", which the v0.2 rule treats as no
        # violation; defaulting to 1.0 would silently satisfy a MUST.
        assert _level0_axis_scales(self._multiscale()) == {
            "t": None,
            "c": None,
            "z": None,
            "y": None,
            "x": None,
        }

    def test_length_mismatch_is_ignored(self):
        ms = self._multiscale(scale=[1.0, 2.0])
        assert _level0_axis_scales(ms)["t"] is None

    def test_no_datasets_yields_all_none(self):
        ms = self._multiscale(scale=[1.0, 1.0, 1.0, 1.0, 1.0])
        ms.datasets = []
        assert _level0_axis_scales(ms) == dict.fromkeys(["t", "c", "z", "y", "x"])
