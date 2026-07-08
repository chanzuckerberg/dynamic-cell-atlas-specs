"""
Unit tests for validation.validator helpers.

Integration tests against real Zarr stores are skipped when the test data
directory is not present.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from dca_helpers.validation.result import Severity
from dca_helpers.validation.validator import (
    _check_plate_schema,
    _extract_level,
    _gather_plate_fields,
    _get_compression,
    _validate_hcs_plate,
    validate_zarr_node,
)
from dca_helpers.validation.zarr_node import ZarrNodeType
from tests.validation.json_schema_oracle import (
    SchemaNotFoundError,
    _find_schema_path,
)


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
        issues, field_paths = _gather_plate_fields("/p", [{"path": "A/1"}])
        assert issues == []
        assert field_paths == ["A/1/0"]

    def test_multiple_wells(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0"], ["0"]),
        )
        issues, field_paths = _gather_plate_fields(
            "/p", [{"path": "A/1"}, {"path": "B/2"}]
        )
        assert issues == []
        assert field_paths == ["A/1/0", "B/2/0"]

    def test_missing_well_is_error(self, monkeypatch):
        def _raise(*a, **kw):
            raise Exception("not found")

        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _raise)
        issues, field_paths = _gather_plate_fields("/p", [{"path": "A/1"}])
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR
        assert issues[0].loc == ("plate", "wells", "A/1")
        assert field_paths == []

    def test_missing_field_is_error(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0", "missing"], ["0"]),
        )
        issues, field_paths = _gather_plate_fields("/p", [{"path": "A/1"}])
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR
        assert issues[0].loc == ("plate", "wells", "A/1", "missing")

    def test_missing_field_not_in_field_paths(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0", "missing"], ["0"]),
        )
        _, field_paths = _gather_plate_fields("/p", [{"path": "A/1"}])
        assert "A/1/0" in field_paths
        assert "A/1/missing" not in field_paths

    def test_multiple_missing_wells_each_produce_error(self, monkeypatch):
        def _raise(*a, **kw):
            raise Exception("not found")

        monkeypatch.setattr("dca_helpers.validation.discovery.zarr.open_group", _raise)
        issues, _ = _gather_plate_fields("/p", [{"path": "A/1"}, {"path": "B/2"}])
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
        issues, _ = _gather_plate_fields("/p", [{"path": "A/1"}], None)
        assert issues == []

    def test_valid_acquisition_id_no_issue(self, monkeypatch):
        mock = MagicMock()
        mock.attrs = {"ome": {"well": {"images": [{"path": "0", "acquisition": 1}]}}}
        mock.keys.return_value = ["0"]
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: mock,
        )
        issues, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], [{"id": 1}, {"id": 2}]
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
        issues, _ = _gather_plate_fields(
            "/p", [{"path": "A/1"}], [{"id": 1}, {"id": 2}]
        )
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR
        assert issues[0].loc == ("plate", "wells", "A/1", "0", "acquisition")
        assert "99" in issues[0].message

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
        issues, _ = _gather_plate_fields("/p", [{"path": "A/1"}], [{"id": None}])
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
        issues, _ = _gather_plate_fields("/p", [{"path": "A/1"}], [{"id": 1}])
        assert issues == []


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
        issues, _ = _gather_plate_fields("/p", [{"path": "A/1"}])
        assert any(
            i.severity == Severity.ERROR and "field_0" in i.message for i in issues
        )

    def test_alphanumeric_image_path_no_error(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0"], ["0"]),
        )
        issues, _ = _gather_plate_fields("/p", [{"path": "A/1"}])
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
        issues, _ = _gather_plate_fields("/p", [{"path": "A/1"}])
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
        _, field_paths = _gather_plate_fields("/p", [{"path": "A/1"}])
        assert field_paths.count("A/1/0") == 1

    def test_unique_image_paths_no_duplicate_error(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0", "1"], ["0", "1"]),
        )
        issues, _ = _gather_plate_fields("/p", [{"path": "A/1"}])
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
        issues, field_paths = _gather_plate_fields(
            "/p", [{"path": "A/1"}], plate_acquisitions=plate_acquisitions
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
        issues, _ = _gather_plate_fields("/p", [{"path": "A/1"}])
        assert any("A/1/1" in i.message and "not declared" in i.message for i in issues)

    def test_labels_sub_group_is_not_flagged_as_orphan(self, monkeypatch):
        """'labels' is an OME-NGFF-specified non-field sub-group and must not be reported."""
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0"], ["0", "labels"]),
        )
        issues, _ = _gather_plate_fields("/p", [{"path": "A/1"}])
        orphan_errors = [i for i in issues if "not declared" in i.message]
        assert orphan_errors == []

    def test_no_orphan_when_all_disk_fields_declared(self, monkeypatch):
        monkeypatch.setattr(
            "dca_helpers.validation.discovery.zarr.open_group",
            lambda *a, **kw: make_well_group(["0", "1"], ["0", "1"]),
        )
        issues, _ = _gather_plate_fields("/p", [{"path": "A/1"}])
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
                return_value=([], [field_path]),
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
            lambda *a, **kw: ([], field_paths),
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
# Integration tests — require SSBD data on disk
# ---------------------------------------------------------------------------

SSBD_DATA = "/tmp/downloads/ssbd"
have_ssbd = os.path.isdir(SSBD_DATA)


@pytest.mark.integration
@pytest.mark.skipif(not have_ssbd, reason=f"SSBD data not found at {SSBD_DATA}")
class TestIntegrationSSBD:
    def test_validate_returns_results(self):
        from dca_helpers.validation.runner import validate

        results = validate(SSBD_DATA, spec_version="0.1")
        assert len(results) > 0

    def test_all_results_have_node_path(self):
        from dca_helpers.validation.runner import validate

        results = validate(SSBD_DATA, spec_version="0.1")
        for r in results:
            assert r.node_path

    def test_no_result_raises_exception(self):
        from dca_helpers.validation.runner import validate

        results = validate(SSBD_DATA, spec_version="0.1")
        # All results should have passed or failed cleanly — no unhandled exceptions
        for r in results:
            assert r.passed is not None


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
