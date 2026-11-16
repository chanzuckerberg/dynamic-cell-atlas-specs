"""Native, metadata-only acceptance tests for persisted HCS chunk rollup.

The store fixture contains only zarr.json metadata. Reports cross a real Parquet
serialization boundary; no comparison depends on in-memory validator objects.
"""

from __future__ import annotations

import copy
import itertools
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dca_helpers.validation.hcs_facts import HCSManifest
from dca_helpers.validation.reporter import build_report
from dca_helpers.validation.result import Severity
from dca_helpers.validation.rollup import create_hcs_manifest, rollup_reports
from dca_helpers.validation.runner import validate

FIXTURE = Path(__file__).parent.parent / "fixtures/v0_2/valid_plate_multiwell.ome.zarr"
SNAPSHOT = "revision-1"
FIELDS = ("A/1/0", "A/1/1", "B/1/0", "B/1/1")
pytestmark = pytest.mark.functional


@pytest.fixture
def plate(tmp_path):
    return Path(shutil.copytree(FIXTURE, tmp_path / "plate.ome.zarr"))


def _edit_json(path, edit):
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def _assignment(perturbation_id="drug-a", **overrides):
    return {
        "control_class": "treatment",
        "definition": {
            "perturbation_id": perturbation_id,
            "modality": "compound",
            "name": perturbation_id,
            "chemical": {"smiles": "CCO"},
        },
        **overrides,
    }


def _set_assignment(plate, field, assignment):
    def edit(data):
        data["attributes"]["dca"]["perturbation"] = copy.deepcopy(assignment)

    _edit_json(plate / field / "zarr.json", edit)


def _set_well_assignment(plate, well, assignment):
    def edit(data):
        data["attributes"].setdefault("dca", {})["well_perturbation"] = {
            "plate": "plate-1",
            "well": well,
            **copy.deepcopy(assignment),
        }

    _edit_json(plate / well / "zarr.json", edit)


def _validate(plate, **kwargs):
    return validate(
        str(plate),
        spec_version="0.2",
        snapshot_id=SNAPSHOT,
        show_progress=False,
        **kwargs,
    )


def _report(plate, **kwargs):
    return build_report(_validate(plate, **kwargs))


def _manifest(plate):
    return create_hcs_manifest(str(plate), snapshot_id=SNAPSHOT, spec_version="0.2")


def _root(run):
    roots = [r for r in run if r.parent_path is None]
    assert len(roots) == 1
    return roots[0]


def _issues(run):
    return [issue for result in run for issue in result.issues]


def _consistency(run):
    return sorted(
        (tuple(str(part) for part in issue.loc), issue.severity.value, issue.message)
        for issue in _root(run).issues
        if issue.loc in (("plate", "axes"), ("plate", "chunk_shape"))
        or (
            "dca" in issue.loc
            and issue.loc[-1] in {"perturbation", "well_perturbation"}
        )
    )


def _outcomes(run):
    """Compare semantic outcomes, excluding run timing and chunk coverage."""
    return {
        result.node_path: (
            result.passed,
            sorted(
                (
                    tuple(str(part) for part in issue.loc),
                    issue.severity.value,
                    issue.message,
                )
                for issue in result.issues
            ),
        )
        for result in run
    }


def _rewrite_report(report, edit, *, drop_columns=()):
    table = pq.read_table(pa.BufferReader(report))
    rows = table.to_pylist()
    edit(rows)
    table = pa.Table.from_pylist(rows, schema=table.schema)
    if drop_columns:
        table = table.drop_columns(list(drop_columns))
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def _field_rows(rows):
    return [row for row in rows if row["record_type"] == "node" and row["parent_path"]]


def _plate_row(rows):
    return next(
        row
        for row in rows
        if row["record_type"] == "node" and row["parent_path"] is None
    )


@pytest.mark.parametrize(
    ("first_assignment", "second_assignment"),
    [
        (_assignment(), _assignment("drug-b")),
        (
            _assignment(),
            _assignment(
                definition={
                    "perturbation_id": "drug-a",
                    "modality": "compound",
                    "name": "drug-a",
                    "chemical": {"smiles": "CCN"},
                }
            ),
        ),
        (
            _assignment(concentration={"value": 1.0, "unit": "uM"}),
            _assignment(concentration={"value": 2.0, "unit": "uM"}),
        ),
        (
            _assignment(
                cotreatment={"perturbation_id": "co-a", "modality": "physical"}
            ),
            _assignment(
                cotreatment={"perturbation_id": "co-b", "modality": "physical"}
            ),
        ),
        (_assignment(timepoint_primary_h=0.0), _assignment(timepoint_primary_h=24.0)),
        (
            _assignment(timepoint_secondary_h=12.0),
            _assignment(timepoint_secondary_h=48.0),
        ),
    ],
    ids=[
        "definition",
        "same-id-different-definition",
        "dose",
        "cotreatment",
        "primary-timepoint",
        "secondary-timepoint",
    ],
)
def test_split_well_conflicts_reappear_after_rollup(
    plate, first_assignment, second_assignment
):
    """Each chunk is locally consistent; the union must catch the real conflict."""
    _set_assignment(plate, "A/1/0", first_assignment)
    _set_assignment(plate, "A/1/1", second_assignment)
    manifest = _manifest(plate)
    chunks = [_validate(plate, field_offset=i, field_limit=1) for i in range(2)]
    assert all(_root(chunk).passed for chunk in chunks)
    assert all(
        result.passed for chunk in chunks for result in chunk if result.parent_path
    )
    assert all(not _consistency(chunk) for chunk in chunks)

    whole = _validate(plate)
    rolled = rollup_reports([build_report(chunk) for chunk in chunks], manifest)
    assert not _root(rolled).passed
    assert _consistency(rolled) == _consistency(whole)
    assert any(
        issue.loc == ("plate", "wells", "A/1", "dca", "perturbation")
        and issue.severity == Severity.ERROR
        for issue in _root(rolled).issues
    )
    assert not rolled.summary.is_partial


def test_distinct_wells_and_optional_well_rows_are_independent(plate):
    for well, name in (("A/1", "drug-a"), ("B/1", "drug-b")):
        for index in range(2):
            _set_assignment(plate, f"{well}/{index}", _assignment(name))
    # A well row is optional; B's valid row is an independent comparison.
    _set_well_assignment(plate, "B/1", _assignment("drug-b"))
    manifest = _manifest(plate)
    reports = [_report(plate, wells=well) for well in ("A/1", "B/1")]
    rolled = rollup_reports(reports, manifest)
    assert rolled.summary.stores_failed == 0
    assert not rolled.summary.is_partial
    assert not _consistency(rolled)
    assert _outcomes(rolled) == _outcomes(_validate(plate))


def test_well_row_disagreement_is_recomputed_from_complete_field_consensus(plate):
    for field in ("A/1/0", "A/1/1"):
        _set_assignment(plate, field, _assignment())
    _set_well_assignment(plate, "A/1", _assignment("drug-b"))
    manifest = _manifest(plate)
    reports = [_report(plate, field_offset=i, field_limit=1) for i in range(2)]
    rolled = rollup_reports(reports, manifest)
    assert _consistency(rolled) == _consistency(_validate(plate))
    assert len(_consistency(rolled)) == 1
    assert _consistency(rolled)[0][0][-1] == "well_perturbation"
    assert not _root(rolled).passed


def test_omitted_and_explicit_null_optional_values_are_equivalent(plate):
    ordinary = _assignment()
    nullable = _assignment(
        concentration=None, cotreatment=None, timepoint_primary_h=None
    )
    nullable["definition"].update(description=None)
    nullable["definition"]["chemical"]["pubchem_cid"] = None
    _set_assignment(plate, "A/1/0", ordinary)
    _set_assignment(plate, "A/1/1", nullable)
    _set_well_assignment(plate, "A/1", nullable)
    # Entirely absent and explicitly null perturbation blocks also agree.
    _set_assignment(plate, "B/1/1", None)
    manifest = _manifest(plate)
    rolled = rollup_reports(
        [_report(plate, field_offset=i, field_limit=1) for i in range(2)], manifest
    )
    assert rolled.summary.stores_failed == 0
    assert not _consistency(rolled)


def test_arbitrary_partitions_permutations_overlaps_and_retries_are_idempotent(plate):
    _set_assignment(plate, "A/1/0", _assignment())
    _set_assignment(plate, "A/1/1", _assignment("drug-b"))
    manifest = _manifest(plate)
    reports = [
        _report(plate, wells=well, field_offset=index, field_limit=1)
        for well in ("A/1", "B/1")
        for index in range(2)
    ]
    expected = _outcomes(_validate(plate))
    # Every ordering of four independently persisted chunks yields the same union.
    for order in itertools.permutations(reports):
        rolled = rollup_reports(iter(order), manifest)
        assert _outcomes(rolled) == expected
        assert not rolled.summary.is_partial
        assert rolled.summary.stores_validated == 5
        assert rolled.summary.level_zarr_jsons_validated == 12
    # Retries and overlap with an entire-well chunk must not inflate totals/issues.
    rolled = rollup_reports(
        reports + reports[:2] + [_report(plate, wells="A/1")], manifest
    )
    assert _outcomes(rolled) == expected
    assert rolled.summary.stores_validated == 5
    assert rolled.summary.level_zarr_jsons_validated == 12


def test_rollup_accepts_bytes_paths_and_manifest_json_without_opening_stores(
    plate, tmp_path, monkeypatch
):
    import zarr

    from dca_helpers.validation import discovery, runner, validator

    def no_pixels(*args, **kwargs):
        raise AssertionError("metadata validation must not read pixel chunks")

    monkeypatch.setattr(zarr.Array, "__getitem__", no_pixels)
    manifest = _manifest(plate)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json())
    restored = HCSManifest.model_validate_json(manifest_path.read_text())
    assert restored == manifest
    reports = [
        _report(plate, wells=well, field_offset=index, field_limit=1)
        for well in ("A/1", "B/1")
        for index in range(2)
    ]
    expected = _outcomes(_validate(plate))
    paths = [tmp_path / f"chunk-{index}.parquet" for index in range(2)]
    for path, report in zip(paths, reports):
        path.write_bytes(report)

    def no_store_open(*args, **kwargs):
        raise AssertionError("rollup must use persisted reports and manifest only")

    for module in (discovery, runner, validator):
        monkeypatch.setattr(module, "open_zarr_group", no_store_open)
    monkeypatch.setattr(zarr, "open_group", no_store_open)
    monkeypatch.setattr(validator, "open_ome_zarr", no_store_open)
    shutil.rmtree(plate)
    rolled = rollup_reports([paths[0], str(paths[1]), *reports[2:]], restored)
    assert _outcomes(rolled) == expected
    assert not rolled.summary.is_partial
    # Rollup output itself remains a normal persistable report.
    assert pq.read_table(pa.BufferReader(build_report(rolled))).num_rows == 6


def _drop_time_axis(plate, field):
    def edit_group(data):
        multiscale = data["attributes"]["ome"]["multiscales"][0]
        multiscale["axes"].pop(0)
        for dataset in multiscale["datasets"]:
            for transform in dataset["coordinateTransformations"]:
                transform["scale"].pop(0)

    _edit_json(plate / field / "zarr.json", edit_group)
    for level in range(3):

        def edit_array(data):
            data["shape"].pop(0)
            data["dimension_names"].pop(0)
            data["chunk_grid"]["configuration"]["chunk_shape"].pop(0)

        _edit_json(plate / field / str(level) / "zarr.json", edit_array)


@pytest.mark.parametrize("change", ["axes", "chunk_shape"])
def test_plate_uniformity_warnings_span_chunk_boundaries(plate, change):
    if change == "axes":
        _drop_time_axis(plate, "A/1/1")
        _drop_time_axis(plate, "B/1/1")
    else:
        for field in ("A/1/1", "B/1/1"):
            _edit_json(
                plate / field / "0/zarr.json",
                lambda data: data["chunk_grid"]["configuration"][
                    "chunk_shape"
                ].__setitem__(2, 64),
            )
    manifest = _manifest(plate)
    chunks = [_validate(plate, field_offset=i, field_limit=1) for i in range(2)]
    assert all(not _consistency(chunk) for chunk in chunks)
    rolled = rollup_reports([build_report(chunk) for chunk in chunks], manifest)
    assert _consistency(rolled) == _consistency(_validate(plate))
    assert any(
        issue.loc == ("plate", change) and issue.severity == Severity.WARNING
        for issue in _root(rolled).issues
    )
    assert _root(rolled).passed  # Uniformity findings themselves remain SHOULDs.


@pytest.mark.parametrize(
    "selection", [{"field_limit": 1}, {"wells": "A/1"}, {"field_limit": 0}]
)
def test_incomplete_field_coverage_cannot_pass_as_complete(plate, selection):
    manifest = _manifest(plate)
    rolled = rollup_reports([_report(plate, **selection)], manifest)
    assert rolled.summary.is_partial
    assert not _root(rolled).passed
    assert any(issue.loc == ("rollup", "coverage") for issue in _root(rolled).errors)
    assert _root(rolled).subset_coverage.fields_total == 4
    assert _root(rolled).subset_coverage.fields_selected < 4


def test_missing_declared_field_stays_in_manifest_and_blocks_completion(plate):
    shutil.rmtree(plate / "A/1/1")
    manifest = _manifest(plate)
    assert "A/1/1" in manifest.plate.wells["A/1"].expected_fields
    rolled = rollup_reports([_report(plate)], manifest)
    assert rolled.summary.is_partial
    assert not _root(rolled).passed
    assert _root(rolled).subset_coverage.fields_total == 4
    assert _root(rolled).subset_coverage.fields_selected == 3
    assert any("A/1/1" in issue.message for issue in _issues(rolled))


def test_failed_field_extraction_is_incomplete_not_an_absent_perturbation(plate):
    for field in ("A/1/0", "A/1/1"):
        _set_assignment(plate, field, _assignment())
    # Valid group inventory but an unparseable image metadata contract.
    _edit_json(
        plate / "A/1/1/zarr.json",
        lambda data: data["attributes"]["ome"].update(multiscales=[]),
    )
    manifest = _manifest(plate)
    rolled = rollup_reports([_report(plate)], manifest)
    assert rolled.summary.is_partial
    assert not _root(rolled).passed
    assert any("metadata_extraction" in issue.loc for issue in _issues(rolled))
    assert not _consistency(rolled)


@pytest.fixture
def complete_report(plate):
    return plate, _manifest(plate), _report(plate)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("snapshot_id", "other-revision"),
        ("snapshot_id", None),
        ("input_path", "/other/plate.ome.zarr"),
        ("input_path", None),
        ("dca_spec_version", "0.1"),
        ("dca_validator_version", "999.0"),
        ("reporting_format_version", "0.0.0"),
    ],
)
def test_incompatible_report_provenance_is_rejected(complete_report, column, value):
    _, manifest, report = complete_report

    def edit(rows):
        for row in rows:
            row[column] = value

    with pytest.raises(ValueError):
        rollup_reports([_rewrite_report(report, edit)], manifest)


def test_inconsistent_provenance_within_one_report_is_rejected(complete_report):
    _, manifest, report = complete_report

    def edit(rows):
        _field_rows(rows)[0]["snapshot_id"] = "other-revision"

    with pytest.raises(ValueError):
        rollup_reports([_rewrite_report(report, edit)], manifest)


@pytest.mark.parametrize(
    "facts",
    [None, "not json", '{"format_version":"99.0","kind":"field","read_success":true}'],
)
def test_missing_malformed_or_incompatible_field_facts_are_rejected(
    complete_report, facts
):
    _, manifest, report = complete_report

    def edit(rows):
        _field_rows(rows)[0]["hcs_facts"] = facts

    with pytest.raises(ValueError):
        rollup_reports([_rewrite_report(report, edit)], manifest)


def test_legacy_report_without_facts_column_is_rejected(complete_report):
    _, manifest, report = complete_report
    legacy = _rewrite_report(report, lambda rows: None, drop_columns=("hcs_facts",))
    with pytest.raises(ValueError):
        rollup_reports([legacy], manifest)


@pytest.mark.parametrize(
    "conflict", ["field-facts", "field-outcome", "well-metadata", "plate-metadata"]
)
def test_conflicting_duplicates_or_changed_metadata_are_rejected(
    complete_report, conflict
):
    _, manifest, report = complete_report

    def edit(rows):
        if conflict == "field-facts":
            row = _field_rows(rows)[0]
            facts = json.loads(row["hcs_facts"])
            facts["perturbation"] = _assignment("changed")
            row["hcs_facts"] = json.dumps(facts)
        elif conflict == "field-outcome":
            row = _field_rows(rows)[0]
            row["passed"] = False
            row["issues"].append(
                {"loc": ["example"], "severity": "error", "message": "Changed outcome"}
            )
        else:
            row = _plate_row(rows)
            facts = json.loads(row["hcs_facts"])
            if conflict == "well-metadata":
                facts["wells"]["A/1"]["metadata_digest"] = "0" * 64
            else:
                facts["metadata_digest"] = "0" * 64
            row["hcs_facts"] = json.dumps(facts)

    with pytest.raises(ValueError):
        rollup_reports([report, _rewrite_report(report, edit)], manifest)


@pytest.mark.parametrize("location", ["plate", "well"])
def test_metadata_drift_between_manifest_and_real_chunk_is_rejected(plate, location):
    manifest = _manifest(plate)
    if location == "plate":
        _edit_json(
            plate / "zarr.json",
            lambda data: data["attributes"]["ome"]["plate"].update(name="changed"),
        )
    else:
        _set_well_assignment(plate, "A/1", {"control_class": "empty"})
    with pytest.raises(ValueError):
        rollup_reports([_report(plate)], manifest)


@pytest.mark.parametrize(
    "bad_images", [None, "not-a-list", [{"path": "0"}, {"path": "0"}], [{"path": ""}]]
)
def test_manifest_rejects_ambiguous_or_malformed_inventory(plate, bad_images):
    _edit_json(
        plate / "A/1/zarr.json",
        lambda data: data["attributes"]["ome"]["well"].update(images=bad_images),
    )
    with pytest.raises(ValueError):
        _manifest(plate)


def test_manifest_rejects_unreadable_well_inventory(plate):
    shutil.rmtree(plate / "A/1")
    with pytest.raises(ValueError):
        _manifest(plate)


@pytest.mark.parametrize("snapshot_id", ["", "   "])
def test_manifest_requires_explicit_nonempty_snapshot(plate, snapshot_id):
    with pytest.raises(ValueError):
        create_hcs_manifest(str(plate), snapshot_id=snapshot_id, spec_version="0.2")


def test_ordinary_validation_does_not_require_snapshot_id(plate):
    run = validate(str(plate), spec_version="0.2", show_progress=False)
    assert run.snapshot_id is None
    assert run.summary.stores_failed == 0
    assert not run.summary.is_partial
    assert len(run.results) == 5


def test_empty_report_collection_is_rejected(plate):
    with pytest.raises(ValueError):
        rollup_reports([], _manifest(plate))


def test_unreadable_parquet_report_is_rejected(plate):
    with pytest.raises(ValueError):
        rollup_reports([b"not a parquet report"], _manifest(plate))


def test_manifest_generation_never_deep_validates_fields(plate, monkeypatch):
    from dca_helpers.validation import validator

    def no_field_validation(*args, **kwargs):
        raise AssertionError("creating the inventory must not deep-validate fields")

    monkeypatch.setattr(validator, "_validate_field", no_field_validation)
    manifest = _manifest(plate)
    assert {
        field
        for well in manifest.plate.wells.values()
        for field in well.expected_fields
    } == set(FIELDS)


@pytest.mark.parametrize("fields_in_well", [0, 2], ids=["empty-well", "nonempty-well"])
def test_malformed_well_assignment_schema_error_survives_rollup(plate, fields_in_well):
    """A schema issue sharing a consistency-check location must never be dropped."""

    def edit_well(data):
        if not fields_in_well:
            data["attributes"]["ome"]["well"]["images"] = []
        data["attributes"]["dca"] = {"well_perturbation": "malformed"}

    _edit_json(plate / "A/1/zarr.json", edit_well)
    if not fields_in_well:
        for index in range(2):
            shutil.rmtree(plate / "A/1" / str(index))
    manifest = _manifest(plate)
    original = _validate(plate)
    schema_errors = [
        issue
        for issue in _root(original).errors
        if issue.loc == ("plate", "wells", "A/1", "dca", "well_perturbation")
        and issue.code is None
    ]
    assert schema_errors
    assert not _root(original).passed
    rolled = rollup_reports([build_report(original)], manifest)
    assert not _root(rolled).passed
    assert rolled.summary.stores_failed > 0
    assert not rolled.summary.is_partial
    assert _outcomes(rolled) == _outcomes(original)
    assert all(issue in _root(rolled).errors for issue in schema_errors)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("kind", "plate"),
        ("format_version", "99.0"),
        ("read_success", "true"),
        ("axes", "tczyx"),
        ("unexpected", "unsupported fact"),
    ],
)
def test_unsupported_or_wrongly_typed_field_facts_are_rejected(
    complete_report, key, value
):
    _, manifest, report = complete_report

    def edit(rows):
        row = _field_rows(rows)[0]
        facts = json.loads(row["hcs_facts"])
        facts[key] = value
        row["hcs_facts"] = json.dumps(facts)

    with pytest.raises(ValueError):
        rollup_reports([_rewrite_report(report, edit)], manifest)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("passed", None),
        ("skipped", None),
        ("node_type", None),
        ("node_type", "UNRECOGNIZED_TYPE"),
        ("node_path", None),
        ("issues", None),
        ("issues", [None]),
        ("issues", [{"loc": None, "severity": "warning", "message": "broken"}]),
        ("issues", [{"loc": ["example"], "severity": "unknown", "message": "broken"}]),
    ],
)
def test_malformed_node_rows_are_rejected(complete_report, column, value):
    _, manifest, report = complete_report

    def edit(rows):
        _field_rows(rows)[0][column] = value

    with pytest.raises(ValueError):
        rollup_reports([_rewrite_report(report, edit)], manifest)


@pytest.mark.parametrize(
    "shape",
    ["missing-summary", "duplicate-summary", "missing-root", "unknown-record-type"],
)
def test_malformed_report_structure_is_rejected(complete_report, shape):
    _, manifest, report = complete_report

    def edit(rows):
        if shape == "missing-summary":
            rows[:] = [
                row for row in rows if row["record_type"] != "validation_run_summary"
            ]
        elif shape == "duplicate-summary":
            rows.append(copy.deepcopy(rows[0]))
        elif shape == "missing-root":
            rows.remove(_plate_row(rows))
        else:
            _field_rows(rows)[0]["record_type"] = "unknown"

    with pytest.raises(ValueError):
        rollup_reports([_rewrite_report(report, edit)], manifest)


def test_chunk_local_well_row_finding_is_replaced_by_global_field_conflict(plate):
    _set_assignment(plate, "A/1/0", _assignment())
    _set_assignment(plate, "A/1/1", _assignment("drug-b"))
    _set_well_assignment(plate, "A/1", _assignment())
    manifest = _manifest(plate)
    chunks = [_validate(plate, field_offset=index, field_limit=1) for index in range(2)]
    assert _root(chunks[0]).passed
    assert any(
        issue.code == "hcs.well_perturbation_agreement"
        for issue in _root(chunks[1]).errors
    )
    rolled = rollup_reports([build_report(chunk) for chunk in chunks], manifest)
    assert _outcomes(rolled) == _outcomes(_validate(plate))
    assert [issue.code for issue in _root(rolled).errors] == [
        "hcs.perturbation_agreement"
    ]


def test_orphan_field_structural_error_is_preserved(plate):
    shutil.copytree(plate / "A/1/0", plate / "A/1/9")
    manifest = _manifest(plate)
    whole = _validate(plate)
    rolled = rollup_reports(
        [_report(plate, field_offset=index, field_limit=1) for index in range(2)],
        manifest,
    )
    assert not _root(rolled).passed
    assert not rolled.summary.is_partial
    assert any(
        "not declared in ome.well.images" in issue.message
        for issue in _root(rolled).errors
    )
    assert _outcomes(rolled) == _outcomes(whole)


@pytest.mark.parametrize(
    "wells", [None, "not-a-list", [{"path": "A/1"}, {"path": "A/1"}], [{"path": ""}]]
)
def test_manifest_rejects_malformed_plate_inventory(plate, wells):
    _edit_json(
        plate / "zarr.json",
        lambda data: data["attributes"]["ome"]["plate"].update(wells=wells),
    )
    with pytest.raises(ValueError):
        _manifest(plate)


@pytest.mark.parametrize("level_count", [-100, None])
def test_corrupt_successful_field_level_counts_are_rejected(
    complete_report, level_count
):
    _, manifest, report = complete_report

    def edit(rows):
        row = _field_rows(rows)[0]
        assert row["passed"]
        row["multiscale_level_count"] = level_count

    with pytest.raises(ValueError):
        rollup_reports([_rewrite_report(report, edit)], manifest)


def test_successful_field_with_levels_requires_chunk_shape_evidence(complete_report):
    _, manifest, report = complete_report

    def edit(rows):
        row = _field_rows(rows)[0]
        assert row["passed"] and row["multiscale_level_count"] > 0
        facts = json.loads(row["hcs_facts"])
        facts["chunk_shape"] = None
        row["hcs_facts"] = json.dumps(facts)

    with pytest.raises(ValueError):
        rollup_reports([_rewrite_report(report, edit)], manifest)


def test_zero_level_error_report_remains_representable_and_incomplete(complete_report):
    """A source validation failure is distinct from corrupted successful evidence."""
    _, manifest, report = complete_report

    def edit(rows):
        row = _field_rows(rows)[0]
        row["passed"] = False
        row["multiscale_level_count"] = 0
        row["issues"].append(
            {
                "loc": ["multiscale_level_count"],
                "severity": "error",
                "message": "Source image has no multiscale levels",
                "code": None,
            }
        )
        facts = json.loads(row["hcs_facts"])
        facts["chunk_shape"] = None
        row["hcs_facts"] = json.dumps(facts)

    rolled = rollup_reports([_rewrite_report(report, edit)], manifest)
    assert rolled.summary.is_partial
    assert not _root(rolled).passed
    assert _root(rolled).subset_coverage.fields_selected == 3
    assert any(
        issue.message == "Source image has no multiscale levels"
        for issue in _issues(rolled)
    )
    assert any(issue.loc == ("rollup", "coverage") for issue in _root(rolled).errors)


def test_native_image_with_no_declared_levels_rolls_up_as_incomplete(plate):
    _edit_json(
        plate / "A/1/1/zarr.json",
        lambda data: data["attributes"]["ome"]["multiscales"][0].update(datasets=[]),
    )
    manifest = _manifest(plate)
    original = _validate(plate)
    field = next(result for result in original if result.node_path.endswith("/A/1/1"))
    assert not field.passed
    assert any("metadata_extraction" in issue.loc for issue in field.errors)
    rolled = rollup_reports([build_report(original)], manifest)
    assert rolled.summary.is_partial
    assert not _root(rolled).passed
    assert _root(rolled).subset_coverage.fields_selected == 3
    rolled_field = next(
        result for result in rolled if result.node_path == field.node_path
    )
    assert rolled_field.errors == field.errors
