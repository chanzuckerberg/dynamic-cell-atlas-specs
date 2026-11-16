"""CLI contracts for manifest creation and offline HCS consistency rollup."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from dca_helpers.validation.cli import main as validate_cli
from dca_helpers.validation.hcs_facts import HCSManifest
from dca_helpers.validation.reporter import build_report
from dca_helpers.validation.rollup import create_hcs_manifest
from dca_helpers.validation.rollup_cli import main
from dca_helpers.validation.runner import validate

FIXTURE = Path(__file__).parent.parent / "fixtures/v0_2/valid_plate_multiwell.ome.zarr"
SNAPSHOT = "immutable-revision-1"
pytestmark = pytest.mark.functional


def _inputs(tmp_path):
    path = str(FIXTURE)
    manifest = create_hcs_manifest(path, SNAPSHOT)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(manifest.model_dump_json())
    reports = []
    for offset in (0, 1):
        report = tmp_path / f"chunk-{offset}.parquet"
        run = validate(
            path,
            snapshot_id=SNAPSHOT,
            field_offset=offset,
            field_limit=1,
            show_progress=False,
        )
        report.write_bytes(build_report(run))
        reports.append(report)
    return manifest_file, reports


def test_manifest_command_reads_only_metadata(tmp_path):
    output = tmp_path / "nested" / "manifest.json"
    result = CliRunner().invoke(
        main,
        [
            "manifest",
            str(FIXTURE),
            "--snapshot-id",
            SNAPSHOT,
            "--output-file",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = HCSManifest.model_validate_json(output.read_text())
    assert manifest.snapshot_id == SNAPSHOT
    assert sum(len(w.expected_fields) for w in manifest.plate.wells.values()) == 4


def test_validate_cli_persists_snapshot_id(tmp_path):
    result = CliRunner().invoke(
        validate_cli,
        [
            str(FIXTURE),
            "--snapshot-id",
            SNAPSHOT,
            "--field-limit",
            "1",
            "--output",
            "parquet",
            "--output-directory",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    import pyarrow.parquet as pq

    rows = pq.read_table(next(tmp_path.glob("*.parquet"))).to_pylist()
    assert all(row["snapshot_id"] == SNAPSHOT for row in rows)


@pytest.mark.parametrize("output", ["text", "json", "parquet"])
def test_complete_rollup_output_modes(tmp_path, output):
    manifest, reports = _inputs(tmp_path)
    args = ["check", str(manifest), *(str(p) for p in reports), "--output", output]
    if output == "parquet":
        args += ["--output-directory", str(tmp_path / "results")]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    if output == "json":
        rows = json.loads(result.output)
        assert len(rows) == 5
        assert all(row["passed"] for row in rows)
        assert not any(row.get("partial") for row in rows)
    elif output == "parquet":
        assert len(list((tmp_path / "results").glob("*.parquet"))) == 1
    else:
        assert "PARTIAL" not in result.output


@pytest.mark.parametrize("output", ["text", "json"])
def test_incomplete_rollup_exits_nonzero_and_exposes_coverage(tmp_path, output):
    manifest, reports = _inputs(tmp_path)
    result = CliRunner().invoke(
        main, ["check", str(manifest), str(reports[0]), "--output", output]
    )
    assert result.exit_code == 1
    assert "Incomplete HCS rollup" in result.output
    if output == "json":
        root = json.loads(result.output)[0]
        assert root["partial"]
        assert root["subset_coverage"]["fields_total"] == 4
        assert root["subset_coverage"]["fields_selected"] == 2
    else:
        assert "PARTIAL" in result.output


def test_manifest_snapshot_mismatch_is_clean_cli_error(tmp_path):
    manifest, reports = _inputs(tmp_path)
    data = json.loads(manifest.read_text())
    data["snapshot_id"] = "other-revision"
    manifest.write_text(json.dumps(data))
    result = CliRunner().invoke(
        main, ["check", str(manifest), *(str(p) for p in reports)]
    )
    assert result.exit_code == 1
    assert "Incompatible snapshot_id" in result.output
    assert "Traceback" not in result.output


def test_output_directory_requires_parquet(tmp_path):
    manifest, reports = _inputs(tmp_path)
    result = CliRunner().invoke(
        main,
        ["check", str(manifest), str(reports[0]), "--output-directory", str(tmp_path)],
    )
    assert result.exit_code == 2
    assert "requires --output parquet" in result.output
