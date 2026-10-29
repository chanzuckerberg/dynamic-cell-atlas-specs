"""End to end: the engine writes a store that the repo's validator accepts."""

import json
import shutil
from pathlib import Path

import pytest

from dca_helpers.metadata import (
    DcaValidationError,
    load_dca_config,
    merge_dca_attributes,
    resolve_dca,
    validate_dca,
    write_normalization_statistics,
)
from dca_helpers.validation.runner import validate

FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "v0_2"
    / "valid_plate_multiwell.ome.zarr"
)

BASE_TOML = """
[dca.study]
source = "Example Lab"
study_type = ["Screen"]
description = "Example screen"
year_imaged = "2024"
experimenter = [{ first_name = "A", last_name = "B" }]
dataset_name = { column = "store_dataset_name" }
release_date = { column = "store_name_date" }
license = "CC-BY-4.0"

[dca.biosample]
group_id = { column = "store_group_id" }
organism = [{ ontology_term_id = "NCBITaxon:9606", label = "Homo sapiens" }]
tissue_type = "cell line"
cell_line_name = { column = "cell_line" }
tissue = [{ ontology_term_id = "CVCL:0042", label = "U2OS" }]
disease = [{ ontology_term_id = "MONDO:0009807", label = "osteosarcoma" }]
development_stage = [{ ontology_term_id = "unknown", label = "unknown" }]

[dca.specimen]
specimen_state = "fixed"

[dca.study_component]
imaging_method = [{ ontology_term_id = "FBbi:00000246", label = "fluorescence microscopy" }]
assay = [{ ontology_term_id = "EFO:0007550", label = "high content screen" }]

[dca.channels.DAPI]
name = "DAPI"
description = "Nuclear stain"

[dca.channels.GFP]
name = "GFP"
description = "Green fluorescent protein"

[dca.channels.mCherry]
name = "mCherry"
description = "Red fluorescent protein"

[dca.provenance]
"""

STAT = {"p1": 1.0, "p5": 2.0, "p95": 9.0, "p99": 10.0}


def _fov_files(store: Path) -> list[Path]:
    return sorted(store.glob("*/*/*/zarr.json"))


def _read_attrs(zarr_json: Path) -> dict:
    return json.loads(zarr_json.read_text())["attributes"]


def _write_attrs(zarr_json: Path, attrs: dict) -> None:
    doc = json.loads(zarr_json.read_text())
    doc["attributes"] = attrs
    zarr_json.write_text(json.dumps(doc, indent=2, allow_nan=False))


@pytest.fixture
def bare_store(tmp_path: Path) -> Path:
    """The multiwell fixture with every FOV's ``dca`` block removed."""
    store = tmp_path / "plate.ome.zarr"
    shutil.copytree(FIXTURE, store)
    for fov in _fov_files(store):
        attrs = _read_attrs(fov)
        attrs.pop("dca")
        _write_attrs(fov, attrs)
    return store


@pytest.fixture
def base_block(tmp_path: Path) -> dict:
    base = tmp_path / "base.toml"
    base.write_text(BASE_TOML)
    return load_dca_config(base)


def _channel_names(store: Path, fov: Path) -> list[str]:
    original = _read_attrs(FIXTURE / fov.relative_to(store))
    return [c["name"] for c in original["dca"]["channels"]]


def _row(fov: Path, store: Path, **overrides: str) -> dict:
    well = "/".join(fov.relative_to(store).parts[:-1])
    return {
        "store_dataset_name": "example_plate",
        "store_group_id": well,
        "store_name_date": "2024-01-01",
        "cell_line": "U2OS",
        **overrides,
    }


def _resolve_all(store: Path, block: dict, rows: dict[Path, dict]) -> dict[Path, dict]:
    payloads = {}
    for fov in _fov_files(store):
        where = "/".join(fov.relative_to(store).parts[:-1])
        resolved = resolve_dca(
            block, rows[fov], channel_keys=_channel_names(store, fov), where=where
        )
        payloads[fov] = validate_dca(resolved, where=where)
    return payloads


def _write_all(payloads: dict[Path, dict]) -> None:
    for fov, payload in payloads.items():
        attrs = _read_attrs(fov)
        merge_dca_attributes(
            attrs,
            {k: v for k, v in payload.items() if k != "channels"},
            payload["channels"],
        )
        stats = {
            str(c["index"]): {"dataset_statistics": STAT} for c in payload["channels"]
        }
        write_normalization_statistics(attrs, stats)
        _write_attrs(fov, attrs)


def test_engine_output_passes_the_validator(bare_store, base_block):
    assert len(_fov_files(bare_store)) == 4
    assert not any("dca" in _read_attrs(f) for f in _fov_files(bare_store))
    rows = {fov: _row(fov, bare_store) for fov in _fov_files(bare_store)}

    before = validate(str(bare_store), show_progress=False)
    assert not all(r.passed for r in before), "a store without dca must fail"

    _write_all(_resolve_all(bare_store, base_block, rows))

    run = validate(str(bare_store), show_progress=False)
    errors = [
        (r.node_path, i.message)
        for r in run
        for i in r.issues
        if i.severity.value == "error"
    ]
    assert errors == []
    assert sum(r.parent_path is not None for r in run) == 4, "one result per FOV"
    assert all(r.passed for r in run)


def test_a_row_missing_a_required_value_raises_naming_the_fov_and_field(
    bare_store, base_block
):
    fovs = _fov_files(bare_store)
    rows = {fov: _row(fov, bare_store) for fov in fovs}
    rows[fovs[2]] = _row(fovs[2], bare_store, cell_line="")

    with pytest.raises(DcaValidationError) as excinfo:
        _resolve_all(bare_store, base_block, rows)

    assert excinfo.value.where == "B/1/0"
    assert excinfo.value.fields == ["biosample"]
    assert "cell_line_name is required" in str(excinfo.value)
