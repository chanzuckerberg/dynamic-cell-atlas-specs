"""Store walk: the beta perturbation identifiers warn and the nodes still pass.

Writes one assignment into the image-level ``dca.perturbation`` of every field of
well A/1 and into that well's ``dca.well_perturbation``, then runs the real
``validate()`` stack over the copied multiwell fixture.
"""

import json
import shutil
from pathlib import Path

import pytest

from dca_helpers.validation.result import Severity
from dca_helpers.validation.runner import validate
from tests.conftest import ASPIRIN_PUBCHEM_CID

FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "v0_2"
    / "valid_plate_multiwell.ome.zarr"
)

ASSIGNMENT = {
    "control_class": "treatment",
    "definition": {
        "perturbation_id": "P1",
        "modality": "compound",
        "chemical": {"pubchem_cid": ASPIRIN_PUBCHEM_CID},
    },
    "cotreatment": {
        "perturbation_id": "P2",
        "modality": "crispr",
        "genetic": {"gene_symbol": "TP53"},
    },
}

EXPECTED = {
    "definition.chemical.morgan_fingerprint is missing for modality 'compound' "
    "(beta in v0.2; required from v0.3)",
    "cotreatment.genetic.ensembl_gene_id is missing for modality 'crispr' "
    "(beta in v0.2; required from v0.3)",
}


def _set_dca(zarr_json: Path, **blocks) -> None:
    doc = json.loads(zarr_json.read_text())
    doc["attributes"].setdefault("dca", {}).update(blocks)
    zarr_json.write_text(json.dumps(doc, indent=2))


@pytest.fixture
def store(tmp_path: Path) -> Path:
    store = tmp_path / "plate.ome.zarr"
    shutil.copytree(FIXTURE, store)
    for field in ("0", "1"):
        _set_dca(store / "A" / "1" / field / "zarr.json", perturbation=ASSIGNMENT)
    row = {"plate": "plate", "well": "A1", **ASSIGNMENT}
    _set_dca(store / "A" / "1" / "zarr.json", well_perturbation=row)
    return store


def test_store_walk_reports_beta_ids_as_warnings_and_passes(store):
    run = validate(str(store), spec_version="0.2", show_progress=False)

    errors = [
        (r.node_path, i.message)
        for r in run
        for i in r.issues
        if i.severity is Severity.ERROR
    ]
    assert errors == []
    assert all(r.passed for r in run)

    beta = {
        (r.node_path, i.loc, i.message)
        for r in run
        for i in r.issues
        if i.severity is Severity.WARNING and "beta in v0.2" in i.message
    }
    fields = {f"{store}/A/1/{field}" for field in ("0", "1")}
    image_level = {(path, message) for path, _, message in beta if path in fields}
    assert image_level == {(path, m) for path in fields for m in EXPECTED}

    well_level = {
        message
        for _, loc, message in beta
        if loc[:5] == ("plate", "wells", "A/1", "dca", "well_perturbation")
    }
    assert well_level == EXPECTED
