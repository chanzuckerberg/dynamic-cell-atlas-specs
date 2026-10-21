import importlib
import subprocess
import sys

import pytest
import requests

from dca_helpers.metadata import (
    ensembl_from_ncbi_gene,
    morgan_fingerprint,
    search_ensembl,
    search_pubchem,
)
from tests.conftest import (
    ASPIRIN_MORGAN_FINGERPRINT,
    ASPIRIN_PUBCHEM_CID,
    TP53_ENSEMBL_GENE_ID,
)

from ._registry import UnrecordedRequest

perturbation_ids = importlib.import_module("dca_helpers.metadata.perturbation_ids")

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


def test_search_ensembl_keeps_gene_ids_and_drops_lrg_records(replay_registry):
    assert search_ensembl("TP53") == [TP53_ENSEMBL_GENE_ID]
    assert replay_registry.requests == [
        (
            "https://rest.ensembl.org/xrefs/symbol/homo_sapiens/TP53",
            {"object_type": "gene", "content-type": "application/json"},
        )
    ]


def test_search_ensembl_returns_every_gene_id_in_registry_order(replay_registry):
    ids = search_ensembl("CYP2D6")
    assert ids[0] == "ensembl:ENSG00000100197"
    assert len(ids) == 7
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("ensembl:ENSG") for i in ids)


def test_search_ensembl_unknown_symbol_is_an_empty_list(replay_registry):
    assert search_ensembl("NOTAREALGENEXYZ") == []


def test_ensembl_path_segments_are_percent_encoded(replay_registry):
    with pytest.raises(UnrecordedRequest):
        search_ensembl("A/B C", species="homo sapiens")
    with pytest.raises(UnrecordedRequest):
        ensembl_from_ncbi_gene("7157", species="mus/musculus")
    assert [url for url, _ in replay_registry.requests] == [
        "https://rest.ensembl.org/xrefs/symbol/homo%20sapiens/A%2FB%20C",
        "https://rest.ensembl.org/xrefs/symbol/mus%2Fmusculus/7157",
    ]


@pytest.mark.parametrize("ncbi_gene_id", ["NCBIGene:7157", "7157"])
def test_ensembl_from_ncbi_gene_accepts_curie_or_bare_id(replay_registry, ncbi_gene_id):
    assert ensembl_from_ncbi_gene(ncbi_gene_id) == [TP53_ENSEMBL_GENE_ID]
    assert replay_registry.requests[-1] == (
        "https://rest.ensembl.org/xrefs/symbol/homo_sapiens/7157",
        {
            "object_type": "gene",
            "external_db": "EntrezGene",
            "content-type": "application/json",
        },
    )


@pytest.mark.parametrize("bad", ["TP53", "NCBIGene:", "HGNC:11998", ""])
def test_ensembl_from_ncbi_gene_rejects_a_non_numeric_id(bad):
    with pytest.raises(ValueError, match="NCBI Gene id"):
        ensembl_from_ncbi_gene(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "aspirin"},
        {"smiles": ASPIRIN_SMILES},
        {"inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"},
    ],
    ids=["name", "smiles", "inchikey"],
)
def test_search_pubchem_finds_aspirin(replay_registry, kwargs):
    assert search_pubchem(**kwargs) == [ASPIRIN_PUBCHEM_CID]
    [(url, params)] = replay_registry.requests
    [namespace] = kwargs
    assert (
        url
        == f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{namespace}/cids/JSON"
    )
    assert params == kwargs


def test_search_pubchem_not_found_is_an_empty_list(replay_registry):
    assert search_pubchem(name="not a real compound name") == []


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"name": "aspirin", "smiles": ASPIRIN_SMILES}, {"name": ""}],
    ids=["none", "two", "empty"],
)
def test_search_pubchem_needs_exactly_one_identifier(kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        search_pubchem(**kwargs)


@pytest.mark.parametrize(
    "search", [lambda: search_ensembl("TP53"), lambda: search_pubchem(name="aspirin")]
)
def test_registry_failure_propagates_from_the_http_boundary(monkeypatch, search):
    def down(url, params):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(perturbation_ids, "_get_with_retry", down)
    with pytest.raises(requests.ConnectionError):
        search()


def test_morgan_fingerprint_matches_the_pinned_aspirin_value():
    assert morgan_fingerprint(ASPIRIN_SMILES) == ASPIRIN_MORGAN_FINGERPRINT


def test_morgan_fingerprint_ignores_kekule_form():
    assert morgan_fingerprint("CC(=O)OC1=CC=CC=C1C(=O)O") == ASPIRIN_MORGAN_FINGERPRINT


@pytest.mark.parametrize("smiles", ["C1CC(", "not a smiles", ""])
def test_morgan_fingerprint_rejects_invalid_smiles(smiles):
    with pytest.raises(ValueError, match="SMILES"):
        morgan_fingerprint(smiles)


def test_morgan_fingerprint_without_rdkit_names_the_extra(hide_rdkit):
    with pytest.raises(ImportError, match=r"dca_helpers\[chem\]"):
        morgan_fingerprint(ASPIRIN_SMILES)


def test_metadata_package_imports_without_rdkit():
    code = (
        "import sys\n"
        "sys.modules['rdkit'] = None\n"
        "import dca_helpers.metadata as m\n"
        "assert callable(m.morgan_fingerprint)\n"
        "assert 'rdkit.Chem' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
