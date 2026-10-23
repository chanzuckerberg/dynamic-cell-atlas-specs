import pytest

from dca_helpers.metadata import (
    ensembl_from_ncbi_gene,
    pubchem_smiles,
    search_ensembl,
    search_pubchem,
)
from tests.conftest import ASPIRIN_PUBCHEM_CID, TP53_ENSEMBL_GENE_ID


@pytest.mark.live
def test_live_tp53_symbol_and_ncbi_gene_map_to_one_ensembl_gene():
    assert search_ensembl("TP53")[0] == TP53_ENSEMBL_GENE_ID
    assert ensembl_from_ncbi_gene("NCBIGene:7157")[0] == TP53_ENSEMBL_GENE_ID


@pytest.mark.live
def test_live_fly_symbol_maps_to_a_flybase_gene_id():
    assert search_ensembl("w", species="drosophila_melanogaster") == [
        "ensembl:FBgn0003996"
    ]


@pytest.mark.live
def test_live_aspirin_by_name_and_smiles():
    assert search_pubchem(name="aspirin")[0] == ASPIRIN_PUBCHEM_CID
    assert search_pubchem(smiles="CC(=O)Oc1ccccc1C(=O)O")[0] == ASPIRIN_PUBCHEM_CID


@pytest.mark.live
@pytest.mark.parametrize(
    "smiles, cid",
    [("C/C=C/C", "PubChem:62695"), ("C#N", "PubChem:768")],
    ids=["slash", "hash"],
)
def test_live_smiles_with_url_special_characters(smiles, cid):
    assert search_pubchem(smiles=smiles)[0] == cid


@pytest.mark.live
def test_live_aspirin_record_smiles():
    assert pubchem_smiles(ASPIRIN_PUBCHEM_CID) == "CC(=O)OC1=CC=CC=C1C(=O)O"
