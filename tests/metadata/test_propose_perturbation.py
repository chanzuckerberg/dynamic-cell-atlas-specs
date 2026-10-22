import importlib

import pytest
import requests

from dca_helpers.metadata.propose import Proposal, propose
from tests.conftest import (
    ASPIRIN_MORGAN_FINGERPRINT,
    ASPIRIN_PUBCHEM_CID,
    TP53_ENSEMBL_GENE_ID,
)

perturbation_ids = importlib.import_module("dca_helpers.metadata.perturbation_ids")

ENSEMBL = "perturbation.definition.genetic.ensembl_gene_id"
PUBCHEM = "perturbation.definition.chemical.pubchem_cid"
MORGAN = "perturbation.definition.chemical.morgan_fingerprint"
GENETIC = "perturbation.definition.genetic."
CHEMICAL = "perturbation.definition.chemical."
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


def _one(field, **kwargs):
    kwargs.setdefault("hints", {})
    [p] = propose([field], **kwargs)
    return p


def test_ensembl_from_gene_symbol(replay_registry):
    p = _one(ENSEMBL, hints={GENETIC + "gene_symbol": "TP53"})
    assert p == Proposal(
        ENSEMBL, TP53_ENSEMBL_GENE_ID, f"Ensembl {TP53_ENSEMBL_GENE_ID}", "ensembl"
    )


def test_ensembl_prefers_the_ncbi_gene_hint(replay_registry):
    hints = {GENETIC + "ncbi_gene_id": "NCBIGene:7157", GENETIC + "gene_symbol": "X"}
    assert _one(ENSEMBL, hints=hints).value == TP53_ENSEMBL_GENE_ID
    [(url, _)] = replay_registry.requests
    assert url.endswith("/homo_sapiens/7157")


def test_ensembl_lists_alternative_gene_ids(replay_registry):
    p = _one(ENSEMBL, hints={GENETIC + "gene_symbol": "CYP2D6"})
    assert p.value == "ensembl:ENSG00000100197"
    assert p.source.startswith("Ensembl ensembl:ENSG00000100197 (also: ensembl:ENSG")


def test_ensembl_unknown_symbol_is_no_match(replay_registry):
    p = _one(ENSEMBL, hints={GENETIC + "gene_symbol": "NOTAREALGENEXYZ"})
    assert (p.value, p.source) == (None, "no match")


def test_ensembl_invalid_ncbi_hint_degrades():
    p = _one(ENSEMBL, hints={GENETIC + "ncbi_gene_id": "TP53"})
    assert (p.value, p.source) == (None, f"invalid hint: {GENETIC}ncbi_gene_id")


@pytest.mark.parametrize(
    "hint, value",
    [
        ("inchikey", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"),
        ("smiles", ASPIRIN_SMILES),
    ],
)
def test_pubchem_from_a_structure_hint(replay_registry, hint, value):
    p = _one(PUBCHEM, hints={CHEMICAL + hint: value})
    assert p == Proposal(
        PUBCHEM, ASPIRIN_PUBCHEM_CID, f"PubChem {ASPIRIN_PUBCHEM_CID}", "pubchem"
    )


def test_pubchem_prefers_inchikey_over_smiles_and_name(replay_registry):
    hints = {
        "perturbation.definition.name": "never recorded",
        CHEMICAL + "smiles": "never recorded",
        CHEMICAL + "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
    }
    assert _one(PUBCHEM, hints=hints).value == ASPIRIN_PUBCHEM_CID


def test_pubchem_falls_back_to_the_perturbation_name(replay_registry):
    p = _one(PUBCHEM, hints={"perturbation.definition.name": "aspirin"})
    assert p.value == ASPIRIN_PUBCHEM_CID


def test_pubchem_unknown_name_is_no_match(replay_registry):
    p = _one(
        PUBCHEM, hints={"perturbation.definition.name": "not a real compound name"}
    )
    assert (p.value, p.source) == (None, "no match")


@pytest.mark.parametrize(
    "field, rule, reason",
    [
        (ENSEMBL, "ensembl", f"no hint: {GENETIC}ncbi_gene_id or {GENETIC}gene_symbol"),
        (
            PUBCHEM,
            "pubchem",
            f"no hint: {CHEMICAL}inchikey or {CHEMICAL}smiles"
            " or perturbation.definition.name",
        ),
    ],
)
def test_lookup_rules_without_a_hint_name_every_accepted_hint(field, rule, reason):
    assert _one(field) == Proposal(field, None, reason, rule)


@pytest.mark.parametrize("field, rule", [(ENSEMBL, "ensembl"), (PUBCHEM, "pubchem")])
def test_lookup_rules_skip_the_registry_offline(field, rule):
    hints = {GENETIC + "gene_symbol": "TP53", CHEMICAL + "smiles": ASPIRIN_SMILES}
    assert _one(field, hints=hints, online=False) == Proposal(
        field, None, "lookup skipped (offline)", rule
    )


@pytest.mark.parametrize(
    "field, hints, host",
    [
        (ENSEMBL, {GENETIC + "gene_symbol": "TP53"}, "rest.ensembl.org"),
        (PUBCHEM, {CHEMICAL + "smiles": "C"}, "pubchem.ncbi.nlm.nih.gov"),
        (
            MORGAN,
            {CHEMICAL + "pubchem_cid": "PubChem:2244"},
            "pubchem.ncbi.nlm.nih.gov",
        ),
    ],
)
def test_registry_failure_degrades(monkeypatch, field, hints, host):
    def down(url, params):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(perturbation_ids, "_get_with_retry", down)
    p = _one(field, hints=hints)
    assert (p.value, p.source) == (None, f"lookup failed: {host}")


def test_morgan_fingerprints_the_pubchem_record(replay_registry):
    rdkit = pytest.importorskip("rdkit")
    p = _one(MORGAN, hints={CHEMICAL + "pubchem_cid": ASPIRIN_PUBCHEM_CID})
    assert p == Proposal(
        MORGAN,
        ASPIRIN_MORGAN_FINGERPRINT,
        f"RDKit {rdkit.__version__} Morgan radius 2, 2048 bits, "
        f"from {ASPIRIN_PUBCHEM_CID} record SMILES",
        "morgan",
    )
    [(url, _)] = replay_registry.requests
    assert url.endswith("/compound/cid/2244/property/SMILES/JSON")


@pytest.mark.parametrize("online", [True, False])
def test_morgan_proposes_na_for_an_na_cid(replay_registry, online):
    p = _one(MORGAN, hints={CHEMICAL + "pubchem_cid": "NA"}, online=online)
    assert p == Proposal(MORGAN, "NA", f"{CHEMICAL}pubchem_cid is NA", "morgan")
    assert replay_registry.requests == []


@pytest.mark.parametrize(
    "hints, online, reason",
    [
        ({}, True, f"no hint: {CHEMICAL}pubchem_cid"),
        (
            {CHEMICAL + "smiles": ASPIRIN_SMILES},
            True,
            f"no hint: {CHEMICAL}pubchem_cid",
        ),
        (
            {CHEMICAL + "pubchem_cid": ASPIRIN_PUBCHEM_CID},
            False,
            "lookup skipped (offline)",
        ),
        (
            {CHEMICAL + "pubchem_cid": "PubChem:999999999"},
            True,
            f"invalid hint: {CHEMICAL}pubchem_cid",
        ),
        (
            {CHEMICAL + "pubchem_cid": "aspirin"},
            True,
            f"invalid hint: {CHEMICAL}pubchem_cid",
        ),
    ],
    ids=[
        "no-hint",
        "smiles-is-not-the-input",
        "offline",
        "no-record-smiles",
        "not-a-cid",
    ],
)
def test_morgan_degrades_to_a_reasoned_none(replay_registry, hints, online, reason):
    p = _one(MORGAN, hints=hints, online=online)
    assert p == Proposal(MORGAN, None, reason, "morgan")


def test_morgan_without_rdkit_names_the_extra(hide_rdkit):
    p = _one(MORGAN, hints={CHEMICAL + "pubchem_cid": ASPIRIN_PUBCHEM_CID})
    assert p == Proposal(
        MORGAN, None, "rdkit not installed (dca_helpers[chem])", "morgan"
    )
