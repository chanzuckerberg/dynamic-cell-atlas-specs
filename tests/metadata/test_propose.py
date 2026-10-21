import importlib

import pytest
import requests

from dca_helpers.metadata.propose import Proposal, propose
from dca_helpers.validation.ontology_lookup import TermRecord

propose_module = importlib.import_module("dca_helpers.metadata.propose")


def _one(field, **kwargs):
    kwargs.setdefault("hints", {})
    [p] = propose([field], **kwargs)
    return p


def test_cellosaurus_proposes_curie_and_tissue_term(monkeypatch):
    monkeypatch.setattr(
        propose_module, "search_cellosaurus", lambda name: [("CVCL:0023", "A549")]
    )
    hints = {"biosample.cell_line_name": "A549"}
    assert _one("biosample.cell_line_id", hints=hints).value == "CVCL:0023"
    assert _one("biosample.tissue", hints=hints).value == [
        {"ontology_term_id": "CVCL:0023", "label": "A549"}
    ]


def test_ambiguous_match_lists_alternatives_in_source(monkeypatch):
    monkeypatch.setattr(
        propose_module,
        "search_cellosaurus",
        lambda name: [("CVCL:0023", "A549"), ("CVCL:X1", "A549/ATCC")],
    )
    p = _one("biosample.cell_line_id", hints={"biosample.cell_line_name": "A549"})
    assert p.value == "CVCL:0023"
    assert p.source == "Cellosaurus CVCL:0023 (also: CVCL:X1 A549/ATCC)"


def test_ncbitaxon_for_a_virus_name(monkeypatch):
    monkeypatch.setattr(
        propose_module,
        "search_ncbitaxon",
        lambda name: [("NCBITaxon:64320", "Zika virus")],
    )
    p = _one(
        "perturbation.definition.perturbation_id",
        hints={"perturbation.definition.name": "Zika virus"},
    )
    assert (p.value, p.rule) == ("NCBITaxon:64320", "ncbitaxon")


def test_ols_label_looks_up_the_hinted_term(monkeypatch):
    seen = []

    def fake(iri):
        seen.append(iri)
        return TermRecord(source="OLS", names=("lung", "pulmonary"))

    monkeypatch.setattr(propose_module, "lookup_ols", fake)
    p = _one(
        "biosample.organ.label",
        hints={"biosample.organ.ontology_term_id": "UBERON:0002048"},
    )
    assert seen == ["http://purl.obolibrary.org/obo/UBERON_0002048"]
    assert (p.value, p.rule) == ("lung", "ols_label")


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"online": False}, "lookup skipped (offline)"),
        ({"hints": {}}, "no hint: biosample.cell_line_name"),
    ],
    ids=["offline", "no-hint"],
)
def test_lookup_rules_degrade_to_a_reasoned_none(kwargs, reason):
    kwargs = {"hints": {"biosample.cell_line_name": "A549"}, **kwargs}
    assert _one("biosample.cell_line_id", **kwargs) == Proposal(
        "biosample.cell_line_id", None, reason, "cellosaurus"
    )


def test_no_candidates_is_no_match(monkeypatch):
    monkeypatch.setattr(propose_module, "search_cellosaurus", lambda name: [])
    p = _one("biosample.cell_line_id", hints={"biosample.cell_line_name": "nope"})
    assert (p.value, p.source) == (None, "no match")


def test_registry_failure_degrades_to_a_reasoned_none(monkeypatch):
    def boom(name):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(propose_module, "search_cellosaurus", boom)
    p = _one("biosample.cell_line_id", hints={"biosample.cell_line_name": "A549"})
    assert (p.value, p.source) == (None, "lookup failed: api.cellosaurus.org")


def test_ols_failure_names_the_ols_host(monkeypatch):
    def boom(iri):
        raise propose_module.ontology_lookup._RegistryRecentlyUnreachable("down")

    monkeypatch.setattr(propose_module, "lookup_ols", boom)
    p = _one(
        "biosample.organ.label",
        hints={"biosample.organ.ontology_term_id": "UBERON:0002048"},
    )
    assert (p.value, p.source) == (None, "lookup failed: www.ebi.ac.uk")


def test_unknown_field_gets_no_proposal():
    unknown = [
        "study.license",
        "study.release_date",
        "channels.0.channel_type",
    ]
    assert propose(unknown, hints={}) == []


def _fake_cellosaurus(monkeypatch, lines):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"Cellosaurus": {"cell-line-list": lines}}

    monkeypatch.setattr(propose_module, "_get_with_retry", lambda url, params: Resp())


def test_search_cellosaurus_keeps_only_exact_names(monkeypatch):
    _fake_cellosaurus(
        monkeypatch,
        [
            {
                "accession-list": [{"value": "CVCL_DX70"}],
                "name-list": [{"value": "A549 Cas9"}],
            },
            {
                "accession-list": [{"value": "CVCL_0023"}],
                "name-list": [{"value": "A-549"}, {"value": "A549"}],
            },
        ],
    )
    assert propose_module.search_cellosaurus("a549") == [("CVCL:0023", "A549")]


def test_fuzzy_only_cellosaurus_result_is_no_match(monkeypatch):
    _fake_cellosaurus(
        monkeypatch,
        [
            {
                "accession-list": [{"value": "CVCL_YJ67"}],
                "name-list": [{"value": "VAMPIRO HEK293"}],
            }
        ],
    )
    p = _one("biosample.cell_line_id", hints={"biosample.cell_line_name": "HEK293"})
    assert (p.value, p.source) == (None, "no match")
