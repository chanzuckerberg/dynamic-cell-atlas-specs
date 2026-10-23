import importlib

import pytest
import requests

from dca_helpers.metadata.propose import Proposal, propose

from ._registry import UnrecordedRequest

propose_module = importlib.import_module("dca_helpers.metadata.propose")


def _one(field, **kwargs):
    kwargs.setdefault("hints", {})
    [p] = propose([field], **kwargs)
    return p


def test_cellosaurus_proposes_curie_and_tissue_term(replay_registry):
    hints = {"biosample.cell_line_name": "A549"}
    assert _one("biosample.cell_line_id", hints=hints).value == "CVCL:0023"
    assert _one("biosample.tissue", hints=hints).value == [
        {"ontology_term_id": "CVCL:0023", "label": "A549"}
    ]
    assert [params["q"] for _, params in replay_registry.requests[:2]] == [
        'id:"A549"',
        'sy:"A549"',
    ]


def test_cellosaurus_parent_line_beats_its_derivatives(replay_registry):
    p = _one("biosample.cell_line_id", hints={"biosample.cell_line_name": "HEK293"})
    assert (p.value, p.source) == ("CVCL:0045", "Cellosaurus CVCL:0045")


def test_cellosaurus_derivative_only_result_is_no_match(replay_registry):
    p = _one("biosample.cell_line_id", hints={"biosample.cell_line_name": "HeLa Cas9"})
    assert (p.value, p.source) == (None, "no match")


def test_ncbitaxon_for_a_virus_name_lists_alternatives(replay_registry):
    p = _one(
        "perturbation.definition.perturbation_id",
        hints={"perturbation.definition.name": "Zika virus"},
    )
    assert (p.value, p.rule) == ("NCBITaxon:64320", "ncbitaxon")
    assert p.source.startswith("NCBITaxon NCBITaxon:64320 (also: NCBITaxon:2043570 ")


def test_ols_label_looks_up_the_hinted_term(replay_registry):
    p = _one(
        "biosample.organ.label",
        hints={"biosample.organ.ontology_term_id": "UBERON:0002048"},
    )
    assert (p.value, p.source, p.rule) == ("lung", "OLS UBERON:0002048", "ols_label")
    assert replay_registry.requests == [
        (
            "https://www.ebi.ac.uk/ols4/api/terms",
            {"iri": "http://purl.obolibrary.org/obo/UBERON_0002048"},
        )
    ]


def test_unrecorded_request_fails_loudly(replay_registry):
    with pytest.raises(UnrecordedRequest):
        propose(
            ["biosample.cell_line_id"],
            hints={"biosample.cell_line_name": "never recorded"},
        )


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


def test_registry_failure_degrades_to_a_reasoned_none(monkeypatch):
    def down(url, params):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(propose_module, "_get_with_retry", down)
    p = _one("biosample.cell_line_id", hints={"biosample.cell_line_name": "A549"})
    assert (p.value, p.source) == (None, "lookup failed: api.cellosaurus.org")


def test_unknown_field_gets_no_proposal():
    unknown = [
        "study.license",
        "study.release_date",
        "channels.0.channel_type",
    ]
    assert propose(unknown, hints={}) == []
