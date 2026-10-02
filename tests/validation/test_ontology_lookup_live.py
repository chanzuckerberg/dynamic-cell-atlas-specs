"""Live tests for the ontology-lookup guard: real calls to OLS4 and Cellosaurus.

Skipped unless ``--live`` is passed. Run them periodically (not per-commit) to
catch the two things a mocked suite structurally cannot:

  1. **The API contract moved.** Every other test stubs ``requests.get`` and
     hands back a payload shaped the way OLS4 answered in 2026. If OLS renames
     ``_embedded.terms``, drops ``is_defining_ontology``, or changes how the
     ancestors endpoint wants its IRI encoded, the mocked suite stays green and
     the validator breaks in the field.
  2. **A term moved under us.** The routing table and the canary labels below
     are hand-maintained mirrors of what the registries actually hold. An
     ontology dropping off OLS, or renaming a term whose label we assert, is
     invisible offline.

Failures here are not necessarily *our* bug — a registry outage looks the same
as a contract change. Read the failure before changing code: if OLS is simply
down, the guard already degrades to WARNING by design and nothing needs fixing.

    pytest tests/validation/test_ontology_lookup_live.py --live -v

Every CURIE/label pair below was verified against the live service when written.
Do not add one without checking it; a guessed label turns this file into a
generator of false alarms.
"""

from __future__ import annotations

import pytest
import requests

from dca_helpers.validation import ontology_lookup
from dca_helpers.validation.ontology_lookup import (
    _OLS_PREFIXES,
    OLS_BASE_URL,
    check_ontology_terms,
    lookup_cellosaurus,
    lookup_ols,
    lookup_ols_ancestors,
)
from dca_helpers.validation.result import Severity
from dca_helpers.validation.versions.v0_2._ontology_terms import Term

pytestmark = [pytest.mark.live, pytest.mark.slow]


@pytest.fixture(autouse=True)
def _clear_failure_cache():
    """Keep one real outage from cascading into every later test in the file.

    The negative cache marks a host unreachable for a TTL, so without this a
    single genuine failure would short-circuit the rest of the run and report
    them as skips rather than letting each probe for itself.
    """
    ontology_lookup._registry_failures.clear()
    yield
    ontology_lookup._registry_failures.clear()


# One real term per OLS-routed prefix, so a prefix silently leaving OLS fails
# here. Labels verified live; see the module docstring before editing.
CANARY_TERMS: dict[str, tuple[str, str]] = {
    "CL": ("CL:0000000", "cell"),
    "EFO": ("EFO:0000001", "experimental factor"),
    "FBbi": ("FBbi:00000251", "confocal microscopy"),
    "FBdv": ("FBdv:00005342", "prepupal stage"),
    "HsapDv": ("HsapDv:0000087", "adult stage"),
    "MmusDv": ("MmusDv:0000036", "Theiler stage 27"),
    "MONDO": ("MONDO:0000001", "disease"),
    "NCBITaxon": ("NCBITaxon:9606", "Homo sapiens"),
    "PATO": ("PATO:0000461", "normal"),
    "UBERON": ("UBERON:0000955", "brain"),
    "WBls": ("WBls:0000024", "L1 larva Ce"),
    "ZFA": ("ZFA:0000047", "peripheral olfactory organ"),
    "ZFS": ("ZFS:0000029", "pharyngula:prim-5"),
}

# Cellosaurus is a separate service with a separate response shape.
CANARY_CELL_LINE = ("CVCL:0030", "HeLa")


def _biosample(term_id: str, label: str) -> dict:
    return {"biosample": {"organism": [{"ontology_term_id": term_id, "label": label}]}}


def test_every_routed_prefix_has_a_canary():
    """Adding a prefix to the routing table must come with a live canary.

    Otherwise a newly routed ontology is only ever exercised against a mock,
    which is how the v0.2.4 prefixes went unchecked in the first place.
    """
    missing = set(_OLS_PREFIXES) - set(CANARY_TERMS)
    assert not missing, (
        f"{sorted(missing)} are routed to OLS but have no live canary term. "
        "Add one (verified against the live service) to CANARY_TERMS."
    )


@pytest.mark.parametrize(
    ("prefix", "term_id", "label"),
    [(p, t, lbl) for p, (t, lbl) in sorted(CANARY_TERMS.items())],
)
def test_canary_term_still_resolves_with_the_same_label(prefix, term_id, label):
    """Each ontology is still on OLS and still gives this term this label."""
    iri = Term.parse(term_id).iri()
    assert iri is not None, f"{term_id} did not expand; check the schema prefix table"

    record = lookup_ols(iri)
    assert record is not None, f"{term_id} no longer resolves on OLS"
    assert record.names, f"{term_id} resolved but OLS returned no names"

    names = {name.strip().lower() for name in record.names}
    assert label.lower() in names, (
        f"{term_id} resolved but '{label}' is no longer among its names "
        f"({sorted(record.names)[:5]}). Either the term was renamed — update "
        f"CANARY_TERMS — or the label in the spec docs needs revisiting."
    )


def test_ols_terms_response_keeps_its_shape():
    """The `/terms?iri=` contract `lookup_ols` parses, asserted directly.

    `lookup_ols` reads `_embedded.terms[]`, prefers the entry flagged
    `is_defining_ontology`, and takes `label` plus `synonyms`. Assert those keys
    exist rather than only that the wrapper works, so a contract change reports
    as "shape moved" instead of a confusing downstream miss.
    """
    resp = requests.get(
        f"{OLS_BASE_URL}/terms",
        params={"iri": "http://purl.obolibrary.org/obo/NCBITaxon_9606"},
        headers={"User-Agent": "dca-helpers-validator live test"},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    assert "_embedded" in payload, "OLS dropped the _embedded envelope"
    terms = payload["_embedded"].get("terms")
    assert terms, "OLS returned no terms for NCBITaxon:9606"
    assert any("label" in term for term in terms), "no term carries `label`"
    assert any(
        "is_defining_ontology" in term for term in terms
    ), "no term carries `is_defining_ontology`; lookup_ols's preference is dead code"


def test_ancestors_endpoint_still_reports_obo_ids():
    """The subtree check depends on a double-encoded IRI path and `obo_id`.

    `FBbi:00000251` (confocal microscopy) descends from `FBbi:00000222`
    (imaging method) — the ancestor `preparation_method` / `imaging_method` are
    constrained to. If OLS changes the encoding it wants, this returns nothing
    and every subtree check silently degrades to a warning.
    """
    term = Term.parse("FBbi:00000251")
    ancestors = lookup_ols_ancestors(term.ontology, term.iri())

    assert ancestors, "ancestors endpoint returned nothing; check IRI encoding"
    assert "FBbi:00000222" in ancestors, (
        "FBbi:00000251 no longer descends from FBbi:00000222 in OLS; the "
        "_SUBTREE_CONSTRAINTS ancestor for imaging_method needs revisiting"
    )


@pytest.mark.parametrize(
    ("ancestor", "expected_label"),
    [
        ("FBbi:00000001", "sample preparation method"),
        ("FBbi:00000222", "imaging method"),
    ],
)
def test_subtree_ancestors_still_exist_with_expected_meaning(ancestor, expected_label):
    """The two hardcoded ancestor CURIEs still mean what the constraint assumes.

    A wrong ancestor here would reject every valid `preparation_method` /
    `imaging_method`, so it is worth pinning the label and not just existence.
    """
    record = lookup_ols(Term.parse(ancestor).iri())
    assert record is not None, f"{ancestor} no longer resolves"
    names = {name.strip().lower() for name in record.names}
    assert expected_label in names, (
        f"{ancestor} is now {sorted(record.names)[:3]}, not '{expected_label}'; "
        "_SUBTREE_CONSTRAINTS is pointing at the wrong node"
    )


def test_cellosaurus_still_resolves_a_cell_line():
    """Cellosaurus keeps its own shape: Cellosaurus.cell-line-list[].name-list[]."""
    term_id, label = CANARY_CELL_LINE
    record = lookup_cellosaurus(Term.parse(term_id).cellosaurus_accession)

    assert record is not None, f"{term_id} no longer resolves on Cellosaurus"
    assert record.source == "Cellosaurus"
    assert record.names, "Cellosaurus resolved but returned no names"
    names = {name.strip().lower() for name in record.names}
    assert label.lower() in names, (
        f"'{label}' is no longer among {term_id}'s Cellosaurus names "
        f"({sorted(record.names)[:5]})"
    )


def test_end_to_end_clean_metadata_reports_nothing():
    """The whole guard over a small real object: no issues, no false positives."""
    term_id, label = CANARY_TERMS["NCBITaxon"]
    assert check_ontology_terms(_biosample(term_id, label)) == []


def test_end_to_end_wrong_label_is_reported():
    """A real term with a wrong label is an ERROR against the live registry."""
    issues = check_ontology_terms(_biosample("NCBITaxon:9606", "Mus musculus"))
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR


def test_end_to_end_wrong_subtree_is_reported():
    """A real FBbi term from the wrong branch is an ERROR.

    `FBbi:00000251` (confocal microscopy) is a genuine FBbi term with a correct
    label, so only the subtree check can reject it in `preparation_method`.
    """
    obj = {
        "specimen": {
            "preparation_method": [
                {"ontology_term_id": "FBbi:00000251", "label": "confocal microscopy"}
            ]
        }
    }
    issues = check_ontology_terms(obj)
    assert any(
        issue.severity == Severity.ERROR and "subtree" in issue.message
        for issue in issues
    ), f"expected a subtree ERROR, got {[(i.severity, i.message) for i in issues]}"
