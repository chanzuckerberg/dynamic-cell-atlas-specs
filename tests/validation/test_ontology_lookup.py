"""
Unit tests for the opt-in ontology-lookup consistency guard.

The guard resolves each ``ontology_term_id`` against the registry its prefix
belongs to (OLS4 for OBO/EFO ontologies, the Cellosaurus API for ``CVCL``) and
compares the stored ``label`` to the registry's canonical names:

  * id does not resolve                        -> ERROR
  * label matches none of the registry names   -> ERROR
  * registry unreachable                        -> WARNING (unverified, not invalid)
  * prefix no configured source hosts           -> skipped (not verifiable)
  * sentinel (unknown / na / unavailable)       -> skipped, no request

All network access is mocked; these tests never touch a live registry.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml

from dca_helpers.validation import ontology_lookup
from dca_helpers.validation.ontology_lookup import (
    TermRecord,
    check_ontology_terms,
    lookup_cellosaurus,
    lookup_ols,
    lookup_ols_ancestors,
)
from dca_helpers.validation.result import Severity
from dca_helpers.validation.versions.v0_2 import _ontology_terms, experimental_metadata
from dca_helpers.validation.versions.v0_2._ontology_terms import (
    Term,
    curie_to_iri,
    iter_ontology_terms,
)

_MEMOIZED = (
    lookup_ols,
    lookup_ols_ancestors,
    lookup_cellosaurus,
    _ontology_terms._prefix_iri_map,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    """Reset every cross-test cache.

    Beyond the memoized lookups, `_registry_failures` is a module-level negative
    cache: a test that simulates an unreachable registry would otherwise make the
    next test skip its request entirely and see a cached failure instead.
    """

    def _reset():
        for fn in _MEMOIZED:
            fn.cache_clear()
        ontology_lookup._registry_failures.clear()

    _reset()
    yield
    _reset()


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Skip real backoff sleeps so retry paths run instantly under test."""
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 500:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _fake_ols(terms_by_iri, ancestor_ids=None, calls=None):
    """requests.get stand-in for the OLS API.

    Serves both endpoints: ``/terms?iri=`` (keyed by IRI) and any
    ``…/ancestors`` request (returns ``ancestor_ids`` as obo_ids — fine because
    each subtree test uses a single term). ``calls`` records request URLs.
    """

    def _get(url, params=None, timeout=None, headers=None):
        if calls is not None:
            calls.append(url)
        if url.endswith("/ancestors"):
            terms = [{"obo_id": ancestor} for ancestor in (ancestor_ids or [])]
            return _FakeResponse({"_embedded": {"terms": terms}})
        iri = (params or {}).get("iri")
        return _FakeResponse({"_embedded": {"terms": terms_by_iri.get(iri, [])}})

    return _get


# --- curie_to_iri -----------------------------------------------------------


def test_curie_to_iri_expands_known_prefix():
    assert (
        curie_to_iri("FBbi:00000251") == "http://purl.obolibrary.org/obo/FBbi_00000251"
    )
    assert curie_to_iri("NCBITaxon:9606") == (
        "http://purl.obolibrary.org/obo/NCBITaxon_9606"
    )


def test_curie_to_iri_rejects_unknown_prefix_and_non_curie():
    assert curie_to_iri("NOTAPREFIX:123") is None
    assert curie_to_iri("no-colon-here") is None
    assert curie_to_iri("FBbi:") is None


# --- iter_ontology_terms ----------------------------------------------------


def test_iter_ontology_terms_walks_nested_structure():
    obj = {
        "biosample": {
            "organism": [
                {"ontology_term_id": "NCBITaxon:9606", "label": "Homo sapiens"}
            ],
            "tissue_type": "tissue",  # not an ontology term
        },
        "study_component": {
            "imaging_method": [
                {"ontology_term_id": "FBbi:00000251", "label": "confocal microscopy"}
            ]
        },
    }
    found = {(term_id, label) for _loc, term_id, label in iter_ontology_terms(obj)}
    assert found == {
        ("NCBITaxon:9606", "Homo sapiens"),
        ("FBbi:00000251", "confocal microscopy"),
    }


def test_iter_ontology_terms_reports_location_path():
    obj = {"biosample": {"organism": [{"ontology_term_id": "X:1", "label": "y"}]}}
    ((loc, _term_id, _label),) = list(iter_ontology_terms(obj))
    assert loc == ("biosample", "organism", 0)


# --- OLS-routed terms -------------------------------------------------------


def _term(term_id, label):
    return {"biosample": {"organism": [{"ontology_term_id": term_id, "label": label}]}}


def test_resolves_and_label_matches_is_clean(monkeypatch):
    iri = "http://purl.obolibrary.org/obo/NCBITaxon_9606"
    monkeypatch.setattr(
        "requests.get",
        _fake_ols({iri: [{"label": "Homo sapiens", "is_defining_ontology": True}]}),
    )
    assert check_ontology_terms(_term("NCBITaxon:9606", "Homo sapiens")) == []


def test_label_match_is_case_insensitive(monkeypatch):
    iri = "http://purl.obolibrary.org/obo/NCBITaxon_9606"
    monkeypatch.setattr("requests.get", _fake_ols({iri: [{"label": "Homo sapiens"}]}))
    assert check_ontology_terms(_term("NCBITaxon:9606", "  homo SAPIENS ")) == []


def test_synonym_match_is_accepted(monkeypatch):
    iri = "http://purl.obolibrary.org/obo/NCBITaxon_9606"
    monkeypatch.setattr(
        "requests.get",
        _fake_ols({iri: [{"label": "Homo sapiens", "synonyms": ["human"]}]}),
    )
    assert check_ontology_terms(_term("NCBITaxon:9606", "human")) == []


def test_label_mismatch_is_error(monkeypatch):
    iri = "http://purl.obolibrary.org/obo/NCBITaxon_9606"
    monkeypatch.setattr(
        "requests.get",
        _fake_ols({iri: [{"label": "Homo sapiens", "synonyms": []}]}),
    )
    issues = check_ontology_terms(_term("NCBITaxon:9606", "Mus musculus"))
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert "does not match" in issues[0].message


def test_unresolvable_id_is_error(monkeypatch):
    monkeypatch.setattr("requests.get", _fake_ols({}))  # no terms for any IRI
    issues = check_ontology_terms(_term("NCBITaxon:0000000", "nonexistent"))
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert "does not resolve" in issues[0].message


def test_prefers_defining_ontology(monkeypatch):
    iri = "http://purl.obolibrary.org/obo/NCBITaxon_9606"
    monkeypatch.setattr(
        "requests.get",
        _fake_ols(
            {
                iri: [
                    {"label": "imported copy", "is_defining_ontology": False},
                    {"label": "Homo sapiens", "is_defining_ontology": True},
                ]
            }
        ),
    )
    assert isinstance(lookup_ols(iri), TermRecord)
    assert check_ontology_terms(_term("NCBITaxon:9606", "Homo sapiens")) == []


def test_repeated_curie_is_fetched_once(monkeypatch):
    iri = "http://purl.obolibrary.org/obo/NCBITaxon_9606"
    calls: list[str] = []
    monkeypatch.setattr(
        "requests.get", _fake_ols({iri: [{"label": "Homo sapiens"}]}, calls=calls)
    )
    obj = {
        "biosample": {
            "organism": [
                {"ontology_term_id": "NCBITaxon:9606", "label": "Homo sapiens"},
                {"ontology_term_id": "NCBITaxon:9606", "label": "Homo sapiens"},
            ]
        }
    }
    assert check_ontology_terms(obj) == []
    # memoized: two identical terms, one request to the OLS /terms endpoint
    assert calls == ["https://www.ebi.ac.uk/ols4/api/terms"]


# --- sentinels / non-verifiable prefixes / network failures -----------------


@pytest.mark.parametrize("sentinel", ["unknown", "na", "unavailable"])
def test_sentinels_are_skipped_without_network(monkeypatch, sentinel):
    def _boom(*a, **k):
        raise AssertionError("must not call a registry for a sentinel")

    monkeypatch.setattr("requests.get", _boom)
    assert check_ontology_terms(_term(sentinel, sentinel)) == []


@pytest.mark.parametrize(
    "term_id",
    ["HGNC:11998", "NCBIGene:7157", "ensembl:ENSG00000141510", "PubChem:2244"],
)
def test_non_ols_prefixes_are_skipped_not_errored(monkeypatch, term_id):
    def _boom(*a, **k):
        raise AssertionError("must not call a registry for an unhosted prefix")

    monkeypatch.setattr("requests.get", _boom)
    assert check_ontology_terms(_term(term_id, "anything")) == []


def test_network_failure_is_warning_not_error(monkeypatch):
    def _raise(*a, **k):
        raise ConnectionError("OLS unreachable")

    monkeypatch.setattr("requests.get", _raise)
    issues = check_ontology_terms(_term("NCBITaxon:9606", "Homo sapiens"))
    assert len(issues) == 1
    assert issues[0].severity == Severity.WARNING
    assert "could not verify" in issues[0].message


# --- Cellosaurus-routed terms (CVCL) ----------------------------------------


def _cellosaurus_ok(names, calls=None):
    def _get(url, params=None, timeout=None, headers=None):
        if calls is not None:
            calls.append(url)
        payload = {
            "Cellosaurus": {
                "cell-line-list": [{"name-list": [{"value": n} for n in names]}]
            }
        }
        return _FakeResponse(payload)

    return _get


def test_cvcl_routes_to_cellosaurus_and_matches(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("requests.get", _cellosaurus_ok(["HeLa", "Hela"], calls=calls))
    assert check_ontology_terms(_term("CVCL:0030", "HeLa")) == []
    # accession form uses an underscore, and the Cellosaurus host (not OLS).
    assert calls == ["https://api.cellosaurus.org/cell-line/CVCL_0030"]


def test_cvcl_label_mismatch_is_error(monkeypatch):
    monkeypatch.setattr("requests.get", _cellosaurus_ok(["HeLa"]))
    issues = check_ontology_terms(_term("CVCL:0030", "Not HeLa"))
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert "Cellosaurus" in issues[0].message


def test_cvcl_missing_accession_is_error(monkeypatch):
    def _get(url, params=None, timeout=None, headers=None):
        return _FakeResponse({}, status_code=404)

    monkeypatch.setattr("requests.get", _get)
    issues = check_ontology_terms(_term("CVCL:9999999", "ghost line"))
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert "does not resolve" in issues[0].message


def test_cvcl_server_error_is_warning(monkeypatch):
    def _get(url, params=None, timeout=None, headers=None):
        return _FakeResponse({}, status_code=503)

    monkeypatch.setattr("requests.get", _get)
    issues = check_ontology_terms(_term("CVCL:0030", "HeLa"))
    assert len(issues) == 1
    assert issues[0].severity == Severity.WARNING


# --- retry / backoff / User-Agent -------------------------------------------


def test_transient_status_is_retried_then_succeeds(monkeypatch):
    responses = [
        _FakeResponse({}, status_code=503),  # transient — retried
        _FakeResponse({"_embedded": {"terms": [{"label": "Homo sapiens"}]}}),
    ]
    attempts: list[str] = []

    def _get(url, params=None, timeout=None, headers=None):
        attempts.append(url)
        return responses[len(attempts) - 1]

    monkeypatch.setattr("requests.get", _get)
    assert check_ontology_terms(_term("NCBITaxon:9606", "Homo sapiens")) == []
    assert len(attempts) == 2  # first 503, then success


def test_persistent_transient_status_exhausts_retries_and_warns(monkeypatch):
    attempts: list[str] = []

    def _get(url, params=None, timeout=None, headers=None):
        attempts.append(url)
        return _FakeResponse({}, status_code=503)

    monkeypatch.setattr("requests.get", _get)
    issues = check_ontology_terms(_term("NCBITaxon:9606", "Homo sapiens"))
    assert len(issues) == 1
    assert issues[0].severity == Severity.WARNING
    assert len(attempts) == ontology_lookup._MAX_ATTEMPTS  # all attempts used


def test_request_sends_descriptive_user_agent(monkeypatch):
    seen_headers: list[dict] = []

    def _get(url, params=None, timeout=None, headers=None):
        seen_headers.append(headers or {})
        return _FakeResponse({"_embedded": {"terms": [{"label": "Homo sapiens"}]}})

    monkeypatch.setattr("requests.get", _get)
    check_ontology_terms(_term("NCBITaxon:9606", "Homo sapiens"))
    assert seen_headers, "expected at least one request"
    assert "dca-helpers" in seen_headers[0].get("User-Agent", "")


# --- subtree / ancestor membership ------------------------------------------


def _imaging(term_id, label):
    return {
        "study_component": {
            "imaging_method": [{"ontology_term_id": term_id, "label": label}]
        }
    }


def _prep(term_id, label):
    return {
        "specimen": {
            "preparation_method": [{"ontology_term_id": term_id, "label": label}]
        }
    }


def test_imaging_method_in_subtree_passes(monkeypatch):
    iri = "http://purl.obolibrary.org/obo/FBbi_00000251"
    monkeypatch.setattr(
        "requests.get",
        _fake_ols(
            {iri: [{"label": "confocal microscopy"}]},
            ancestor_ids=[
                "FBbi:00000222",
                "FBbi:00000241",
            ],  # imaging method / microscopy
        ),
    )
    assert check_ontology_terms(_imaging("FBbi:00000251", "confocal microscopy")) == []


def test_preparation_method_in_subtree_passes(monkeypatch):
    iri = "http://purl.obolibrary.org/obo/FBbi_00000026"
    monkeypatch.setattr(
        "requests.get",
        _fake_ols(
            {iri: [{"label": "sectioned tissue"}]},
            ancestor_ids=["FBbi:00000001"],  # sample preparation method
        ),
    )
    assert check_ontology_terms(_prep("FBbi:00000026", "sectioned tissue")) == []


def test_wrong_subtree_is_error_even_when_label_matches(monkeypatch):
    # A sample-prep term (label correct) dropped into imaging_method: it resolves
    # and its label matches, but it is not under "imaging method".
    iri = "http://purl.obolibrary.org/obo/FBbi_00000010"
    monkeypatch.setattr(
        "requests.get",
        _fake_ols(
            {iri: [{"label": "formaldehyde fixed tissue"}]},
            ancestor_ids=["FBbi:00000001"],  # sample-prep branch, not imaging
        ),
    )
    issues = check_ontology_terms(
        _imaging("FBbi:00000010", "formaldehyde fixed tissue")
    )
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert "descendant of 'FBbi:00000222'" in issues[0].message


def test_subtree_network_failure_is_warning(monkeypatch):
    def _get(url, params=None, timeout=None, headers=None):
        if url.endswith("/ancestors"):
            raise ConnectionError("OLS unreachable")
        return _FakeResponse({"_embedded": {"terms": [{"label": "sectioned tissue"}]}})

    monkeypatch.setattr("requests.get", _get)
    issues = check_ontology_terms(_prep("FBbi:00000026", "sectioned tissue"))
    assert len(issues) == 1
    assert issues[0].severity == Severity.WARNING
    assert "subtree" in issues[0].message


def test_unconstrained_field_makes_no_ancestor_call(monkeypatch):
    iri = "http://purl.obolibrary.org/obo/NCBITaxon_9606"
    calls: list[str] = []
    monkeypatch.setattr(
        "requests.get", _fake_ols({iri: [{"label": "Homo sapiens"}]}, calls=calls)
    )
    assert check_ontology_terms(_term("NCBITaxon:9606", "Homo sapiens")) == []
    assert not any(url.endswith("/ancestors") for url in calls)


def test_term_equal_to_required_ancestor_skips_ancestor_call(monkeypatch):
    iri = "http://purl.obolibrary.org/obo/FBbi_00000001"
    calls: list[str] = []
    monkeypatch.setattr(
        "requests.get",
        _fake_ols({iri: [{"label": "sample preparation method"}]}, calls=calls),
    )
    assert (
        check_ontology_terms(_prep("FBbi:00000001", "sample preparation method")) == []
    )
    assert not any(url.endswith("/ancestors") for url in calls)


# --- schema drift -----------------------------------------------------------
#
# The routing table is a hand-maintained mirror of which registry hosts which
# prefix, so it drifts the moment the spec admits a new ontology. It drifted
# exactly once already: v0.2.4 broadened `ontology_term_id` coverage with the
# multi-organism anatomy / stage ontologies (ZFA, ZFS, MmusDv, FBdv, WBls) and
# this module kept the pre-v0.2.4 allowlist, which routes them nowhere. A
# missing prefix is silently *skipped* rather than errored, so nothing else in
# the suite would notice. Follows the drift-detection idiom in
# tests/validation/versions/v0_2/test_schema_drift.py.

_PATTERN_SENTINELS = frozenset({"unknown", "na", "unavailable"})


def _schema_ontology_prefixes(root_dir) -> set[str]:
    """Every CURIE prefix an ``ontology_term_id`` pattern in the v0.2 schema admits."""
    schema = yaml.safe_load(
        (root_dir / "schema" / "v0.2" / "dca_experimental_metadata.yaml").read_text(
            encoding="utf-8"
        )
    )
    prefixes: set[str] = set()
    for definition in (schema.get("classes") or {}).values():
        slot = ((definition.get("slot_usage") or {}).get("ontology_term_id")) or {}
        pattern = slot.get("pattern")
        if not pattern:
            continue
        # Pull the `Prefix:` tokens out of the alternation, dropping the
        # sentinels (which stand in for a CURIE rather than naming a term).
        prefixes |= {
            token
            for token in re.findall(r"([A-Za-z][A-Za-z0-9]*):", pattern)
            if token not in _PATTERN_SENTINELS
        }
    return prefixes


def test_ols_prefixes_cover_schema_ontology_patterns(root_dir):
    """Every prefix the schema admits must route to OLS or Cellosaurus."""
    routed = set(ontology_lookup._OLS_PREFIXES) | {ontology_lookup._CELLOSAURUS_PREFIX}
    unrouted = _schema_ontology_prefixes(root_dir) - routed
    assert not unrouted, (
        f"ontology_term_id prefixes {sorted(unrouted)} are admitted by the v0.2 "
        "schema but route to no registry, so --check-ontology skips them "
        "silently. Add each to _OLS_PREFIXES (OLS4-hosted) or route it "
        "explicitly."
    )


def test_ols_prefixes_are_all_expandable():
    """Every routed OLS prefix must expand via the schema's prefix table.

    A prefix in the allowlist with no `prefixes:` entry expands to None, which
    `_lookup_term` reports as "does not resolve" — an ERROR on valid metadata.
    """
    unexpandable = [
        prefix
        for prefix in ontology_lookup._OLS_PREFIXES
        if curie_to_iri(f"{prefix}:00000001") is None
    ]
    assert not unexpandable, (
        f"{sorted(unexpandable)} are routed to OLS but absent from the schema's "
        "`prefixes:` block, so every such term would be reported unresolvable."
    )


# Real CURIE/label pairs, verified against OLS4 at the time of writing, so the
# fixtures stay meaningful to a reader even though the transport is mocked.
@pytest.mark.parametrize(
    ("term_id", "label", "iri"),
    [
        (
            "ZFA:0000047",
            "peripheral olfactory organ",
            "http://purl.obolibrary.org/obo/ZFA_0000047",
        ),
        (
            "ZFS:0000029",
            "pharyngula:prim-5",
            "http://purl.obolibrary.org/obo/ZFS_0000029",
        ),
        (
            "MmusDv:0000036",
            "Theiler stage 27",
            "http://purl.obolibrary.org/obo/MmusDv_0000036",
        ),
        (
            "FBdv:00005342",
            "prepupal stage",
            "http://purl.obolibrary.org/obo/FBdv_00005342",
        ),
        ("WBls:0000024", "L1 larva Ce", "http://purl.obolibrary.org/obo/WBls_0000024"),
    ],
)
def test_v0_2_4_organism_ontologies_are_checked_not_skipped(
    monkeypatch, term_id, label, iri
):
    """The v0.2.4 stage / anatomy ontologies resolve rather than falling through.

    A skipped prefix and a matching label both report zero issues, so assert on
    the request actually being made — that is what distinguishes the two.
    """
    calls: list[str] = []
    monkeypatch.setattr(
        "requests.get", _fake_ols({iri: [{"label": label}]}, calls=calls)
    )
    assert check_ontology_terms(_term(term_id, label)) == []
    assert calls, f"{term_id} was skipped — no lookup was attempted"


def test_v0_2_4_organism_ontology_label_mismatch_is_an_error(monkeypatch):
    iri = "http://purl.obolibrary.org/obo/ZFS_0000029"
    monkeypatch.setattr(
        "requests.get", _fake_ols({iri: [{"label": "pharyngula:prim-5"}]})
    )
    issues = check_ontology_terms(_term("ZFS:0000029", "adult stage"))
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR


def test_both_ontology_passes_share_one_traversal():
    """The offline and network passes must use the same walk, not two copies.

    They each carried a byte-identical private copy before this was extracted, so
    the definition of "what counts as an ontology term" could drift in one
    without the other. Identity, not equivalence: a re-added local copy would
    behave the same on the day it was written, which is exactly why the
    duplication survived unnoticed the first time.
    """
    canonical = _ontology_terms.iter_ontology_terms
    assert experimental_metadata.iter_ontology_terms is canonical
    assert ontology_lookup.iter_ontology_terms is canonical


# --- Term -------------------------------------------------------------------
#
# One place that knows how to take a CURIE apart. Before this existed the same
# splitting was open-coded as term.split(":", 1)[0], term.replace(":", "_") and
# term.startswith("CVCL:") across two modules.


def test_term_parses_a_curie():
    term = Term.parse("FBbi:00000251")
    assert (term.prefix, term.local) == ("FBbi", "00000251")
    assert term.is_curie and not term.is_sentinel


@pytest.mark.parametrize("value", ["unknown", "na", "unavailable"])
def test_term_recognises_sentinels(value):
    term = Term.parse(value)
    assert term.is_sentinel
    assert not term.is_curie, "a sentinel stands in for a CURIE, it is not one"


@pytest.mark.parametrize("value", ["no-colon-here", "FBbi:", ":00000251", ""])
def test_term_rejects_non_curies(value):
    assert not Term.parse(value).is_curie


def test_term_cellosaurus_accession_uses_underscore():
    """Cellosaurus accessions are CVCL_0030; the schema stores CVCL:0030."""
    assert Term.parse("CVCL:0030").cellosaurus_accession == "CVCL_0030"


def test_term_ontology_is_lowercased_for_ols():
    assert Term.parse("FBbi:00000251").ontology == "fbbi"


def test_term_iri_expands_and_declines():
    assert Term.parse("FBbi:00000251").iri() == (
        "http://purl.obolibrary.org/obo/FBbi_00000251"
    )
    assert Term.parse("NOTAPREFIX:123").iri() is None
    assert Term.parse("unknown").iri() is None


def test_sentinel_set_is_shared_with_the_offline_pass():
    """One definition of "this value stands in for a CURIE", not two.

    `experimental_metadata` carried its own identical copy (added with the v0.2.4
    cross-field rules) alongside this module's. Same drift shape as the duplicated
    traversal: identity, so a re-added local copy fails here.
    """
    assert experimental_metadata.SENTINELS is _ontology_terms.SENTINELS
    assert _ontology_terms.SENTINELS == {"unknown", "na", "unavailable"}


def test_registry_failure_is_logged_as_well_as_reported(monkeypatch, caplog):
    """An unreachable registry is a warning in the report *and* in the log.

    The report hides warnings unless --show-warnings is passed, so a long run
    against a registry that has gone down would otherwise look healthy until it
    finished.
    """

    def _get(url, params=None, timeout=None, headers=None):
        raise ConnectionError("OLS unreachable")

    monkeypatch.setattr("requests.get", _get)
    with caplog.at_level("WARNING", logger="dca_helpers.validation.ontology_lookup"):
        issues = check_ontology_terms(_term("NCBITaxon:9606", "Homo sapiens"))

    assert len(issues) == 1 and issues[0].severity == Severity.WARNING
    assert "NCBITaxon:9606" in caplog.text
    assert "biosample.organism.0" in caplog.text, "log should locate the term"


# --- subtree-constraint drift ------------------------------------------------
#
# _SUBTREE_CONSTRAINTS is hand-maintained the same way _OLS_PREFIXES was, and
# _OLS_PREFIXES is what silently drifted when v0.2.4 broadened ontology coverage.
# Same failure shape applies here: a spec version that adds a subtree-constrained
# field, or moves an existing field's required ancestor, leaves the field with no
# branch check and nothing fails.

_SUBTREE_IN_DESCRIPTION = re.compile(r"descendants?\s+of\s+``([A-Za-z]+:\d+)``")

# Slots whose description states a subtree but which deliberately are NOT in
# _SUBTREE_CONSTRAINTS, with the reason. The table maps a slot to ONE ancestor
# applied to every term in it, so a rule that is conditional cannot live there.
_CONDITIONAL_SUBTREE_SLOTS = {
    # `tissue` admits UBERON / CL / CVCL / ZFA, and the ZFA:0100000 subtree rule
    # applies only to ZFA terms in the anatomy role (it keys off `tissue_type`,
    # and the cell-type role uses the ZFA:0009000 branch instead). Adding it flat
    # would reject every UBERON, CL and CVCL tissue term. Left unchecked rather
    # than checked wrongly; see the slot description in the schema.
    "tissue",
}


def _schema_subtree_slots(root_dir) -> dict[str, str]:
    """Slots whose schema description states a required ancestor CURIE."""
    schema = yaml.safe_load(
        (root_dir / "schema" / "v0.2" / "dca_experimental_metadata.yaml").read_text(
            encoding="utf-8"
        )
    )
    found: dict[str, str] = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict) and isinstance(
                    value.get("description"), str
                ):
                    # Descriptions are YAML folded scalars; collapse the wrapping
                    # so a CURIE split across two lines still matches.
                    match = _SUBTREE_IN_DESCRIPTION.search(
                        " ".join(value["description"].split())
                    )
                    if match:
                        found[key] = match.group(1)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return found


def test_schema_stated_subtrees_are_constrained_or_explicitly_exempt(root_dir):
    """A slot the schema says must sit under an ancestor is checked, or exempt on purpose."""
    stated = _schema_subtree_slots(root_dir)
    unhandled = {
        slot: ancestor
        for slot, ancestor in stated.items()
        if slot not in ontology_lookup._SUBTREE_CONSTRAINTS
        and slot not in _CONDITIONAL_SUBTREE_SLOTS
    }
    assert not unhandled, (
        f"the schema states a required ancestor for {sorted(unhandled)} but "
        "_SUBTREE_CONSTRAINTS does not cover them, so --check-ontology gives those "
        "fields no branch check. Add each to _SUBTREE_CONSTRAINTS, or to "
        "_CONDITIONAL_SUBTREE_SLOTS with the reason it cannot be a flat mapping."
    )


def test_constrained_slot_ancestors_match_the_schema(root_dir):
    """Where the schema names the ancestor, the table must agree with it.

    Catches the subtler half: the field stays constrained, but to the wrong node
    after a schema revision. Only covers slots whose description states the
    ancestor — `imaging_method`'s is a code-side choice (the schema says only
    "FBbi only"), pinned instead by the live test on FBbi:00000222's label.
    """
    stated = _schema_subtree_slots(root_dir)
    mismatched = {
        slot: (ontology_lookup._SUBTREE_CONSTRAINTS[slot], ancestor)
        for slot, ancestor in stated.items()
        if slot in ontology_lookup._SUBTREE_CONSTRAINTS
        and ontology_lookup._SUBTREE_CONSTRAINTS[slot] != ancestor
    }
    assert not mismatched, (
        "_SUBTREE_CONSTRAINTS disagrees with the schema description "
        f"(slot: (table, schema)): {mismatched}"
    )


def test_constrained_slots_still_exist_in_the_schema(root_dir):
    """A renamed slot would silently disable its branch check.

    `_required_ancestor` matches on the slot name appearing in a term's path, so
    a slot renamed in the schema stops matching and the constraint quietly stops
    applying to anything.
    """
    schema = yaml.safe_load(
        (root_dir / "schema" / "v0.2" / "dca_experimental_metadata.yaml").read_text(
            encoding="utf-8"
        )
    )
    slot_names: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict) and (
                    "range" in value or "description" in value
                ):
                    slot_names.add(key)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    missing = set(ontology_lookup._SUBTREE_CONSTRAINTS) - slot_names
    assert not missing, (
        f"{sorted(missing)} are in _SUBTREE_CONSTRAINTS but name no slot in the "
        "v0.2 schema, so their subtree check can never fire."
    )


def test_concurrent_requests_are_capped(monkeypatch):
    """Breadth is bounded, not just per-request retry.

    `validate` nests ThreadPoolExecutors and memoization only dedupes identical
    CURIEs, so a cold-cache run over many *distinct* terms could otherwise put one
    request per term in flight against a free, no-published-limit service.
    """
    import threading as _threading

    in_flight = 0
    peak = 0
    lock = _threading.Lock()

    def _get(url, params=None, timeout=None, headers=None):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            _threading.Event().wait(0.01)  # hold the slot briefly
            return _FakeResponse({"_embedded": {"terms": [{"label": "x"}]}})
        finally:
            with lock:
                in_flight -= 1

    monkeypatch.setattr("requests.get", _get)
    # Distinct CURIEs, so memoization cannot collapse them into one request.
    terms = [
        {"ontology_term_id": f"NCBITaxon:{9000 + i}", "label": "x"} for i in range(40)
    ]
    with ThreadPoolExecutor(max_workers=40) as pool:
        list(
            pool.map(
                lambda t: check_ontology_terms({"biosample": {"organism": [t]}}), terms
            )
        )

    assert peak <= ontology_lookup._MAX_CONCURRENT_REQUESTS, (
        f"peaked at {peak} concurrent requests, above the "
        f"{ontology_lookup._MAX_CONCURRENT_REQUESTS} cap"
    )
    assert peak > 1, "test did not actually exercise concurrency"


# --- negative caching --------------------------------------------------------
#
# lru_cache stores return values, not exceptions, so without a failure memory a
# dead registry is re-retried per unique CURIE. With the concurrency cap those
# retries serialise, turning a slow run into an apparently hung one.


def _always_fails(calls):
    def _get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        raise ConnectionError("registry down")

    return _get


def test_dead_registry_is_probed_once_then_short_circuited(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("requests.get", _always_fails(calls))

    first = check_ontology_terms(_term("NCBITaxon:9606", "Homo sapiens"))
    after_first = len(calls)
    assert after_first == ontology_lookup._MAX_ATTEMPTS, "first term spends the budget"
    assert first[0].severity == Severity.WARNING

    # A *different* CURIE, so memoization cannot be what saves us.
    second = check_ontology_terms(_term("UBERON:0000955", "brain"))
    assert len(calls) == after_first, "second term retried a registry known to be down"
    assert len(second) == 1
    assert second[0].severity == Severity.WARNING, "still unverified, never invalid"
    assert "unreachable in the last" in second[0].message


def test_failure_cache_expires_and_reprobes(monkeypatch):
    """On expiry the next caller probes for real — a TTL cache, not a breaker."""
    calls: list[str] = []
    monkeypatch.setattr("requests.get", _always_fails(calls))
    monkeypatch.setattr(ontology_lookup, "_FAILURE_TTL_SECONDS", 0.0)

    check_ontology_terms(_term("NCBITaxon:9606", "Homo sapiens"))
    after_first = len(calls)
    check_ontology_terms(_term("UBERON:0000955", "brain"))
    assert len(calls) > after_first, "expired entry should not suppress the next probe"


def test_recovery_clears_the_cache_immediately(monkeypatch):
    """A definite response means the registry is up; the entry goes at once.

    Without eager clearing a recovered registry would stay suppressed for the
    rest of the TTL, so a run that outlives a blip would report terms unverified
    long after the service came back.
    """
    down = {"value": True}

    def _get(url, params=None, timeout=None, headers=None):
        if down["value"]:
            raise ConnectionError("registry down")
        return _FakeResponse({"_embedded": {"terms": [{"label": "brain"}]}})

    monkeypatch.setattr("requests.get", _get)
    check_ontology_terms(_term("NCBITaxon:9606", "Homo sapiens"))
    assert ontology_lookup._registry_failures, "failure should have been recorded"

    # Expire the entry so the next call probes, then let the probe succeed.
    monkeypatch.setattr(ontology_lookup, "_FAILURE_TTL_SECONDS", 0.0)
    down["value"] = False
    assert check_ontology_terms(_term("UBERON:0000955", "brain")) == []
    assert (
        not ontology_lookup._registry_failures
    ), "a successful response must clear the entry, not leave it to expire"


def test_failure_cache_is_keyed_per_registry(monkeypatch):
    """OLS being down must not suppress Cellosaurus lookups, or vice versa."""
    calls: list[str] = []

    def _get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        if "ols4" in url:
            raise ConnectionError("OLS down")
        return _FakeResponse(
            {"Cellosaurus": {"cell-line-list": [{"name-list": [{"value": "HeLa"}]}]}}
        )

    monkeypatch.setattr("requests.get", _get)
    check_ontology_terms(_term("NCBITaxon:9606", "Homo sapiens"))
    ols_calls = len(calls)

    issues = check_ontology_terms(_term("CVCL:0030", "HeLa"))
    assert issues == [], "Cellosaurus should be unaffected by an OLS outage"
    assert any(
        "cellosaurus" in url for url in calls[ols_calls:]
    ), "Cellosaurus was wrongly short-circuited by the OLS failure"
