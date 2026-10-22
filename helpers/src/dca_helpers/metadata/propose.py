"""Proposals for DCA fields a pipeline's finalize step could not resolve.

Rules look a hinted name or term up in Cellosaurus, NCBITaxon, OLS, Ensembl or
PubChem, or compute a compound's Morgan fingerprint from the PubChem record of a
hinted CID. A proposal is a suggestion for a human to confirm. Nothing here writes
to a store.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import requests

from dca_helpers.metadata.perturbation_ids import (
    ensembl_from_ncbi_gene,
    morgan_fingerprint,
    pubchem_smiles,
    rdkit_version,
    search_ensembl,
    search_pubchem,
)
from dca_helpers.validation import ontology_lookup
from dca_helpers.validation.ontology_lookup import (
    CELLOSAURUS_BASE_URL,
    ENSEMBL_BASE_URL,
    OLS_BASE_URL,
    PUBCHEM_BASE_URL,
    _get_with_retry,
    _registry_of,
    lookup_ols,
)
from dca_helpers.validation.versions.v0_2._ontology_terms import Term

logger = logging.getLogger(__name__)

_LOOKUP_ERRORS = (
    requests.exceptions.RequestException,
    OSError,
    ontology_lookup._TransientHTTPError,
    ontology_lookup._RegistryRecentlyUnreachable,
)
_MAX_CANDIDATES = 5
_SEARCH_ROWS = 100

_GENETIC = "perturbation.definition.genetic."
_CHEMICAL = "perturbation.definition.chemical."
_SMILES_HINT = _CHEMICAL + "smiles"
_PUBCHEM_CID_HINT = _CHEMICAL + "pubchem_cid"
#: PubChem namespace per hint, in the order the ``pubchem`` rule tries them.
_PUBCHEM_HINTS = {
    _CHEMICAL + "inchikey": "inchikey",
    _SMILES_HINT: "smiles",
    "perturbation.definition.name": "name",
}

Search = Callable[[str], list[tuple[str, str]]]


@dataclass(frozen=True)
class Proposal:
    """A suggested value for one DCA field. ``value`` is None when no rule could fill it."""

    field: str
    value: Any
    source: str
    rule: str


def search_cellosaurus(name: str) -> list[tuple[str, str]]:
    """Return ``(curie, label)`` for Cellosaurus lines named exactly ``name``.

    A line matches when its identifier or a synonym equals ``name``
    (case-insensitive). The identifier and synonym fields are searched in two
    requests, because Cellosaurus identifiers and synonyms differ (A549 is a
    synonym of the identifier ``A-549``). The label is the matching name.
    Derivatives that only contain ``name`` are never returned.
    """
    target = name.strip().lower()
    found: dict[str, str] = {}
    for search_field in ("id", "sy"):
        resp = _get_with_retry(
            f"{CELLOSAURUS_BASE_URL}/search/cell-line",
            {
                "q": f'{search_field}:"{name}"',
                "format": "json",
                "fld": ["ac", "id", "sy"],
                "rows": _SEARCH_ROWS,
            },
        )
        resp.raise_for_status()
        for line in resp.json().get("Cellosaurus", {}).get("cell-line-list", []):
            accession = next(
                (a["value"] for a in line.get("accession-list", []) if "value" in a),
                None,
            )
            label = next(
                (
                    n["value"]
                    for n in line.get("name-list", [])
                    if "value" in n and n["value"].strip().lower() == target
                ),
                None,
            )
            if accession and label:
                found.setdefault(accession.replace("_", ":", 1), label)
    return list(found.items())[:_MAX_CANDIDATES]


def search_ncbitaxon(name: str) -> list[tuple[str, str]]:
    """Return ``(curie, label)`` NCBITaxon candidates for an exact organism name."""
    resp = _get_with_retry(
        f"{OLS_BASE_URL}/search",
        {"q": name, "ontology": "ncbitaxon", "exact": "true", "rows": 5},
    )
    resp.raise_for_status()
    docs = resp.json().get("response", {}).get("docs", [])
    return [
        (d["obo_id"], d["label"]) for d in docs if d.get("obo_id") and d.get("label")
    ]


def _ols_names(curie: str) -> list[tuple[str, str]]:
    iri = Term.parse(curie).iri()
    record = lookup_ols(iri) if iri else None
    return [(curie, name) for name in record.names[:1]] if record else []


def _with_alternatives(label: str, candidates: Sequence[tuple[str, str]]) -> str:
    curie = candidates[0][0]
    source = f"{label} {curie}"
    if len(candidates) > 1:
        source += (
            " (also: " + ", ".join(f"{c} {n}".rstrip() for c, n in candidates[1:]) + ")"
        )
    return source


def _unlabelled(curies: list[str]) -> list[tuple[str, str]]:
    return [(curie, "") for curie in curies[:_MAX_CANDIDATES]]


def _ensembl_searches(hints: Mapping[str, str]) -> dict[str, Search]:
    """Searches for the ``ensembl`` rule: NCBI Gene id first, then gene symbol."""
    species = hints.get("species", "homo_sapiens")

    def from_ncbi_gene(value: str) -> list[tuple[str, str]]:
        return _unlabelled(ensembl_from_ncbi_gene(value, species=species))

    def from_symbol(value: str) -> list[tuple[str, str]]:
        return _unlabelled(search_ensembl(value, species=species))

    return {
        _GENETIC + "ncbi_gene_id": from_ncbi_gene,
        _GENETIC + "gene_symbol": from_symbol,
    }


def _pubchem_search(namespace: str) -> Search:
    return lambda value: _unlabelled(search_pubchem(**{namespace: value}))


def _lookup(
    field: str,
    rule: str,
    host: str,
    hints: Mapping[str, str],
    online: bool,
    searches: Mapping[str, Search],
) -> tuple[Proposal | None, list[tuple[str, str]]]:
    """Run the search for the first hint key in ``searches`` that ``hints`` has.

    Returns a degraded Proposal, or None plus the candidates.
    """
    hint_key = next((key for key in searches if key in hints), None)
    if hint_key is None:
        return Proposal(field, None, "no hint: " + " or ".join(searches), rule), []
    if not online:
        return Proposal(field, None, "lookup skipped (offline)", rule), []
    try:
        candidates = searches[hint_key](hints[hint_key])
    except _LOOKUP_ERRORS as error:
        logger.warning("%s lookup for %s failed: %s", host, field, error)
        return Proposal(field, None, f"lookup failed: {host}", rule), []
    except ValueError as error:
        logger.warning("hint %s for %s is invalid: %s", hint_key, field, error)
        return Proposal(field, None, f"invalid hint: {hint_key}", rule), []
    if not candidates:
        return Proposal(field, None, "no match", rule), []
    return None, candidates


def _propose_one(
    field: str,
    hints: Mapping[str, str],
    online: bool,
) -> Proposal | None:
    if field in ("biosample.cell_line_id", "biosample.tissue"):
        failed, found = _lookup(
            field,
            "cellosaurus",
            _registry_of(CELLOSAURUS_BASE_URL),
            hints,
            online,
            {"biosample.cell_line_name": search_cellosaurus},
        )
        if failed:
            return failed
        curie, label = found[0]
        value = (
            curie
            if field == "biosample.cell_line_id"
            else [{"ontology_term_id": curie, "label": label}]
        )
        return Proposal(
            field, value, _with_alternatives("Cellosaurus", found), "cellosaurus"
        )

    if field.endswith(".label"):
        hint_key = field.removesuffix("label") + "ontology_term_id"
        failed, found = _lookup(
            field,
            "ols_label",
            _registry_of(OLS_BASE_URL),
            hints,
            online,
            {hint_key: _ols_names},
        )
        return failed or Proposal(field, found[0][1], f"OLS {found[0][0]}", "ols_label")

    if field == "perturbation.definition.perturbation_id":
        failed, found = _lookup(
            field,
            "ncbitaxon",
            _registry_of(OLS_BASE_URL),
            hints,
            online,
            {"perturbation.definition.name": search_ncbitaxon},
        )
        if failed:
            return failed
        return Proposal(
            field, found[0][0], _with_alternatives("NCBITaxon", found), "ncbitaxon"
        )

    if field == _GENETIC + "ensembl_gene_id":
        failed, found = _lookup(
            field,
            "ensembl",
            _registry_of(ENSEMBL_BASE_URL),
            hints,
            online,
            _ensembl_searches(hints),
        )
        if failed:
            return failed
        return Proposal(
            field, found[0][0], _with_alternatives("Ensembl", found), "ensembl"
        )

    if field == _CHEMICAL + "pubchem_cid":
        failed, found = _lookup(
            field,
            "pubchem",
            _registry_of(PUBCHEM_BASE_URL),
            hints,
            online,
            {key: _pubchem_search(ns) for key, ns in _PUBCHEM_HINTS.items()},
        )
        if failed:
            return failed
        return Proposal(
            field, found[0][0], _with_alternatives("PubChem", found), "pubchem"
        )

    if field == _CHEMICAL + "morgan_fingerprint":
        return _propose_morgan(field, hints, online)

    return None


def _propose_morgan(field: str, hints: Mapping[str, str], online: bool) -> Proposal:
    """Fingerprint the PubChem record SMILES of the hinted CID.

    A CID hint of ``NA`` proposes ``NA``, offline too: a compound with no PubChem
    record has no record to fingerprint. ``source`` names the RDKit version, which
    belongs in the field's provenance.
    """
    cid = hints.get(_PUBCHEM_CID_HINT)
    if cid == "NA":
        return Proposal(field, "NA", f"{_PUBCHEM_CID_HINT} is NA", "morgan")
    try:
        version = rdkit_version()
    except ImportError:
        return Proposal(
            field, None, "rdkit not installed (dca_helpers[chem])", "morgan"
        )
    failed, found = _lookup(
        field,
        "morgan",
        _registry_of(PUBCHEM_BASE_URL),
        hints,
        online,
        {_PUBCHEM_CID_HINT: lambda value: [(pubchem_smiles(value), "")]},
    )
    if failed:
        return failed
    try:
        value = morgan_fingerprint(found[0][0])
    except ValueError:
        return Proposal(field, None, "invalid SMILES", "morgan")
    return Proposal(
        field,
        value,
        f"RDKit {version} Morgan radius 2, 2048 bits, from {cid} record SMILES",
        "morgan",
    )


def propose(
    unresolved: Sequence[str],
    *,
    hints: Mapping[str, str],
    online: bool = True,
) -> list[Proposal]:
    """Propose a value for each unresolved field a rule covers.

    Fields no rule covers get no proposal. A lookup rule that cannot answer (hint
    missing or invalid, offline, registry error, no candidates) returns a
    ``Proposal`` with ``value=None`` and the reason in ``source``. The ``morgan``
    rule is a lookup too (the PubChem record SMILES of the CID hint) and degrades the
    same way, and also when RDKit is not installed or cannot parse the SMILES; a CID
    hint of ``NA`` proposes ``NA`` without a lookup. This function does not raise on
    lookup failure.
    """
    proposals = (_propose_one(field, hints, online) for field in unresolved)
    return [p for p in proposals if p is not None]
