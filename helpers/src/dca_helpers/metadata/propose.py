"""Taxonomy proposals for DCA fields a pipeline's finalize step could not resolve.

Rules look a hinted name or term up in Cellosaurus or OLS. A proposal is
a suggestion for a human to confirm. Nothing here writes to a store.
"""

import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import requests

from dca_helpers.validation import ontology_lookup
from dca_helpers.validation.ontology_lookup import (
    CELLOSAURUS_BASE_URL,
    OLS_BASE_URL,
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


@dataclass(frozen=True)
class Proposal:
    """A suggested value for one DCA field. ``value`` is None when no rule could fill it."""

    field: str
    value: Any
    source: str
    rule: str


@functools.lru_cache(maxsize=4096)
def search_cellosaurus(name: str) -> tuple[tuple[str, str], ...]:
    """Return ``(curie, label)`` for Cellosaurus lines named exactly ``name``.

    A line matches when its identifier or a synonym equals ``name``
    (case-insensitive). The identifier and synonym fields are searched in two
    requests, because Cellosaurus identifiers and synonyms differ (A549 is a
    synonym of the identifier ``A-549``). The label is the matching name.
    Derivatives that only contain ``name`` are never returned. Results are cached
    per name, since ``cell_line_id`` and ``tissue`` share one hint; a failed lookup
    raises and is not cached.
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
    return tuple(found.items())[:_MAX_CANDIDATES]


def _ols_names(curie: str) -> list[tuple[str, str]]:
    iri = Term.parse(curie).iri()
    record = lookup_ols(iri) if iri else None
    return [(curie, name) for name in record.names[:1]] if record else []


def _with_alternatives(label: str, candidates: Sequence[tuple[str, str]]) -> str:
    curie = candidates[0][0]
    source = f"{label} {curie}"
    if len(candidates) > 1:
        source += " (also: " + ", ".join(f"{c} {n}" for c, n in candidates[1:]) + ")"
    return source


def _lookup(
    field: str,
    rule: str,
    hint_key: str,
    host: str,
    hints: Mapping[str, str],
    online: bool,
    search: Callable[[str], Sequence[tuple[str, str]]],
) -> tuple[Proposal | None, Sequence[tuple[str, str]]]:
    """Run ``search`` on a hint. Returns a degraded Proposal, or None plus the candidates."""
    if hint_key not in hints:
        return Proposal(field, None, f"no hint: {hint_key}", rule), []
    if not online:
        return Proposal(field, None, "lookup skipped (offline)", rule), []
    try:
        candidates = search(hints[hint_key])
    except _LOOKUP_ERRORS as error:
        logger.warning("%s lookup for %s failed: %s", host, field, error)
        return Proposal(field, None, f"lookup failed: {host}", rule), []
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
            "biosample.cell_line_name",
            _registry_of(CELLOSAURUS_BASE_URL),
            hints,
            online,
            search_cellosaurus,
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
            hint_key,
            _registry_of(OLS_BASE_URL),
            hints,
            online,
            _ols_names,
        )
        return failed or Proposal(field, found[0][1], f"OLS {found[0][0]}", "ols_label")

    return None


def propose(
    unresolved: Sequence[str],
    *,
    hints: Mapping[str, str],
    online: bool = True,
) -> list[Proposal]:
    """Propose a value for each unresolved field a rule covers.

    Fields no rule covers get no proposal. A lookup rule that cannot answer (hint
    missing, offline, registry error, no candidates) returns a ``Proposal`` with
    ``value=None`` and the reason in ``source``. This function does not raise on
    lookup failure.
    """
    proposals = (_propose_one(field, hints, online) for field in unresolved)
    return [p for p in proposals if p is not None]
