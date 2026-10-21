"""Proposals for DCA fields a pipeline's finalize step could not resolve.

A proposal is a suggestion for a human to confirm. Nothing here writes to a store.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import requests

from dca_helpers.metadata.store_facts import name_date
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
_VIRTUAL_STAIN_TARGETS = {
    "nuclei_prediction": "nucleus",
    "membrane_prediction": "plasma membrane",
}
_MAX_CANDIDATES = 5
_LABELFREE_PREFIXES = ("Phase", "BF", "Brightfield")


@dataclass(frozen=True)
class Proposal:
    """A suggested value for one DCA field. ``value`` is None when no rule could fill it."""

    field: str
    value: Any
    source: str
    rule: str


def search_cellosaurus(name: str) -> list[tuple[str, str]]:
    """Return ``(curie, label)`` Cellosaurus candidates for a cell line name, best first.

    Cellosaurus identifiers and synonyms differ (A549 is the synonym of the
    identifier ``A-549``), so the query covers both. Lines with a name equal to
    ``name`` (case-insensitive) come first and carry that name as the label; the
    rest follow in registry order with their identifier.
    """
    resp = _get_with_retry(
        f"{CELLOSAURUS_BASE_URL}/search/cell-line",
        {
            "q": f'idsy:"{name}"',
            "format": "json",
            "fld": ["ac", "id", "sy"],
            "rows": 1000,
        },
    )
    resp.raise_for_status()
    target = name.strip().lower()
    exact, other = [], []
    for line in resp.json().get("Cellosaurus", {}).get("cell-line-list", []):
        accession = next(
            (a["value"] for a in line.get("accession-list", []) if "value" in a), None
        )
        names = [n["value"] for n in line.get("name-list", []) if "value" in n]
        if not accession or not names:
            continue
        curie = accession.replace("_", ":", 1)
        match = next((n for n in names if n.strip().lower() == target), None)
        if match:
            exact.append((curie, match))
        else:
            other.append((curie, names[0]))
    return (exact + other)[:_MAX_CANDIDATES]


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
        source += " (also: " + ", ".join(f"{c} {n}" for c, n in candidates[1:]) + ")"
    return source


def _lookup(
    field: str,
    rule: str,
    hint_key: str,
    host: str,
    hints: Mapping[str, str],
    online: bool,
    search: Callable[[str], list[tuple[str, str]]],
) -> tuple[Proposal | None, list[tuple[str, str]]]:
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


def _channel_index(field: str) -> int:
    return int(field.split(".")[1])


def _channel_name(field: str, channel_names: Sequence[str]) -> str | None:
    index = _channel_index(field)
    return channel_names[index] if index < len(channel_names) else None


def _channel_type(name: str) -> str:
    if name.endswith("_prediction"):
        return "predicted"
    if name.startswith(_LABELFREE_PREFIXES):
        return "labelfree"
    return "fluorescence"


def _is_channel_field(field: str, leaf: str) -> bool:
    parts = field.split(".")
    return (
        len(parts) >= 3
        and parts[0] == "channels"
        and parts[1].isdigit()
        and (".".join(parts[2:]) == leaf)
    )


def _propose_one(
    field: str,
    dataset_name: str,
    channel_names: Sequence[str],
    hints: Mapping[str, str],
    online: bool,
) -> Proposal | None:
    if field in (
        "study.release_date",
        "acquisition.acquisition_date",
        "study.year_imaged",
    ):
        date = name_date(dataset_name)
        if date is None:
            return Proposal(field, None, "no date in dataset name", "name_date")
        value = date[:4] if field == "study.year_imaged" else date
        return Proposal(field, value, "dataset name", "name_date")

    if _is_channel_field(field, "channel_type"):
        name = _channel_name(field, channel_names)
        if name is None:
            return Proposal(field, None, "no channel name", "channel_type_from_name")
        return Proposal(
            field, _channel_type(name), "channel name", "channel_type_from_name"
        )

    if _is_channel_field(field, "biological_annotation.biological_target"):
        name = _channel_name(field, channel_names)
        if name not in _VIRTUAL_STAIN_TARGETS:
            return None
        return Proposal(
            field, _VIRTUAL_STAIN_TARGETS[name], "channel name", "virtual_stain_target"
        )

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

    if field == "perturbation.definition.perturbation_id":
        failed, found = _lookup(
            field,
            "ncbitaxon",
            "perturbation.definition.name",
            _registry_of(OLS_BASE_URL),
            hints,
            online,
            search_ncbitaxon,
        )
        if failed:
            return failed
        return Proposal(
            field, found[0][0], _with_alternatives("NCBITaxon", found), "ncbitaxon"
        )

    return None


def propose(
    unresolved: Sequence[str],
    *,
    dataset_name: str,
    channel_names: Sequence[str],
    hints: Mapping[str, str],
    online: bool = True,
) -> list[Proposal]:
    """Propose a value for each unresolved field a rule covers.

    Fields no rule covers get no proposal. A lookup rule that cannot answer (hint
    missing, offline, registry error, no candidates) returns a ``Proposal`` with
    ``value=None`` and the reason in ``source``. This function does not raise on
    lookup failure.
    """
    proposals = (
        _propose_one(field, dataset_name, channel_names, hints, online)
        for field in unresolved
    )
    return [p for p in proposals if p is not None]
