"""EBI OLS4 name search for DCA ontologies (name -> CURIE). No API key.

Vendored from the curation agent's OLS client so both grounding paths send the
same OLS request and filter its results the same way. Two changes from the
source: it is synchronous, and requests go through ``_get_with_retry`` so they
share the validator's retry, backoff and concurrency limits. The source's
reverse ``lookup`` is left out, because ``ontology_lookup.lookup_ols`` already
covers CURIE -> label. See https://www.ebi.ac.uk/ols4/help
"""

from dataclasses import dataclass
from typing import Any

from dca_helpers.validation.ontology_lookup import OLS_BASE_URL, _get_with_retry


@dataclass(frozen=True)
class Ontology:
    """One OLS-resolvable ontology.

    ``ols_name`` is the OLS4 ontology id (lowercase, used in URLs);
    ``curie_prefix`` is the obo_id prefix as OLS returns it (e.g. ``NCBITaxon``,
    but lowercase ``hgnc``); ``domain`` is the concept it grounds.
    """

    ols_name: str
    curie_prefix: str
    domain: str


ONTOLOGIES: dict[str, Ontology] = {
    "ncbitaxon": Ontology("ncbitaxon", "NCBITaxon", "organism / taxon"),
    "uberon": Ontology("uberon", "UBERON", "tissue / anatomical structure"),
    "cl": Ontology("cl", "CL", "cell type"),
    "mondo": Ontology("mondo", "MONDO", "disease"),
    "pato": Ontology("pato", "PATO", "phenotypic quality (e.g. normal / healthy)"),
    "hsapdv": Ontology("hsapdv", "HsapDv", "human developmental stage"),
    "mmusdv": Ontology("mmusdv", "MmusDv", "mouse developmental stage"),
    "efo": Ontology("efo", "EFO", "assay / experiment class"),
    "fbbi": Ontology("fbbi", "FBbi", "biological imaging / sample-prep method"),
    "hgnc": Ontology("hgnc", "hgnc", "gene symbol"),
}


def search(
    query: str,
    *,
    ontology: str | None = None,
    exact: bool = True,
    rows: int = 20,
    query_fields: str = "label,synonym",
    field_list: str = "obo_id,label",
) -> list[dict[str, Any]]:
    """Search OLS4 and return the raw ``docs``.

    ``query_fields=label,synonym`` keeps matches on names only; otherwise a
    common name like "mouse" also matches viruses whose label contains it.
    """
    params: dict[str, Any] = {
        "q": query,
        "type": "class",
        "exact": str(exact).lower(),
        "queryFields": query_fields,
        "fieldList": field_list,
        "rows": rows,
    }
    if ontology:
        params["ontology"] = ontology
    resp = _get_with_retry(f"{OLS_BASE_URL}/search", params)
    resp.raise_for_status()
    return resp.json().get("response", {}).get("docs", [])


def resolve(
    name: str, ontology: str, *, exact: bool = True, rows: int = 20
) -> list[dict[str, str]]:
    """Resolve ``name`` to ``{curie, label}`` matches within ``ontology``.

    ``ontology`` is an ols_name from :data:`ONTOLOGIES`. Results are filtered to
    the ontology's own CURIE prefix (OLS returns the term as imported into
    sibling ontologies too). Empty when nothing matches.
    """
    if (ont := ONTOLOGIES.get(ontology)) is None:
        raise ValueError(f"unknown ontology {ontology!r}")
    docs = search(name, ontology=ontology, exact=exact, rows=rows)
    prefix = f"{ont.curie_prefix}:".lower()
    return [
        {"curie": d["obo_id"], "label": d.get("label", "")}
        for d in docs
        if str(d.get("obo_id", "")).lower().startswith(prefix)
    ]
