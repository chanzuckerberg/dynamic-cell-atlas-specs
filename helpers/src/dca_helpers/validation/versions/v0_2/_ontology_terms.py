"""What an ontology term *is*, shared by the offline and network-backed checks.

Two passes need to find and pick apart the ontology terms in a metadata object,
and they run in different worlds:

  * ``experimental_metadata._sentinel_label_issues`` — always on, offline. Checks
    that a sentinel ``ontology_term_id`` agrees with its ``label``.
  * ``ontology_lookup.check_ontology_terms`` — opt-in, network-backed. Resolves
    each term against its registry.

Between them they had two byte-identical copies of the traversal, two copies of
the sentinel set, and CURIE string-surgery (``split(":")``, ``replace(":", "_")``,
``startswith("CVCL:")``) scattered across both modules. This module holds the
single answer to each: :func:`iter_ontology_terms` finds terms, :data:`SENTINELS`
says which values stand in for a CURIE, and :class:`Term` parses one.

Why a third module rather than one importing the other: ``experimental_metadata``
already imports ``ontology_lookup`` to dispatch ``--check-ontology``, so putting
this in ``experimental_metadata`` would make the two modules require each other.
Putting it in ``ontology_lookup`` would point the always-on offline path at the
module that owns the HTTP client and imports ``requests`` at module scope. This
module imports neither.

Note this shares the *definitions*, not the pass: with ``--check-ontology`` the
object is still walked twice. Folding them into a single loop would entangle the
always-on rules with the opt-in ones, and the walk is a cheap in-memory
recursion next to the HTTP requests it feeds.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Iterator

from dca_helpers.validation.versions.v0_2 import _generated_models

# Sentinels stand in for a CURIE where a subclass permits them (see OntologyTerm
# in the schema): ``unknown`` (absent from source), ``na`` (does not apply),
# ``unavailable`` (known but no ontology term exists). None is a CURIE, so none
# is looked up, and each constrains its ``label`` differently — the rule the
# offline ``_sentinel_label_issues`` pass enforces.
SENTINELS: frozenset[str] = frozenset({"unknown", "na", "unavailable"})


@functools.lru_cache(maxsize=1)
def _prefix_iri_map() -> dict[str, str]:
    """Map each CURIE prefix to its IRI base, read from the generated model.

    The generated ``linkml_meta['prefixes']`` is the schema's authoritative
    prefix table (e.g. ``FBbi`` -> ``http://purl.obolibrary.org/obo/FBbi_``),
    so expansion never drifts from the CURIE patterns the offline validator
    enforces.
    """
    prefixes = _generated_models.linkml_meta["prefixes"]
    return {
        entry["prefix_prefix"]: entry["prefix_reference"]
        for entry in (prefixes[key] for key in prefixes)
    }


@dataclass(frozen=True)
class Term:
    """A parsed ``ontology_term_id``.

    Built with :meth:`parse` rather than directly, so every caller splits a CURIE
    the same way. A value that is not ``prefix:local`` (a sentinel, or anything
    malformed) parses with empty ``prefix`` / ``local`` and ``is_curie`` False,
    which is how callers tell "not a CURIE" from "a CURIE that did not resolve".
    """

    value: str  # exactly as stored, for error messages
    prefix: str  # "" when the value is not a CURIE
    local: str  # "" when the value is not a CURIE

    @classmethod
    def parse(cls, value: str) -> Term:
        prefix, separator, local = value.partition(":")
        if not separator or not local:
            return cls(value=value, prefix="", local="")
        return cls(value=value, prefix=prefix, local=local)

    @property
    def is_curie(self) -> bool:
        return bool(self.prefix and self.local)

    @property
    def is_sentinel(self) -> bool:
        return self.value in SENTINELS

    @property
    def cellosaurus_accession(self) -> str:
        """``CVCL:0030`` -> ``CVCL_0030``, the form the Cellosaurus API takes."""
        return f"{self.prefix}_{self.local}"

    @property
    def ontology(self) -> str:
        """The OLS ontology id for this term's prefix (OLS lowercases them)."""
        return self.prefix.lower()

    def iri(self) -> str | None:
        """Expand to a full IRI, or None if not a CURIE or the prefix is unknown."""
        if not self.is_curie:
            return None
        base = _prefix_iri_map().get(self.prefix)
        return None if base is None else f"{base}{self.local}"


def curie_to_iri(curie: str) -> str | None:
    """Expand a CURIE (e.g. ``FBbi:00000251``) to a full IRI, or None.

    Thin wrapper over :meth:`Term.iri` for callers holding a bare string.
    """
    return Term.parse(curie).iri()


def iter_ontology_terms(
    obj: object, path: tuple[str | int, ...] = ()
) -> Iterator[tuple[tuple[str | int, ...], str, str]]:
    """Yield ``(loc, ontology_term_id, label)`` for every OntologyTerm in ``obj``.

    Structural rather than schema-driven: any mapping carrying both keys as
    strings is a term. That tracks the schema by shape rather than by an
    enumerated field list, so an OntologyTerm-typed field added in a later spec
    version is covered without editing this module — which is how the
    image-level ``perturbation`` block (v0.2.3) became covered for free. ``loc``
    is the dotted/indexed path to the term, matching :class:`Issue` locations.
    """
    if isinstance(obj, dict):
        term_id, label = obj.get("ontology_term_id"), obj.get("label")
        if isinstance(term_id, str) and isinstance(label, str):
            yield path, term_id, label
        for key, value in obj.items():
            yield from iter_ontology_terms(value, path + (key,))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from iter_ontology_terms(value, path + (index,))
