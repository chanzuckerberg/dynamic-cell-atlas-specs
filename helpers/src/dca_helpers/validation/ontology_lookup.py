"""Opt-in ontology-term lookup consistency guard for DCA ontology terms.

Offline validation (the default) enforces only the *pattern* of each
``ontology_term_id`` — that it is a well-formed CURIE. The label is carried
explicitly on every :class:`OntologyTerm` precisely so a stored record is
human-readable *without* a lookup, which means nothing offline checks that the
CURIE names a real term or that its label is the one the source registry
actually gives. This module adds that check: an opt-in, network-backed pass
that resolves each term, compares the stored ``label`` to the registry's
canonical label or one of its synonyms, and — for fields constrained to an
ontology subtree — checks the term descends from the required ancestor. It is
the deferred ontology consistency guard.

It runs only when ``dca-validate --check-ontology`` is passed (equivalently
``validate(..., check_ontology=True)``). When it runs, a term that does not
resolve, or whose label does not match, is an ERROR (MUST) for that run; a term
that cannot be checked because the registry is unreachable is a WARNING, so a
flaky network degrades to "unverified" rather than failing the store.

Terms are routed by CURIE prefix to the registry that actually hosts them,
mirroring ``validate_id`` / ``validate_cell_strain_object`` in
cryoet-data-portal-backend:

  * OBO-library + EFO ontologies (``CL`` / ``EFO`` / ``FBbi`` / ``FBdv`` /
    ``HsapDv`` / ``MmusDv`` / ``MONDO`` / ``NCBITaxon`` / ``PATO`` / ``UBERON``
    / ``WBls`` / ``ZFA`` / ``ZFS``) -> EBI OLS4. The list tracks every prefix a
    v0.2 ``ontology_term_id`` pattern admits, including the multi-organism
    anatomy and development-stage ontologies added in v0.2.4.
  * Cellosaurus cell lines (``CVCL``) -> the Cellosaurus API. OLS does not host
    Cellosaurus, so a cell-line id would otherwise fail spuriously.
  * Any other prefix (gene / compound registries such as ``HGNC`` /
    ``NCBIGene`` / ``ensembl`` / ``PubChem``, and internal doc prefixes) is not
    hosted by a source this guard knows, so it is skipped (not verifiable, not
    an error).

Adapted to the synchronous DCA validator: one ``requests`` call per unique
CURIE, memoized with ``functools.lru_cache`` so a term repeated across many
stores is fetched once per process. CURIE-to-IRI expansion reuses the
``prefixes`` block from the generated LinkML model, so the ontology base URLs
stay in sync with the schema (source of truth) rather than being duplicated.

What a term *is* — finding terms in a metadata object, the sentinel set, and
CURIE parsing — lives in ``versions.v0_2._ontology_terms``, shared with the
always-on offline sentinel check so the two passes cannot disagree. This module
is only how to *ask a registry* about one.

Importing this module pulls in ``requests``. That is deliberate and costs the
offline path nothing: ``experimental_metadata`` imports it inside the
``if check_ontology:`` branch, so an offline run never loads it at all.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from dataclasses import dataclass
from urllib.parse import quote, urlsplit

import requests

from dca_helpers.validation.result import Issue, Severity
from dca_helpers.validation.versions.v0_2._ontology_terms import (
    Term,
    iter_ontology_terms,
)

logger = logging.getLogger(__name__)

# OLS4 REST API. The ``/terms?iri=`` endpoint returns every term across all
# loaded ontologies whose IRI matches, in ``_embedded.terms``.
OLS_BASE_URL = "https://www.ebi.ac.uk/ols4/api"

# Cellosaurus REST API. ``/cell-line/<accession>`` returns the cell line's
# recommended name and synonyms in ``Cellosaurus.cell-line-list[].name-list``.
CELLOSAURUS_BASE_URL = "https://api.cellosaurus.org"

# Seconds to wait on a single request before treating the source as unreachable.
_REQUEST_TIMEOUT = 30

# Both OLS and Cellosaurus are free, key-less, fair-use services with no
# published rate limit; they can throttle abusive bursts (429 / 5xx). Identify
# the client so the maintainers can reach us before blocking, and retry
# transient failures with exponential backoff rather than failing the store.
_USER_AGENT = (
    "dca-helpers-validator "
    "(+https://github.com/chanzuckerberg/dynamic-cell-atlas-specs)"
)
_MAX_ATTEMPTS = 4
_BACKOFF_BASE_SECONDS = 0.5  # 0.5s, 1s, 2s between the 4 attempts
# HTTP statuses worth retrying: rate-limit and transient server errors. Other
# 4xx are definitive (the caller reads them as "not found").
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Ceiling on concurrent requests to the registries, across the whole process.
#
# Retry/backoff handles one flaky call; it does nothing about breadth. ``validate``
# nests ThreadPoolExecutors (one over stores, another over an HCS plate's fields),
# and memoization only dedupes *identical* CURIEs — so a large cold-cache run can
# put a request per unique term in flight at once. These are free, key-less
# services with no published limit, and being throttled is the good outcome; being
# blocked is the bad one. The semaphore is module-level rather than per-run
# because the thread pools are too, and it wraps only the request itself: a thread
# sleeping between retries releases its slot rather than holding the burst open.
_MAX_CONCURRENT_REQUESTS = 8
_request_slots = threading.BoundedSemaphore(_MAX_CONCURRENT_REQUESTS)

# Negative cache: how long a registry stays marked unreachable after a request
# to it has exhausted every retry.
#
# ``functools.lru_cache`` stores return values, not exceptions, so without this
# nothing remembers that a registry is down. Every distinct CURIE pays the full
# retry cost again (3.5s of backoff, or ~2min if the host hangs rather than
# refusing), and so does the *same* CURIE on the next store. The concurrency cap
# above sharpens that: capped at 8 in flight, those retries serialise into
# batches instead of overlapping, so a large run against a dead registry goes
# from slow to apparently hung.
#
# Deliberately a TTL cache rather than a circuit breaker: on expiry the next
# caller simply probes for real, so recovery needs no half-open state machine,
# and any definite response clears the entry immediately. The cost of being
# wrong is bounded — a blip marks terms "unverified" for the window, which is
# already the documented WARNING behaviour for an unreachable registry, not a
# validation failure.
_FAILURE_TTL_SECONDS = 30.0
_registry_failures: dict[str, tuple[float, Exception]] = {}
_failure_lock = threading.Lock()

# CURIE prefixes whose terms are hosted by OLS4 (OBO library + EFO). Kept as an
# explicit allowlist because OLS hosts only ontologies, not the gene / cell-line
# / compound *registries* that other prefixes point at (see module docstring).
#
# This must cover every prefix reachable from an ``ontology_term_id`` pattern in
# the schema: a prefix missing here is *skipped*, which reads as "checked" while
# checking nothing. The multi-organism stage and anatomy ontologies (ZFA / ZFS /
# MmusDv / FBdv / WBls) came in with the v0.2.4 coverage broadening. The drift
# test ``test_ols_prefixes_cover_schema_ontology_patterns`` fails when a future
# spec version adds a prefix that is not classified here or as Cellosaurus.
_OLS_PREFIXES = frozenset(
    {
        "CL",
        "EFO",
        "FBbi",
        "FBdv",
        "HsapDv",
        "MmusDv",
        "MONDO",
        "NCBITaxon",
        "PATO",
        "UBERON",
        "WBls",
        "ZFA",
        "ZFS",
    }
)

# Prefix routed to the Cellosaurus API rather than OLS.
_CELLOSAURUS_PREFIX = "CVCL"

# Fields whose term must fall within a specific ontology subtree, mapping the
# LinkML slot name to the required ancestor CURIE. The CURIE *prefix* (which
# ontology) is already enforced offline by the schema's slot patterns; this adds
# the *branch* (which part of that ontology). Kept in sync with the schema by
# hand, like the if/then rules in experimental_metadata.py:
#   - preparation_method: descendants of FBbi:00000001 ("sample preparation
#     method") — stated explicitly in the slot description.
#   - imaging_method: descendants of FBbi:00000222 ("imaging method"), the FBbi
#     node the slot mirrors. Tighten to FBbi:00000241 ("microscopy") to reject
#     non-microscopy modalities.
_SUBTREE_CONSTRAINTS: dict[str, str] = {
    "preparation_method": "FBbi:00000001",
    "imaging_method": "FBbi:00000222",
}


@dataclass(frozen=True)
class TermRecord:
    """A resolved term: the names a stored label is allowed to match.

    ``names`` holds the registry's canonical label plus its synonyms. Empty
    means the registry confirmed the term but returned no names, so there is
    nothing to compare a label against (existence has already passed).
    """

    source: str  # "OLS" or "Cellosaurus" — for error messages
    names: tuple[str, ...]


class _TransientHTTPError(Exception):
    """A retryable HTTP status (429 / 5xx) that persisted across all attempts."""


class _RegistryRecentlyUnreachable(Exception):
    """Skipped without a request: this registry failed within the TTL window."""


#: Everything :func:`get_with_retry` raises when a registry cannot answer, for callers
#: outside this module that degrade a failed lookup instead of propagating it.
REGISTRY_ERRORS: tuple[type[Exception], ...] = (
    requests.exceptions.RequestException,
    OSError,
    _TransientHTTPError,
    _RegistryRecentlyUnreachable,
)


def registry_of(url: str) -> str:
    """The host a URL belongs to — the grain the failure memory is keyed on."""
    return urlsplit(url).netloc


def _recent_failure(registry: str) -> Exception | None:
    """The cached failure for ``registry`` if one is still inside the TTL."""
    with _failure_lock:
        entry = _registry_failures.get(registry)
        if entry is None:
            return None
        failed_at, error = entry
        if time.monotonic() - failed_at >= _FAILURE_TTL_SECONDS:
            # Expired. Drop it so the next caller probes for real; that natural
            # re-probe is what keeps this a cache rather than a circuit breaker.
            del _registry_failures[registry]
            return None
        return error


def get_with_retry(url: str, params: dict) -> object:
    """GET ``url`` with a descriptive User-Agent, retrying transient failures.

    Retries connection errors, timeouts, and 429/5xx responses up to
    ``_MAX_ATTEMPTS`` with exponential backoff. Returns the response for any
    other status (including definitive 4xx, which callers read as "not found").
    Raises the last error when transient failures persist, so the caller can
    downgrade to a WARNING.

    A persistent failure is remembered per registry for ``_FAILURE_TTL_SECONDS``
    and short-circuits every other request to that host in the window, before
    taking a semaphore slot. See ``_registry_failures``.
    """
    registry = registry_of(url)
    cached = _recent_failure(registry)
    if cached is not None:
        raise _RegistryRecentlyUnreachable(
            f"{registry} was unreachable in the last {_FAILURE_TTL_SECONDS:.0f}s "
            f"({type(cached).__name__}: {cached}); not retried for this term"
        )

    headers = {"User-Agent": _USER_AGENT}
    last_error: Exception = _TransientHTTPError(f"no response from {url}")
    for attempt in range(_MAX_ATTEMPTS):
        try:
            # Held around the call only, so the backoff sleep below does not
            # occupy a slot other threads could be using.
            with _request_slots:
                resp = requests.get(
                    url, params=params, headers=headers, timeout=_REQUEST_TIMEOUT
                )
        except (requests.exceptions.RequestException, OSError) as error:
            last_error = error
        else:
            if resp.status_code not in _RETRYABLE_STATUS:
                # Any definite answer, including a 4xx, means the registry is
                # up. Clear eagerly so recovery is immediate rather than
                # waiting out a TTL that is no longer true.
                with _failure_lock:
                    _registry_failures.pop(registry, None)
                return resp
            last_error = _TransientHTTPError(f"HTTP {resp.status_code} from {url}")
        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))

    with _failure_lock:
        _registry_failures[registry] = (time.monotonic(), last_error)
    logger.warning(
        "%s unreachable after %d attempts; skipping further lookups against it "
        "for %.0fs (%s: %s)",
        registry,
        _MAX_ATTEMPTS,
        _FAILURE_TTL_SECONDS,
        type(last_error).__name__,
        last_error,
    )
    raise last_error


@functools.lru_cache(maxsize=4096)
def lookup_ols(iri: str) -> TermRecord | None:
    """Fetch the OLS term for ``iri``, or None if no term matches.

    Memoized per process. Raises on network/HTTP failure; the caller turns that
    into a WARNING so an unreachable OLS does not hard-fail a store. ``None`` (no
    matching term) *is* cached — it is a definitive answer, not a transient
    failure.
    """
    resp = get_with_retry(f"{OLS_BASE_URL}/terms", {"iri": iri})
    resp.raise_for_status()
    terms = resp.json().get("_embedded", {}).get("terms", [])
    if not terms:
        return None
    # A term can appear in several ontologies (imports); prefer the ontology
    # that defines it, whose label/synonyms are authoritative.
    term = next((t for t in terms if t.get("is_defining_ontology")), terms[0])
    names: list[str] = []
    if term.get("label"):
        names.append(term["label"])
    names.extend(term.get("synonyms") or [])
    return TermRecord(source="OLS", names=tuple(names))


@functools.lru_cache(maxsize=4096)
def lookup_ols_ancestors(ontology: str, iri: str) -> frozenset[str] | None:
    """Return the ``obo_id`` set of a term's ancestors in ``ontology``, or None.

    None means OLS has no such term in that ontology (a definitive 4xx). OLS4's
    ancestor endpoint takes the term IRI *double* URL-encoded in the path.
    Memoized per process; raises on transient failure (caller -> WARNING).
    """
    encoded = quote(quote(iri, safe=""), safe="")
    url = f"{OLS_BASE_URL}/ontologies/{ontology}/terms/{encoded}/ancestors"
    resp = get_with_retry(url, {"size": 1000})
    if 400 <= resp.status_code < 500:
        return None
    terms = resp.json().get("_embedded", {}).get("terms", [])
    return frozenset(term["obo_id"] for term in terms if term.get("obo_id"))


@functools.lru_cache(maxsize=4096)
def lookup_cellosaurus(accession: str) -> TermRecord | None:
    """Fetch the Cellosaurus cell line for ``accession`` (e.g. ``CVCL_0030``).

    Returns None when Cellosaurus reports the accession as absent (4xx), which
    is a definitive "does not resolve". Raises on transient failure (5xx /
    connection), which the caller downgrades to a WARNING.
    """
    resp = get_with_retry(
        f"{CELLOSAURUS_BASE_URL}/cell-line/{accession}",
        {"format": "json", "fld": ["id", "sy"]},
    )
    if 400 <= resp.status_code < 500:
        return None
    data = resp.json()
    names = [
        entry["value"]
        for cell_line in data.get("Cellosaurus", {}).get("cell-line-list", [])
        for entry in cell_line.get("name-list", [])
        if "value" in entry
    ]
    return TermRecord(source="Cellosaurus", names=tuple(names))


def _names_match(stored_label: str, record: TermRecord) -> bool:
    """True when ``stored_label`` equals any of the record's names.

    Comparison is case-insensitive and whitespace-trimmed. When the record
    carries no names there is nothing to compare against, so the label check is
    a no-op (existence already passed).
    """
    if not record.names:
        return True
    target = stored_label.strip().lower()
    return any(target == name.strip().lower() for name in record.names)


def _lookup_term(term: Term) -> tuple[TermRecord | None, bool]:
    """Route a term to its registry and resolve it.

    Returns ``(record, checkable)``. ``checkable`` is False when no configured
    source hosts the prefix, so the term is skipped rather than failed. When
    ``checkable`` is True, ``record`` is the resolved term or None (does not
    resolve). Raises on transient network failure (handled by the caller).
    """
    if term.prefix == _CELLOSAURUS_PREFIX:
        return lookup_cellosaurus(term.cellosaurus_accession), True
    if term.prefix in _OLS_PREFIXES:
        iri = term.iri()
        if iri is None:
            return None, True
        return lookup_ols(iri), True
    return None, False


def _required_ancestor(loc: tuple[str | int, ...]) -> str | None:
    """Required ancestor CURIE for a term at ``loc``, or None if unconstrained.

    Matches on the LinkML slot name in the term's path (e.g. ``imaging_method``),
    so the constraint follows the field regardless of nesting depth.
    """
    for part in loc:
        if isinstance(part, str) and part in _SUBTREE_CONSTRAINTS:
            return _SUBTREE_CONSTRAINTS[part]
    return None


def check_ontology_terms(obj: dict) -> list[Issue]:
    """Verify every ontology term in ``obj`` against its registry.

    For each non-sentinel term whose prefix a configured source hosts: an id
    that does not resolve, or a label matching none of the registry's names, is
    an ERROR; a request that fails is a WARNING (unverified, not invalid). Terms
    whose prefix no source hosts (genes, compounds, doc prefixes) are skipped.

    Terms in a subtree-constrained field (see ``_SUBTREE_CONSTRAINTS``) are
    additionally checked to descend from the required ancestor — a term from the
    wrong branch of the right ontology (e.g. a microscopy term in
    ``preparation_method``) is an ERROR even when it resolves and its label
    matches. Empty list when every checkable term resolves, matches, and sits in
    the required subtree.

    Registry failures are logged at WARNING as well as reported, since the report
    hides warnings unless ``--show-warnings`` is passed and an operator watching a
    long run should see a registry going down as it happens rather than at the end.
    """
    issues: list[Issue] = []
    for loc, term_id, label in iter_ontology_terms(obj):
        term = Term.parse(term_id)
        if term.is_sentinel:
            continue

        try:
            record, checkable = _lookup_term(term)
        except Exception as exc:  # network / HTTP / JSON — unverified, not invalid
            logger.warning(
                "ontology lookup failed for %s at %s: %s: %s",
                term_id,
                ".".join(str(part) for part in loc),
                type(exc).__name__,
                exc,
            )
            issues.append(
                Issue(
                    loc=loc,
                    message=(
                        f"could not verify ontology_term_id '{term_id}' "
                        f"({type(exc).__name__}: {exc})"
                    ),
                    severity=Severity.WARNING,
                )
            )
            continue

        if not checkable:
            # No configured registry hosts this prefix (e.g. gene/compound ids);
            # its existence and label cannot be verified here.
            continue

        if record is None:
            issues.append(
                Issue(
                    loc=loc,
                    message=(
                        f"ontology_term_id '{term_id}' does not resolve to a term "
                        "in its source registry"
                    ),
                    severity=Severity.ERROR,
                )
            )
            continue

        if not _names_match(label, record):
            issues.append(
                Issue(
                    loc=loc,
                    message=(
                        f"label '{label}' does not match any {record.source} name "
                        f"for '{term_id}' (expected one of: "
                        f"{', '.join(record.names)})"
                    ),
                    severity=Severity.ERROR,
                )
            )

        # Subtree membership: some FBbi fields require the term to sit within a
        # specific branch (e.g. preparation_method under "sample preparation
        # method"). Independent of the label check above, so both can fire.
        required_ancestor = _required_ancestor(loc)
        if required_ancestor is None or term_id == required_ancestor:
            continue
        iri = term.iri()
        if iri is None:
            continue
        try:
            ancestors = lookup_ols_ancestors(term.ontology, iri)
        except Exception as exc:  # network / HTTP / JSON — unverified, not invalid
            logger.warning(
                "ontology subtree lookup failed for %s at %s: %s: %s",
                term_id,
                ".".join(str(part) for part in loc),
                type(exc).__name__,
                exc,
            )
            issues.append(
                Issue(
                    loc=loc,
                    message=(
                        f"could not verify the ontology subtree of '{term_id}' "
                        f"({type(exc).__name__}: {exc})"
                    ),
                    severity=Severity.WARNING,
                )
            )
            continue
        if ancestors is None or required_ancestor not in ancestors:
            issues.append(
                Issue(
                    loc=loc,
                    message=(
                        f"ontology_term_id '{term_id}' is outside the required "
                        f"ontology subtree: it must be a descendant of "
                        f"'{required_ancestor}' for this field"
                    ),
                    severity=Severity.ERROR,
                )
            )

    return issues
