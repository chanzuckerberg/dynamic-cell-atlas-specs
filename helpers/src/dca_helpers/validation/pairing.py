"""
Cross-store projection checks (docs/v0.3/projections.rst).

Per-store validation sees one store at a time, so it cannot tell whether a
projection's source exists or matches it. This pass runs once over every result
of a validate() call, after the per-store work, and compares each projection
with its source using the ImageFacts collected during validation, so it needs no
second read of either store.

For each projection store (one whose image groups carry ``dca.projection``):

* ``source_store`` resolves to a store: a bare name is a sibling of the
  projection; a URL or absolute path is used as given (falling back to a
  same-named sibling in the run, which covers a local mirror of a remote
  dataset). Not found → ERROR. Found on disk but not part of this run → WARNING
  (the remaining checks need both stores in one run).
* All image groups name the same ``source_store`` → otherwise ERROR.
* The store name ends in ``_mip.ome.zarr`` → otherwise WARNING.
* Same image groups as the source (HCS wells and fields) → otherwise ERROR.
* Per image group: t, c, y, x sizes and dtype match the source's level 0, the
  t, y, x scales match, and ``z_dim_source`` (when given) equals the source's z
  → otherwise ERROR. A source with z of size 1 → WARNING (nothing to project).

With ``require_mip`` (``dca-validate --require-mip``), additionally: every 3D
image store in the run (z > 1 at level 0) that is not itself a projection MUST
be named as the source of some projection in the run → otherwise ERROR. The spec
does not require projections, so this is opt-in.

Issues are added to existing results: store-level findings to the store's
anchor result (an HCS plate root, or a single-image store's one result), image
findings to that image's result.
"""

from __future__ import annotations

import math
import posixpath
from collections.abc import Callable
from dataclasses import dataclass, field

from dca_helpers.validation.discovery import is_zarr_store
from dca_helpers.validation.result import Issue, Severity, ZarrNodeValidationResult
from dca_helpers.validation.zarr_node import ZarrNodeType

PROJECTION_STORE_SUFFIX = "_mip.ome.zarr"

# Results that are not image stores and never take part in pairing.
_NON_IMAGE_NODE_TYPES = frozenset(
    {ZarrNodeType.IMAGE_LABEL, ZarrNodeType.LABELS_LIST, ZarrNodeType.GEFF}
)
# Lists in messages are cut to this many entries.
_MAX_LISTED = 10


@dataclass
class _Store:
    root: str
    anchor: ZarrNodeValidationResult | None = None
    images: dict[str, ZarrNodeValidationResult] = field(default_factory=dict)

    @property
    def projection_images(self) -> dict[str, ZarrNodeValidationResult]:
        return {
            p: r
            for p, r in self.images.items()
            if r.image_facts is not None and r.image_facts.projection is not None
        }

    @property
    def is_3d(self) -> bool:
        for r in self.images.values():
            facts = r.image_facts
            if facts is not None and "z" in facts.axes:
                if facts.level0_shape[facts.axes.index("z")] > 1:
                    return True
        return False


def _add(
    result: ZarrNodeValidationResult | None,
    severity: Severity,
    loc: tuple,
    message: str,
) -> None:
    if result is None:
        return
    result.issues.append(Issue(loc=loc, message=message, severity=severity))
    if severity == Severity.ERROR:
        result.passed = False


def _listed(items: list[str]) -> str:
    items = sorted(items)
    more = len(items) - _MAX_LISTED
    head = ", ".join(items[:_MAX_LISTED])
    return f"{head} (+{more} more)" if more > 0 else head


def _group_by_store(results: list[ZarrNodeValidationResult]) -> dict[str, _Store]:
    stores: dict[str, _Store] = {}
    for r in results:
        if r.skipped or r.node_type in _NON_IMAGE_NODE_TYPES:
            continue
        if r.parent_path is not None:
            root = r.parent_path.rstrip("/")
            store = stores.setdefault(root, _Store(root))
            store.images[r.node_path[len(root) :].strip("/")] = r
        else:
            root = r.node_path.rstrip("/")
            store = stores.setdefault(root, _Store(root))
            store.anchor = r
            if r.image_facts is not None:
                store.images[""] = r
    return stores


def _is_absolute(location: str) -> bool:
    return "://" in location or location.startswith("/")


def _resolve(projection_root: str, source_store: str, stores: dict[str, _Store]) -> str:
    """The source store's root path: a bare name is a sibling of the projection."""
    sibling = posixpath.join(
        posixpath.dirname(projection_root), posixpath.basename(source_store.rstrip("/"))
    )
    if not _is_absolute(source_store):
        return sibling
    target = source_store.rstrip("/")
    # A URL that is not in the run but whose same-named sibling is: the usual case
    # when validating a local mirror of a remote dataset.
    if target not in stores and sibling in stores:
        return sibling
    return target


def _compare_image(
    path: str,
    mip: ZarrNodeValidationResult,
    src: ZarrNodeValidationResult,
) -> None:
    m, s = mip.image_facts, src.image_facts
    if m is None or s is None:
        return  # an image that could not be opened already carries its error
    where = f"image '{path or '/'}'"
    if s.projection is not None:
        _add(
            mip,
            Severity.ERROR,
            ("projection", "source"),
            f"{where}: the source image is itself a projection.",
        )
        return
    if m.axes != s.axes or "z" not in m.axes:
        _add(
            mip,
            Severity.ERROR,
            ("projection", "source", "axes"),
            f"{where}: axes {list(m.axes)} cannot be compared with the source's "
            f"{list(s.axes)}.",
        )
        return
    zi = m.axes.index("z")
    src_z = s.level0_shape[zi]
    if src_z <= 1:
        _add(
            mip,
            Severity.WARNING,
            ("projection", "source", "z"),
            f"{where}: the source has z of size {src_z}, so there is nothing to "
            "project.",
        )
    kept = [i for i in range(len(m.axes)) if i != zi]
    m_dims = [m.level0_shape[i] for i in kept]
    s_dims = [s.level0_shape[i] for i in kept]
    if m_dims != s_dims:
        names = [m.axes[i] for i in kept]
        _add(
            mip,
            Severity.ERROR,
            ("projection", "source", "shape"),
            f"{where}: level-0 {names} sizes {m_dims} differ from the source's "
            f"{s_dims}; a projection MUST keep every axis but the projected one.",
        )
    if m.level0_dtype != s.level0_dtype:
        _add(
            mip,
            Severity.ERROR,
            ("projection", "source", "dtype"),
            f"{where}: dtype {m.level0_dtype} differs from the source's "
            f"{s.level0_dtype}; a projection MUST keep the source dtype.",
        )
    for ax in ("t", "y", "x"):
        mv, sv = m.level0_scale.get(ax), s.level0_scale.get(ax)
        if mv is not None and sv is not None and not math.isclose(mv, sv, rel_tol=1e-9):
            _add(
                mip,
                Severity.ERROR,
                ("projection", "source", "scale", ax),
                f"{where}: level-0 '{ax}' scale {mv} differs from the source's "
                f"{sv}; it MUST be equal.",
            )
    z_dim_source = (m.projection or {}).get("z_dim_source")
    if isinstance(z_dim_source, int) and z_dim_source != src_z:
        _add(
            mip,
            Severity.ERROR,
            ("dca", "projection", "z_dim_source"),
            f"{where}: z_dim_source is {z_dim_source} but the source's level-0 z "
            f"is {src_z}.",
        )


def _check_projection_store(
    store: _Store,
    stores: dict[str, _Store],
    source_exists: Callable[[str], bool],
) -> str | None:
    """Run the checks for one projection store; return its source's root, if found."""
    projection_images = store.projection_images
    anchor = store.anchor
    if len(projection_images) != len(store.images):
        plain = [p or "/" for p in store.images if p not in projection_images]
        _add(
            anchor,
            Severity.ERROR,
            ("projection",),
            f"Store mixes projected and non-projected image groups; these carry no "
            f"dca.projection: {_listed(plain)}.",
        )

    if not posixpath.basename(store.root).endswith(PROJECTION_STORE_SUFFIX):
        _add(
            anchor,
            Severity.WARNING,
            ("projection", "store_name"),
            f"Projection store name should end in '{PROJECTION_STORE_SUFFIX}' "
            "(projections.rst § Store Layout).",
        )

    names = {
        r.image_facts.projection.get("source_store") for r in projection_images.values()
    }
    names.discard(None)  # a missing source_store is a per-image model error
    if len(names) != 1:
        if len(names) > 1:
            _add(
                anchor,
                Severity.ERROR,
                ("dca", "projection", "source_store"),
                f"Image groups name different source stores: {_listed(list(names))}.",
            )
        return None
    (source_store,) = names
    if not isinstance(source_store, str) or not source_store:
        return None

    target = _resolve(store.root, source_store, stores)
    source = stores.get(target)
    if source is None:
        try:
            exists = source_exists(target)
        except Exception:  # unreadable location: say so, don't crash the run
            exists = None
        if exists is False:
            _add(
                anchor,
                Severity.ERROR,
                ("dca", "projection", "source_store"),
                f"Source store '{source_store}' (resolved to '{target}') does not "
                "exist.",
            )
        else:
            _add(
                anchor,
                Severity.WARNING,
                ("dca", "projection", "source_store"),
                f"Source store '{target}' was not part of this run, so the projection "
                "was not compared with it. Validate a path that contains both stores.",
            )
        return None

    missing = [p or "/" for p in source.images if p not in store.images]
    extra = [p or "/" for p in store.images if p not in source.images]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing from the projection: {_listed(missing)}")
        if extra:
            parts.append(f"not in the source: {_listed(extra)}")
        _add(
            anchor,
            Severity.ERROR,
            ("projection", "source", "hierarchy"),
            "Projection does not mirror its source's image groups; "
            + "; ".join(parts)
            + " (projections.rst § Store Layout).",
        )
    for path, result in store.images.items():
        if path in source.images:
            _compare_image(path, result, source.images[path])
    return target


def check_projection_pairs(
    results: list[ZarrNodeValidationResult],
    *,
    require_mip: bool = False,
    source_exists: Callable[[str], bool] = is_zarr_store,
) -> None:
    """Run the cross-store projection checks, adding issues to ``results`` in place.

    ``source_exists`` is called only for a source that is not part of the run;
    it is a parameter so tests need not touch the filesystem or S3.
    """
    stores = _group_by_store(results)
    sourced: set[str] = set()
    for store in stores.values():
        if store.projection_images:
            target = _check_projection_store(store, stores, source_exists)
            if target is not None:
                sourced.add(target)

    if not require_mip:
        return
    for root, store in stores.items():
        if store.projection_images or not store.is_3d or root in sourced:
            continue
        _add(
            store.anchor,
            Severity.ERROR,
            ("projection", "require_mip"),
            "3D image store has no projection in this run naming it as "
            "dca.projection.source_store (--require-mip). Expected e.g. "
            f"'{posixpath.basename(root).removesuffix('.ome.zarr')}"
            f"{PROJECTION_STORE_SUFFIX}' next to it.",
        )
