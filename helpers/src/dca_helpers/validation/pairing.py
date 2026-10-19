"""
Cross-store projection checks (docs/v0.3/projections.rst).

Per-store validation sees one store at a time, so it cannot tell whether a
projection's source exists or matches it. This pass runs once over every result
of a validate() call, after the per-store work, and compares each projection
with its source using the ImageFacts collected during validation, so it needs no
second read of either store.

A store named after a 3D sibling in the run (``plate_01_proj.ome.zarr`` /
``plate_01_mip.ome.zarr`` next to ``plate_01.ome.zarr``) is a projection of it, so its
image groups MUST carry ``dca.projection`` with ``source_store`` → otherwise ERROR. A
projection that names no source and has no such sibling was projected before deposit,
and none of the checks below apply to it.

For each projection store that names a ``source_store``:

* ``source_store`` resolves to a store: a bare name is a sibling of the
  projection; a URL or absolute path is used as given (falling back to a
  same-named sibling in the run, which covers a local mirror of a remote
  dataset). Not found → ERROR. Found on disk but not part of this run → WARNING
  (the remaining checks need both stores in one run).
* All image groups name the same ``source_store`` → otherwise ERROR.
* The store name ends in ``_proj.ome.zarr`` (or ``_mip.ome.zarr`` when every channel
  uses ``max``) → otherwise WARNING.
* Same image groups as the source (HCS wells and fields) → otherwise ERROR.
* Per image group, against the source's level 0:
  - t, y, x sizes and t, y, x scales match, and ``z_dim_source`` (when given)
    equals the source's z → otherwise ERROR;
  - channels: the uniform form keeps every source channel in order; in the
    per-channel form each projected output's ``source_channel`` (default: its own
    name) is a source channel; ``focus.channel`` is a source channel → otherwise ERROR;
  - dtype equals the source's when every method is ``max`` / ``slice``,
    and is otherwise the source's or a floating-point type → otherwise ERROR.
  A source with z of size 1 → WARNING (nothing to project).

With ``require_projection`` (``dca-validate --require-projection``), additionally:
every 3D image store in the run (z > 1 at level 0) that is not itself a projection
MUST be named as the source of some projection in the run → otherwise ERROR. The spec
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
from dca_helpers.validation.versions.v0_3.models import (
    VALUE_PRESERVING_METHODS,
    projection_methods,
)
from dca_helpers.validation.zarr_node import ZarrNodeType

PROJECTION_STORE_SUFFIX = "_proj.ome.zarr"
# Accepted for a projection whose every channel uses max (the v0.2-era name).
MAX_PROJECTION_STORE_SUFFIX = "_mip.ome.zarr"

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
    kept = [i for i, ax in enumerate(m.axes) if ax in ("t", "y", "x")]
    m_dims = [m.level0_shape[i] for i in kept]
    s_dims = [s.level0_shape[i] for i in kept]
    if m_dims != s_dims:
        names = [m.axes[i] for i in kept]
        _add(
            mip,
            Severity.ERROR,
            ("projection", "source", "shape"),
            f"{where}: level-0 {names} sizes {m_dims} differ from the source's "
            f"{s_dims}; a projection MUST keep t, y and x.",
        )
    _compare_channels(where, mip, m, s)
    projection = m.projection or {}
    methods = projection_methods(projection)
    if m.level0_dtype != s.level0_dtype:
        if methods and methods <= VALUE_PRESERVING_METHODS:
            _add(
                mip,
                Severity.ERROR,
                ("projection", "source", "dtype"),
                f"{where}: dtype {m.level0_dtype} differs from the source's "
                f"{s.level0_dtype}; a projection using only {sorted(methods)} MUST "
                "keep the source dtype.",
            )
        elif not m.level0_dtype.startswith("float"):
            _add(
                mip,
                Severity.ERROR,
                ("projection", "source", "dtype"),
                f"{where}: dtype {m.level0_dtype} differs from the source's "
                f"{s.level0_dtype}; a projection MUST keep the source dtype or, when "
                "a channel uses mean or sum, use a floating-point type.",
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


def _compare_channels(where, mip, m, s) -> None:
    """Output channels map onto source channels (projections.rst § Per-channel form)."""
    projection = m.projection or {}
    c_out = m.level0_shape[m.axes.index("c")] if "c" in m.axes else None
    c_src = s.level0_shape[s.axes.index("c")] if "c" in s.axes else None
    channels = projection.get("channels")
    if isinstance(channels, dict):
        if s.channel_names:
            unknown = sorted(
                (spec.get("source_channel") or name)
                for name, spec in channels.items()
                if isinstance(spec, dict)
                and (spec.get("source_channel") or name) not in s.channel_names
            )
            if unknown:
                _add(
                    mip,
                    Severity.ERROR,
                    ("dca", "projection", "channels"),
                    f"{where}: source channel(s) {unknown} are not channels of the "
                    f"source ({list(s.channel_names)}); set source_channel when an "
                    "output channel is renamed.",
                )
    elif projection.get("method") is not None:
        if c_out != c_src:
            _add(
                mip,
                Severity.ERROR,
                ("projection", "source", "shape", "c"),
                f"{where}: {c_out} channels but the source has {c_src}; the uniform "
                "form keeps every source channel (use the per-channel form to keep "
                "a subset).",
            )
        elif m.channel_names and s.channel_names and m.channel_names != s.channel_names:
            _add(
                mip,
                Severity.ERROR,
                ("dca", "channels"),
                f"{where}: channels {list(m.channel_names)} differ from the source's "
                f"{list(s.channel_names)}; the uniform form keeps them in order.",
            )
    focus = projection.get("focus")
    focus_channel = focus.get("channel") if isinstance(focus, dict) else None
    if isinstance(focus_channel, str) and s.channel_names:
        if focus_channel not in s.channel_names:
            _add(
                mip,
                Severity.ERROR,
                ("dca", "projection", "focus", "channel"),
                f"{where}: focus.channel '{focus_channel}' is not a channel of the "
                f"source ({list(s.channel_names)}).",
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

    names = {
        r.image_facts.projection.get("source_store") for r in projection_images.values()
    }
    names.discard(None)
    if not names:
        # No source named: an image projected before deposit (projections.rst §
        # Projections made before deposit), unless _check_sibling_source finds its 3D
        # source in the run. Naming and pairing do not apply.
        return None

    base = posixpath.basename(store.root)
    store_methods = set().union(
        *(
            projection_methods(r.image_facts.projection)
            for r in projection_images.values()
        )
    )
    max_only = store_methods == {"max"}
    if not (
        base.endswith(PROJECTION_STORE_SUFFIX)
        or (max_only and base.endswith(MAX_PROJECTION_STORE_SUFFIX))
    ):
        _add(
            anchor,
            Severity.WARNING,
            ("projection", "store_name"),
            f"Projection store name should end in '{PROJECTION_STORE_SUFFIX}'"
            + (
                f" (or '{MAX_PROJECTION_STORE_SUFFIX}', as every channel uses max)"
                if max_only
                else ""
            )
            + " (projections.rst § Store Layout).",
        )

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


def _sibling_source(root: str, stores: dict[str, _Store]) -> str | None:
    """The 3D store in the run that ``root`` is named after, if any.

    ``plate_01_proj.ome.zarr`` / ``plate_01_mip.ome.zarr`` next to a 3D, non-projection
    ``plate_01.ome.zarr`` (projections.rst § Projections made before deposit).
    """
    base = posixpath.basename(root)
    for suffix in (PROJECTION_STORE_SUFFIX, MAX_PROJECTION_STORE_SUFFIX):
        if base.endswith(suffix):
            candidate = posixpath.join(
                posixpath.dirname(root), base[: -len(suffix)] + ".ome.zarr"
            )
            source = stores.get(candidate)
            if source is not None and source.is_3d and not source.projection_images:
                return candidate
    return None


def _check_sibling_source(store: _Store, stores: dict[str, _Store]) -> None:
    """A store named after a 3D sibling in the run is its projection, so every image
    group MUST carry ``dca.projection`` with ``source_store``."""
    if store.is_3d:
        return
    sibling = _sibling_source(store.root, stores)
    if sibling is None:
        return
    name = posixpath.basename(sibling)
    images = {p: r for p, r in store.images.items() if r.image_facts is not None}
    no_block = [p or "/" for p, r in images.items() if r.image_facts.projection is None]
    no_source = [
        p or "/"
        for p, r in images.items()
        if r.image_facts.projection is not None
        and not r.image_facts.projection.get("source_store")
    ]
    if no_block:
        _add(
            store.anchor,
            Severity.ERROR,
            ("dca", "projection"),
            f"This store is a projection of '{name}', which is part of this dataset, "
            f"so every image group MUST carry dca.projection naming it as "
            f"source_store; missing on {_listed(no_block)} "
            "(projections.rst § Projections made before deposit).",
        )
    if no_source:
        _add(
            store.anchor,
            Severity.ERROR,
            ("dca", "projection", "source_store"),
            f"This store is a projection of '{name}', which is part of this dataset, "
            f"so dca.projection MUST name it as source_store; missing on "
            f"{_listed(no_source)} (projections.rst § Projection block).",
        )


def check_projection_pairs(
    results: list[ZarrNodeValidationResult],
    *,
    require_projection: bool = False,
    source_exists: Callable[[str], bool] = is_zarr_store,
) -> None:
    """Run the cross-store projection checks, adding issues to ``results`` in place.

    ``source_exists`` is called only for a source that is not part of the run;
    it is a parameter so tests need not touch the filesystem or S3.
    """
    stores = _group_by_store(results)
    sourced: set[str] = set()
    for store in stores.values():
        _check_sibling_source(store, stores)
        if store.projection_images:
            target = _check_projection_store(store, stores, source_exists)
            if target is not None:
                sourced.add(target)

    if not require_projection:
        return
    for root, store in stores.items():
        if store.projection_images or not store.is_3d or root in sourced:
            continue
        _add(
            store.anchor,
            Severity.ERROR,
            ("projection", "require_projection"),
            "3D image store has no projection in this run naming it as "
            "dca.projection.source_store (--require-projection). Expected e.g. "
            f"'{posixpath.basename(root).removesuffix('.ome.zarr')}"
            f"{PROJECTION_STORE_SUFFIX}' next to it.",
        )
