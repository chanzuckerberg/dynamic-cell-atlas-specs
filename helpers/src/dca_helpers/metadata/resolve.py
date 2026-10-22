"""Resolve the config's ``[dca.*]`` block against one group's metadata.

The config carries the DCA metadata in the schema's own shape. A leaf says where its
value comes from:

    [[dca.study.dataset_name]]
    value = "LINCS Cell Painting"                     # curated
    [[dca.perturbation.control_class]]
    column = "pert_type"                              # from source metadata
    map = { DMSO_control = "negcon", compound = "treatment" }

``map`` is a post-processing step on whatever the base value resolved to, not a third
alternative to ``column`` / ``value``. Everything else in the tree just recurses, and
plain literals pass through.

## A blank cell declines, and takes its subtree with it

A ``column`` cell that is blank (:data:`_MISSING_CELLS`) means "this group has no value
here" — a DMSO control well has no compound, so no MoA and no concentration. That leaf
is omitted, and **a node whose every column-leaf came back blank is pruned entirely,
literals included**.

The literals are the point. Five of the spec's six conditional rules have the shape
"a value at one position makes a sibling block mandatory" — ``chemical`` is required
when ``modality`` is ``compound``, ``genetic`` when it is a genetic kind, ``definition``
when ``control_class`` is not ``empty``, ``citation`` when ``source_kind`` is ``agent``,
``cell_line_name`` when ``tissue_type`` is ``cell line``. So dropping a blank leaf but
leaving its literal sibling behind writes a block that fails validation: a control well
would keep ``modality = "compound"`` and be asked for the ``chemical`` it does not have.
Pruning the smallest enclosing node keeps them consistent.

Missing-ness is tested **only on ``column`` cells, never on literals**. That is what
keeps ``na`` unambiguous: source data writes ``nan`` / ``""`` / ``n/a`` and the resolver
reads "blank", while a config that writes ``value = "na"`` always means the DCA
does-not-apply sentinel. The two are separated by where the value came from, so no
opt-out keyword is needed.
"""

import logging
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ValidationError, model_validator

from dca_helpers.validation.versions.v0_2.experimental_metadata import (
    DCAExperimentalMetadataV0_2,
    _BiosampleV0_2,
    _PerturbationAssignmentV0_2,
    validate_experimental_metadata_v0_2,
)

logger = logging.getLogger(__name__)

#: The manifest / obs key carrying a group's resolved DCA metadata.
DCA_KEY = "dca"

#: The prefix ``dca_helpers`` puts on a SHOULD-severity message. Mirrors
#: ``dca_helpers.validation.validator._SHOULD_MARKER``, which is what turns a raised
#: message into ``Severity.WARNING`` rather than ``Severity.ERROR``.
_SHOULD_MARKER = "[SHOULD]"


class DcaValidationError(ValueError):
    """A resolved DCA block failed validation.

    ``str(err)`` is ``invalid DCA metadata for {where}: {issue}; {issue}``. ``issues``
    holds one ``"<dotted.loc>: <message>"`` string per MUST finding and ``fields`` the
    dotted loc of each, in the same order, so a caller can report per field without
    parsing the message.
    """

    def __init__(self, where: str, issues: list[str], fields: list[str]) -> None:
        self.where = where
        self.issues = issues
        self.fields = fields
        super().__init__(f"invalid DCA metadata for {where}: {'; '.join(issues)}")


#: The only keys a leaf spec may hold.
_SPEC_KEYS = frozenset({"column", "value", "map"})

#: Cell contents that mean "this group has no value here". Compared case-insensitively
#: after stripping, and **only** against a value read from a ``column``.
_MISSING_CELLS = frozenset({"", "nan", "none", "null", "na", "n/a"})


#: A leaf, or a whole node, that this group declined. Only ever compared by identity,
#: and never reaches a message or the manifest, so a bare object is enough.
MISSING = object()


def _is_leaf(node: Any) -> bool:
    """True when *node* is a leaf spec rather than a piece of the schema tree.

    Membership alone (``"value" in node``) is not enough, because ``value`` is a real
    field name in this schema: ``Concentration`` is ``{value, unit, text}``. Testing
    that the keys are a *subset* of the spec keys tells the two apart — a schema object
    always carries a sibling field that no spec has.
    """
    return (
        isinstance(node, Mapping)
        and node.keys() <= _SPEC_KEYS
        and ("column" in node or "value" in node)
    )


def _map_lookup(mapping: Mapping[str, Any], base: Any) -> Any:
    """Look *base* up in a leaf's ``map``, canonicalising it to ``str`` first.

    TOML map keys are always strings, but a base read from a ``column`` keeps its source
    dtype — a numeric ``binning`` column hands ``int 2`` while the map is keyed ``"2"``, so
    a raw lookup missed and aborted the whole manifest. Try the value as-is, as ``str``,
    then (for an integral float) as the ``int`` string; a literal ``str`` base hits the
    first candidate unchanged.
    """
    candidates: list[Any] = [base, str(base)]
    if isinstance(base, float) and base.is_integer():
        candidates.append(str(int(base)))
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    raise ValueError(
        f"dca leaf resolved to {base!r}, which is not in its map: {sorted(mapping)}"
    )


def _is_blank(cell: Any) -> bool:
    """True when a ``column`` cell carries no value for this group."""
    if cell is None:
        return True
    if isinstance(cell, float) and cell != cell:  # NaN
        return True
    if isinstance(cell, str):
        return cell.strip().lower() in _MISSING_CELLS
    return False


def _indexed_to_list(node: dict) -> Any:
    """``{"0": a, "1": b}`` -> ``[a, b]``.

    TOML has no integer keys, so ``[dca.a.0]`` and ``[[dca.a.1.b]]`` both parse to a
    dict keyed by digit *strings*. The schema wants a list, and the ordering is the
    numeric one, not the lexical one that would put 10 before 2.
    """
    if node and all(isinstance(k, str) and k.isdigit() for k in node):
        return [node[k] for k in sorted(node, key=int)]
    return node


def _resolve(node: Any, metadata: Mapping[str, Any]) -> tuple[Any, int, int]:
    """Resolve *node*, returning ``(value, columns_seen, columns_blank)``.

    The two counts are what drive pruning: a node is declined when it drew on at least
    one column and every one of them was blank.
    """
    if isinstance(node, Mapping):
        if "column" in node and "value" in node:
            raise ValueError(f"dca leaf has both 'column' and 'value': {node}")

        if not _is_leaf(node):
            out: dict[Any, Any] = {}
            columns = blank = 0
            for key, child in node.items():
                value, seen, missing = _resolve(child, metadata)
                columns += seen
                blank += missing
                if value is not MISSING:
                    out[key] = value
            if columns and columns == blank:
                return MISSING, columns, blank
            return _indexed_to_list(out), columns, blank

        if "column" in node:
            column = node["column"]
            if column not in metadata:
                raise ValueError(
                    f"dca leaf reads column {column!r}, which is not in the group's "
                    f"metadata; it holds {sorted(metadata)}"
                )
            cell = metadata[column]
            if _is_blank(cell):
                return MISSING, 1, 1
            base, columns, blank = cell, 1, 0
        else:
            base, columns, blank = _resolve(node["value"], metadata)
            if base is MISSING:
                return MISSING, columns, blank

        if "map" in node:
            return _map_lookup(node["map"], base), columns, blank
        return base, columns, blank

    if isinstance(node, list):
        parts = [_resolve(item, metadata) for item in node]
        columns = sum(seen for _, seen, _ in parts)
        blank = sum(missing for _, _, missing in parts)
        if columns and columns == blank:
            return MISSING, columns, blank
        kept = [value for value, _, _ in parts if value is not MISSING]
        # A single-value leaf written as `[[dca.a.b]]` parses to a one-item list;
        # collapse it back. Only when that item is a leaf spec, though — a genuine
        # one-element list value (`study_type = ["Other"]`, `organism = [{...}]`)
        # must stay a list.
        if len(node) == 1 and _is_leaf(node[0]):
            return (kept[0] if kept else MISSING), columns, blank
        return kept, columns, blank

    return node, 0, 0


def resolve(node: Any, metadata: Mapping[str, Any]) -> Any:
    """Return *node* with every leaf spec replaced by this group's value.

    Pruning applies to nodes *inside* the tree, never to the tree itself. The top
    level of a ``[dca]`` block is a container of independent blocks — ``study``,
    ``biosample``, ``perturbation`` — not a node whose literals are conditional on its
    columns, so the all-blank rule must not aggregate across them. Without the
    exemption a group that declines every source column (a control well, a group the
    platemap omits) prunes at the root and loses ``study`` and ``biosample`` too,
    leaving a store with no DCA metadata at all rather than one missing a
    perturbation.
    """
    if isinstance(node, Mapping) and not _is_leaf(node):
        # Every top-level child is an independent block, resolved on its own via
        # :func:`_resolve_map` (is_entry always True — nothing here nests further).
        # Deliberately NOT _indexed_to_list. Applying the digit-keys-become-a-list rule
        # at the top level would convert a whole `[dca]` block whose blocks happened to
        # be digit-named into a list, and every caller reaches into it by key. The rule
        # belongs one level down, where `_resolve` applies it to an indexed list inside
        # the schema. `channels` never reaches here at all — `resolve_dca` selects it by
        # key first, because it is keyed by source channel rather than by position.
        return _resolve_map(node, metadata, is_entry=lambda _child: True)
    value, _, _ = _resolve(node, metadata)
    return {} if value is MISSING else value


def _resolve_channels(
    catalogue: Mapping[str, Any],
    metadata: Mapping[str, Any],
    channel_keys: Sequence[str],
) -> list[dict]:
    """Select *channel_keys* out of *catalogue*, in that order, and number them.

    ``catalogue`` is keyed by the **source** channel key — the value in the source
    table's channel-identity column, or the column name for a wide table — and
    ``channel_keys`` is the order ``channel_column_names`` fixed for this store. So the
    binding is by name and the order is stated once, separately: reordering the
    catalogue's sections cannot mislabel a channel, and a store that has five of six
    channels selects five entries rather than needing its own catalogue.

    ``index`` is the C axis of the pixel data, so it is the channel's position in *this
    store* and is derived here, never authored. A five-channel store whose first
    declared channel is RNA gets ``RNA`` at ``index = 0``, which is correct: RNA
    genuinely is plane 0 of that array.

    Each entry is resolved on its own rather than as part of the whole tree, which is
    what lets a pruned channel be **named**. A ``[dca.channels.X]`` whose every column
    leaf resolves blank is declined by the all-blank rule, and under the previous
    whole-tree resolve the survivors silently closed the gap and renumbered — so a
    config routing a per-channel column that happens to be empty in every row deleted
    channels and still exited 0 (the hazard ``config/jump_cp.toml`` warns about at
    length). Here it is an error that says which channel and why.
    """
    selected = []
    for key in channel_keys:
        if key not in catalogue:
            raise ValueError(
                f"channel {key!r} has no [dca.channels.{key}] block; the catalogue "
                f"declares {sorted(catalogue)}"
            )
        value, _, _ = _resolve(catalogue[key], metadata)
        if value is MISSING:
            raise ValueError(
                f"channel {key!r} resolved to nothing: every column its "
                f"[dca.channels.{key}] block reads is blank for this group, so the "
                "whole entry was declined. The C axis is built from the declared "
                "channels, so dropping one here would leave a plane undescribed — "
                "author the fields as literals, or stop routing an empty column to them."
            )
        if not isinstance(value, dict):
            raise ValueError(
                f"channel {key!r} resolved to {type(value).__name__}, not a mapping; "
                f"[dca.channels.{key}] must be a block of channel fields"
            )
        if not str(value.get("name") or "").strip():
            raise ValueError(
                f"channel {key!r} resolved without a usable name: its "
                f"[dca.channels.{key}] block either reads `name` from a column "
                "that is blank for this group, or authors it as an empty string. "
                "`dca.channels` entries and the omero labels are both built from "
                "`name` — and neither the spec model nor dca-validate rejects a "
                "blank one — so every entry must carry a non-blank name."
            )
        value["index"] = len(selected)
        selected.append(value)
    return selected


def _flatten_provenance(node: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested provenance tables into the spec's dotted-field-path map.

    The spec keys ``provenance`` flat, by dotted field path — ``"biosample.tissue"``
    is ONE key. TOML nests: ``[dca.provenance.biosample.tissue]`` parses to
    ``{"biosample": {"tissue": {...}}}``, so a config can author the natural nested
    form and this puts the dots back. The quoted-single-key form
    (``[dca.provenance."biosample.tissue"]``) parses flat already and passes
    through unchanged, so both spellings are accepted.

    Recursion stops at the first mapping carrying ``source_kind`` — the one field
    the spec REQUIRES on every entry — which is what keeps a record's own subtables
    (``citation``) from being mistaken for path segments. A path that extends
    another path's record (``a.b`` and ``a.b.c`` both having provenance) cannot be
    expressed nested, because ``c`` would sit inside ``a.b``'s record; the guard
    below catches that and says to quote the longer path instead.

    The same path authored in BOTH spellings is an error too, not a merge: TOML
    parses ``"biosample.tissue"`` (quoted) and ``biosample.tissue`` (nested) as
    different keys, so both records arrive here, and silently letting one win would
    lose whichever the author thought they were keeping — the exact half-migrated
    shape a quoted→nested conversion invites.
    """
    out: dict[str, Any] = {}

    def _put(path: str, record: Any) -> None:
        if path in out:
            raise ValueError(
                f"provenance entry {path!r} is authored twice — once as a quoted "
                "single key and once nested (TOML parses those as different keys). "
                "Keep exactly one."
            )
        out[path] = record

    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and "source_kind" in value:
            nested = sorted(
                k
                for k, v in value.items()
                if isinstance(v, Mapping) and "source_kind" in v
            )
            if nested:
                raise ValueError(
                    f"provenance entry {path!r} contains what looks like a nested "
                    f"entry for {path}.{nested[0]!s}: a field path that extends "
                    "another provenance-carrying path must be authored flat, with "
                    f'the full path quoted as one key — [dca.provenance."{path}.'
                    f'{nested[0]}"] — so the two records stay separate.'
                )
            _put(path, value)
        elif isinstance(value, Mapping) and value:
            for sub_path, record in _flatten_provenance(value, f"{path}.").items():
                _put(sub_path, record)
        else:
            raise ValueError(
                f"provenance entry {path!r} is not a provenance record: every entry "
                "must carry `source_kind` (the spec requires it on each one), so "
                "either a record is missing its `source_kind` or this path is "
                "mistyped."
            )
    return out


def _resolve_map(
    node: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    is_entry: Callable[[Any], bool],
) -> dict:
    """Resolve a map of *independent* entries: each entry prunes on its own and the
    container is never aggregate-pruned.

    This is the one primitive behind the two prune exemptions in this module. The
    generic :func:`_resolve` declines a whole node when it drew on a column and every
    column was blank — right for a conditional block, wrong for a container whose
    entries are independent, where one blank entry must not take its siblings (or the
    container's literals) with it. *is_entry* is the knob that says how deep the
    independent treatment goes:

    - the ``[dca]`` block root passes ``is_entry = lambda _: True`` — every child
      (``study``, ``biosample``, ``perturbation``) is resolved on its own via
      :func:`_resolve`, so a control well that declines every source column keeps
      ``study`` / ``biosample`` rather than pruning the block to nothing;
    - ``provenance`` passes ``is_entry = has "source_kind"`` — a record resolves on
      its own, and an intermediate nesting segment (a dotted ``field_path`` prefix)
      recurses so the treatment reaches one level deeper. Two column-gated records
      that blank together (an empty well's ``definition`` + ``concentration`` gates)
      then vanish individually instead of pruning the whole map — literal records for
      ``study`` / ``biosample`` included — which would fail validation with
      ``provenance: Field required``. A record whose ``field_path`` names a field that
      pruned SHOULD vanish (it is a MUST error otherwise), so per-entry pruning is
      required, not optional. A map whose every entry pruned resolves to ``{}``, never
      to nothing.

    A pruned entry is dropped; the container is kept. (``channels`` is the third
    independent-entry map, but it is an *ordered selection by key* whose pruned entry
    is an error rather than a drop, so it keeps its own :func:`_resolve_channels`.)
    """
    out: dict[str, Any] = {}
    for key, child in node.items():
        if not is_entry(child) and isinstance(child, Mapping) and child:
            segment = _resolve_map(child, metadata, is_entry=is_entry)
            if segment:
                out[key] = segment
        else:
            value, _, _ = _resolve(child, metadata)
            if value is not MISSING:
                out[key] = value
    return out


def resolve_dca(
    block: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    channel_keys: Sequence[str] | None = None,
    where: str = "this group",
) -> dict:
    """Resolve a whole ``[dca]`` block, selecting this store's channels by key.

    *channel_keys* is the store's source channel keys in C order, from
    :meth:`IngestConfig.channels_for_store`. It is required whenever the block declares
    ``channels``, because the catalogue is an unordered map: nothing in it says which
    plane is which, by design.
    """
    catalogue = block.get("channels")
    provenance = block.get("provenance")
    resolved = _resolve_experimental(
        {k: v for k, v in block.items() if k not in ("channels", "provenance")},
        metadata,
    )
    # `provenance` resolves as a map of independent records (each prunes on its own —
    # see `_resolve_map`; a record is any mapping carrying `source_kind`), and the
    # manifest always carries the spec's flat form, whichever way the config authored
    # it, so everything downstream (validate, the writers, obs) sees one shape.
    if isinstance(provenance, Mapping):
        records = _resolve_map(
            provenance,
            metadata,
            is_entry=lambda child: (
                isinstance(child, Mapping) and "source_kind" in child
            ),
        )
        resolved["provenance"] = _flatten_provenance(records) if records else {}
    elif provenance is not None:
        resolved["provenance"] = provenance
    if catalogue is None:
        return resolved

    # Both shapes a positional catalogue can arrive in: a list (already collapsed by the
    # digit-keys rule somewhere upstream) and the digit-keyed map it collapses from.
    # `IngestConfig` rejects this at load, but `resolve_dca` is public and reached
    # directly, so it owes the same message rather than quietly selecting by "0" / "1".
    if not isinstance(catalogue, Mapping) or (
        catalogue and all(isinstance(k, str) and k.isdigit() for k in catalogue)
    ):
        raise ValueError(
            "[dca.channels] is keyed by position. Key it by the source channel key "
            "instead — [dca.channels.AGP], not [dca.channels.5] — so the binding survives "
            "a reorder; `channel_column_names` fixes the C order."
        )
    if channel_keys is None:
        raise ValueError(
            f"{where}: [dca.channels] is declared but no channel order reached the "
            "resolver, so there is no way to know which catalogue entry is which plane. "
            "Pass `channel_keys`, the store's source channel keys in C order, or drop "
            "the [dca.channels] catalogue from the config."
        )
    resolved["channels"] = _resolve_channels(catalogue, metadata, channel_keys)
    return resolved


def _unwrap_annotation(annotation: Any) -> Any:
    """Strip ``Optional`` / ``Annotated`` down to the type that decides shape."""
    while True:
        origin = typing.get_origin(annotation)
        if origin is typing.Annotated:
            annotation = typing.get_args(annotation)[0]
            continue
        if origin in (typing.Union, types.UnionType):
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(args) != 1:
                return annotation
            annotation = args[0]
            continue
        return annotation


def _as_list_item(annotation: Any) -> Any | None:
    """The item type when *annotation* is a list, else ``None``."""
    inner = _unwrap_annotation(annotation)
    if typing.get_origin(inner) is list:
        args = typing.get_args(inner)
        return args[0] if args else Any
    return None


def _as_model(annotation: Any) -> Any | None:
    """*annotation* as a pydantic model class, else ``None``."""
    inner = _unwrap_annotation(annotation)
    return inner if isinstance(inner, type) and issubclass(inner, BaseModel) else None


def _as_dict_value(annotation: Any) -> Any | None:
    """The value type when *annotation* is a mapping, else ``None``."""
    inner = _unwrap_annotation(annotation)
    if typing.get_origin(inner) is dict:
        args = typing.get_args(inner)
        return args[1] if len(args) == 2 else Any
    return None


def _resolve_against(
    field_type: Any, node: Any, metadata: Mapping[str, Any]
) -> tuple[Any, int, int]:
    """Resolve *node* against its declared *field_type*, returning the pruning triple.

    The schema — not the TOML shape — decides structure, removing three edge cases the
    type-blind :func:`_resolve` must guess at:

    * A **model** field recurses field by field, so ``concentration = {value = {column}}``
      is the ``Concentration.value`` field, not a leaf ``_is_leaf`` would collapse to a scalar.
    * A **list** field is a list by construction (a bare scalar or one-item ``[[...]]`` is
      wrapped/kept), so a column-fed ``study_type`` stays ``["Screen"]`` with no separate
      re-listing repair — this walk is the only shaping step.
    * A **scalar**/unknown field defers to :func:`_resolve`, which still owns the leaf
      grammar (``column`` / ``value`` / ``map``, blank pruning, one-item collapse) — so
      ``map`` canonicalisation and every leaf rule live in one place.
    """
    t = _unwrap_annotation(field_type)

    model = _as_model(t)
    if model is not None:
        # A single model field authored as `[[dca.x.y]]` parses to a one-item list.
        if isinstance(node, list) and len(node) == 1:
            node = node[0]
        if isinstance(node, Mapping):
            fields = model.model_fields
            # A model field mis-authored as a bare leaf sharing no model key (e.g.
            # `concentration = {column = "x"}`) defers to the leaf grammar for a
            # recognisable value + clean validation error. `{value = ...}` is NOT caught —
            # `value` is a real Concentration field — so the `{value = {column}}` fix stands.
            if _is_leaf(node) and not (node.keys() & fields.keys()):
                return _resolve(node, metadata)
            out: dict[Any, Any] = {}
            columns = blank = 0
            for key, child in node.items():
                child_type = fields[key].annotation if key in fields else None
                value, seen, missing = _resolve_against(child_type, child, metadata)
                columns += seen
                blank += missing
                if value is not MISSING:
                    out[key] = value
            if columns and columns == blank:
                return MISSING, columns, blank
            return out, columns, blank
        return _resolve(node, metadata)

    item_type = _as_list_item(t)
    if item_type is not None:
        # `[dca.section.field.0]` authors a list field as a digit-keyed sub-table; convert
        # it to a list BEFORE the wrap below (same rule `_resolve` applies via
        # `_indexed_to_list`), so `{"0": x}` becomes `[x]` rather than the wrap producing
        # `[{"0": x}]` -> a doubly-nested `[[x]]` the spec rejects.
        node = _indexed_to_list(node)
        items = node if isinstance(node, list) else [node]
        parts = [_resolve_against(item_type, item, metadata) for item in items]
        columns = sum(seen for _, seen, _ in parts)
        blank = sum(missing for _, _, missing in parts)
        if columns and columns == blank:
            return MISSING, columns, blank
        kept = [value for value, _, _ in parts if value is not MISSING]
        return kept, columns, blank

    value_type = _as_dict_value(t)
    if value_type is not None and isinstance(node, Mapping):
        out = {}
        columns = blank = 0
        for key, child in node.items():
            value, seen, missing = _resolve_against(value_type, child, metadata)
            columns += seen
            blank += missing
            if value is not MISSING:
                out[key] = value
        if columns and columns == blank:
            return MISSING, columns, blank
        return out, columns, blank

    # Scalar, or a key with no schema type (extra key not in the model): the type-blind
    # leaf grammar handles it (leaf resolution, `map`, blank pruning, one-item collapse).
    return _resolve(node, metadata)


def _resolve_experimental(
    block: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict:
    """Resolve the experimental blocks of a ``[dca]`` block schema-driven, in one path.

    The top level is a container of independent blocks, so — like :func:`resolve` — each
    prunes on its own and the container is never pruned or listified. Every block is
    resolved *against* its declared type through the single :func:`_resolve_against` walk:
    a model field recurses field by field, and any key the model does not declare defers
    to the type-blind :func:`_resolve`. ``provenance`` never reaches this function —
    :func:`resolve_dca` routes it through :func:`_resolve_map` instead, because a
    map of independent records needs the top-level prune exemption one level deeper than
    the dict-value branch's aggregate would grant it (its records carry no list field, so
    the type-blind grammar already leaves their shape spec-correct).

    This one dispatch is why the block reaches :func:`validate_dca` already model-shaped
    (every list field wrapped by construction), so no separate re-listing repair is needed.
    """
    fields = DCAExperimentalMetadataV0_2.model_fields
    out: dict[Any, Any] = {}
    for key, child in block.items():
        annotation = fields[key].annotation if key in fields else None
        value, _, _ = _resolve_against(annotation, child, metadata)
        if value is not MISSING:
            out[key] = value
    return out


def _every_line_is_should(message: str) -> bool:
    """True when every non-blank line of *message* is a SHOULD-severity finding.

    A model validator that accumulates findings raises them newline-joined, and
    ``_convert_errors`` splits them back out one Issue per line. This applies the same
    split, so a validator that raises a MUST and a SHOULD together is correctly read as
    fatal rather than waved through on the strength of its second line.
    """
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    return bool(lines) and all(line.startswith(_SHOULD_MARKER) for line in lines)


@lru_cache(maxsize=1)
def _normalising_model() -> type:
    """The spec model with its SHOULD-raising validators neutralised.

    **For normalisation only.** The gate is the issue list; see :func:`validate_dca`.

    Two of v0.2.4's rules are SHOULDs that ``dca_helpers`` implements as a *raising*
    model validator rather than as entries in the issue list — ``tissue`` must carry a
    ``CVCL:`` accession when ``tissue_type`` is ``cell line``, and ``development_stage``
    must use the organism's stage ontology. The perturbation identifiers that are beta
    in v0.2 (``chemical.pubchem_cid`` / ``chemical.morgan_fingerprint``,
    ``genetic.ensembl_gene_id``) are raised the same way, from
    ``PerturbationAssignment`` for its ``definition`` and ``cotreatment``.
    ``dca_helpers`` documents the consequence on the validator itself: "a consumer
    constructing the model directly to *read* metadata would see an exception where the
    report says warning." This module is that consumer, so it constructs a variant that
    does not raise on them.

    It overrides the two SHOULD methods and nothing else, so every MUST rule in the tree
    still fires: the two on ``Biosample`` itself, the non-``empty`` ⇒ ``definition``
    rule on ``PerturbationAssignment``, and the ones on ``Study``, ``Perturbation`` and
    ``FieldProvenance``, which are untouched. And it re-raises anything that is not
    wholly SHOULD, so a MUST added to either validator later is still fatal rather than
    silently swallowed.

    Built lazily and cached, so the subclasses cost a few hundred microseconds once per
    process. If ``_BiosampleV0_2`` or ``_PerturbationAssignmentV0_2`` ever disappears from
    the validation package, importing this module raises ``ImportError`` naming the
    symbol, which is the right failure for a coupling this deliberate.
    """

    class _NormalisingBiosample(_BiosampleV0_2):  # type: ignore[misc, valid-type]
        @model_validator(mode="after")
        def _cross_slot_should_warnings(self):
            try:
                return super()._cross_slot_should_warnings()
            except ValueError as exc:
                if _every_line_is_should(str(exc)):
                    return self
                raise

    class _NormalisingAssignment(_PerturbationAssignmentV0_2):  # type: ignore[misc, valid-type]
        @model_validator(mode="after")
        def _beta_identifier_warnings(self):
            try:
                return super()._beta_identifier_warnings()
            except ValueError as exc:
                if _every_line_is_should(str(exc)):
                    return self
                raise

    class _NormalisingExperimentalMetadata(DCAExperimentalMetadataV0_2):
        biosample: _NormalisingBiosample
        perturbation: _NormalisingAssignment | None = None

    return _NormalisingExperimentalMetadata


def dca_read_columns(block: Any) -> frozenset[str]:
    """Every source column name any ``[dca.*]`` leaf READS, anywhere in the tree.

    A leaf spec is a mapping carrying ``column`` (see :data:`_SPEC_KEYS` /
    :func:`_is_leaf`). The walk is fully recursive, so it covers
    ``[[dca.<block>.additional_metadata]]`` entries, ``[dca.channels.<key>]`` names,
    nested ``[dca.provenance.*]`` tables and the nested perturbation blocks alike.

    Deliberately keyed on the presence of ``column`` rather than on :func:`_is_leaf`:
    an entry that is *malformed* as a spec still names a column its author meant to
    read, and for the callers below (which use this to decide what NOT to duplicate)
    over-collecting is the safe direction.
    """
    out: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, Mapping):
            column = node.get("column")
            if isinstance(column, str) and column.strip():
                out.add(column.strip())
            for child in node.values():
                _walk(child)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)

    _walk(block)
    return frozenset(out)


def validate_dca(block: dict, *, where: str, warnings: list[str] | None = None) -> dict:
    """Validate *block* and return it normalised, or raise.

    Two jobs, and keeping them apart is the whole point of this function's shape:

    **The gate** is ``validate_experimental_metadata_v0_2``, whose issue list carries a
    severity per finding. Only MUST-severity findings are fatal. SHOULD findings are
    logged and the build proceeds — a recommendation is not a reason to refuse to
    describe a dataset.

    **The normalisation** runs the block back through the spec's own models, which is
    what turns the ``"0.5"`` a ``dtype=str`` platemap hands ``Concentration.value`` into
    a real float, and what fills each ``provenance`` entry's required ``field_path``
    from its map key. Pydantic's *lax* mode does the coercion; strict mode is not an
    option, because it rejects every enum-from-string and date-from-string a TOML config
    can produce.

    These used to be **one** ``model_validate`` call, which made the gate a side effect
    of the transform. Pydantic cannot coerce a value without first constructing the
    model, so a SHOULD-severity validator that raises took the coercion down with it and
    aborted the whole manifest build — for JUMP-CP, on a human-reviewed ``tissue`` term,
    with a message that read ``[SHOULD]``. The issue list had already classified that
    same finding as a warning, because it runs the identical ``model_validate`` and maps
    severity from the message prefix. So the second call added no coverage and only
    escalated. Normalisation now goes through :func:`_normalising_model`, and severity is
    decided in exactly one place.

    Note what this does *not* relax: the model still raises on every MUST, and that
    raise is still wrapped so it names the group — a bare pydantic ``ValidationError``
    says what is wrong but not which of a million groups it came from.

    ``channels`` is excluded from the check and carried through untouched: it is not
    part of ``ExperimentalMetadata`` and the validator rejects it as an unknown block.
    The store-level ``dca-validate`` checks it after the write.

    *block* is expected to be :func:`resolve_dca` output, which is already model-shaped —
    :func:`_resolve_experimental` wraps every list field against the schema by
    construction, so this function does **no** structural repair of its own; the block is
    validated and normalised as it arrives.

    *warnings* lets the caller decide how SHOULD findings surface. When a list is passed,
    each finding is appended to it as ``"<field path>: <message>"`` and nothing is logged,
    so a pipeline that validates once per field of view can deduplicate and report them
    once. When it is ``None``, the findings are logged as one warning naming *where*.
    """
    experimental = {k: v for k, v in block.items() if k != "channels"}

    errors: list[Any] = []
    shoulds: list[Any] = []
    for issue in validate_experimental_metadata_v0_2(experimental):
        severity = getattr(issue.severity, "value", issue.severity)
        (errors if severity == "error" else shoulds).append(issue)

    def locate(issue: Any) -> str:
        return ".".join(map(str, issue.loc))

    def render(issues: list[Any]) -> str:
        return "; ".join(f"{locate(issue)}: {issue.message}" for issue in issues)

    if errors:
        raise DcaValidationError(
            where,
            [f"{locate(issue)}: {issue.message}" for issue in errors],
            [locate(issue) for issue in errors],
        )
    if warnings is not None:
        warnings.extend(f"{locate(issue)}: {issue.message}" for issue in shoulds)
    elif shoulds:
        logger.warning("DCA metadata warnings for %s: %s", where, render(shoulds))

    try:
        model = _normalising_model().model_validate(experimental)
    except ValidationError as exc:
        details = exc.errors()
        raise DcaValidationError(
            where,
            [f"{'.'.join(map(str, d['loc']))}: {d['msg']}" for d in details],
            [".".join(map(str, d["loc"])) for d in details],
        ) from exc
    except ValueError as exc:
        raise DcaValidationError(where, [str(exc)], []) from exc
    normalised = model.model_dump(mode="json", exclude_none=True)
    if "channels" in block:
        normalised["channels"] = block["channels"]
    return normalised


__all__ = [
    "DCA_KEY",
    "MISSING",
    "DcaValidationError",
    "dca_read_columns",
    "resolve",
    "resolve_dca",
    "validate_dca",
]
