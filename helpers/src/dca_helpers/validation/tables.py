"""
Store-level table checks (docs/v0.3/tables.rst), for v0.3 stores only.

Every table is an ngio table at the store root, ``<store>/tables/<name>/``, and
``<store>/tables/obs`` is the store's condition table. These checks run once per
store, after its image groups are validated, from a few metadata reads: the
``tables/`` listing, each table group's ``zarr.json``, each ``table.parquet`` footer,
and, for a table of objects, one label image's ``zarr.json`` and level-0 array
metadata. Column values are read only where a rule needs them, and then only the key
columns: ``zarr_path`` of the condition table, and ``zarr_path`` / ``t`` / ``label``
of an object index (``label`` of a ``feature_table`` only when the Parquet statistics
cannot show that every label is > 0).

ERROR (MUST):

* the store has no ``tables/obs`` condition table, or ``tables/obs`` keeps the v0.2
  ``obs.parquet`` instead of ``table.parquet``; ``tables/obs`` is not a
  ``condition_table``, or another table is one;
* ``tables/zarr.json`` is missing, or its ``tables`` attribute does not list exactly
  the table groups present;
* a table group has no ``zarr.json``, or its attributes fail ``StoreTable``
  (``versions/v0_3/tables.py``);
* ``table.parquet`` is missing or unreadable;
* the condition table has no ``zarr_path`` column, a ``zarr_path`` twice, or a set of
  ``zarr_path`` values that differs from the store's image groups (for an HCS plate,
  the declared fields of its wells; a store whose root is the image group has one
  row). When only some wells were opened (``--wells``, or a well that failed to open),
  only the presence of the opened wells' fields is checked;
* a table of objects (``feature_table``, ``masking_roi_table``) has no ``zarr_path``
  or ``label`` column, a ``label`` that is not an integer or is null or <= 0, only one
  of ``parent_label_name`` / ``parent_label``, or no ``t`` column while its label
  image has more than one timepoint (read from the first validated image group that
  has that label image; skipped when none does);
* a ``masking_roi_table`` has two rows with the same (``zarr_path``, ``t``,
  ``label``) (``t`` left out of the key when the table has no ``t`` column);
* a validated image group of an HCS plate has its own ``tables`` group (reported by
  the caller; for a store whose root is the image group, the two locations are the
  same).

WARNING (SHOULD):

* a table of objects in a plate store lacks ``row``, ``column`` or ``path_in_well``;
* a label image of a validated image group has no ``masking_roi_table`` whose
  ``region`` is ``../labels/<label_name>``. The spec requires an object index only
  for an instance label image and records no label kind, so a semantic mask
  legitimately has none.
"""

from __future__ import annotations

import io
import os
from collections import Counter
from collections.abc import Collection, Iterable

import obstore
import obstore.exceptions as _obs_exc
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pydantic

from dca_helpers.validation.discovery import _is_s3, _s3_store, open_zarr_group
from dca_helpers.validation.result import Issue, Severity
from dca_helpers.validation.versions.v0_3.tables import (
    OBJECT_TABLE_TYPES,
    StoreTableV0_3,
)

# Spec versions that define store-level tables (tables.rst, new in v0.3).
TABLES_SPEC_VERSIONS: frozenset[str] = frozenset({"0.3"})

_NOT_FOUND = (FileNotFoundError, _obs_exc.NotFoundError)
_PLATE_KEY_COLUMNS = ("row", "column", "path_in_well")
_REGION_PREFIX = "../labels/"
_EXAMPLES = 5


def _err(loc: tuple, message: str) -> Issue:
    return Issue(loc=loc, message=message, severity=Severity.ERROR)


def _warn(loc: tuple, message: str) -> Issue:
    return Issue(loc=loc, message=message, severity=Severity.WARNING)


def _some(values: Iterable) -> str:
    """A sorted, truncated rendering of `values` for an issue message."""
    items = sorted(values, key=str)
    shown = ", ".join(repr(v) for v in items[:_EXAMPLES])
    more = len(items) - _EXAMPLES
    return f"{shown} (and {more} more)" if more > 0 else shown


def _join(base: str, *parts: str) -> str:
    return "/".join([base.rstrip("/"), *(p for p in parts if p)])


# ---------------------------------------------------------------------------
# Store access (local or S3 via obstore)
# ---------------------------------------------------------------------------


def _list_child_dirs(path: str) -> "list[str] | None":
    """Names of the sub-groups (directories / common prefixes) under `path`, or
    None when nothing exists there. One LIST call on S3."""
    if _is_s3(path):
        store, key = _s3_store(path)
        listing = obstore.list_with_delimiter(store, prefix=f"{key.rstrip('/')}/")
        if not listing["common_prefixes"] and not listing["objects"]:
            return None
        return sorted(
            p.rstrip("/").rsplit("/", 1)[-1] for p in listing["common_prefixes"]
        )
    if not os.path.isdir(path):
        return None
    return sorted(e.name for e in os.scandir(path) if e.is_dir())


def _exists(path: str) -> bool:
    if _is_s3(path):
        store, key = _s3_store(path)
        try:
            obstore.head(store, key)
        except _obs_exc.NotFoundError:
            return False
        return True
    return os.path.isfile(path)


class _ObstoreFile(io.RawIOBase):
    """A seekable file over an obstore reader, so pyarrow reads only the byte
    ranges it asks for (the footer, then the projected column chunks)."""

    def __init__(self, reader: "obstore.ReadableFile") -> None:
        self._reader = reader

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._reader.seek(offset, whence)

    def tell(self) -> int:
        return self._reader.tell()

    def read(self, size: int = -1) -> bytes:
        return bytes(self._reader.read(None if size is None or size < 0 else size))

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _open_parquet(path: str) -> pq.ParquetFile:
    """Open a Parquet file, reading only its footer."""
    if _is_s3(path):
        store, key = _s3_store(path)
        return pq.ParquetFile(_ObstoreFile(obstore.open_reader(store, key)))
    return pq.ParquetFile(path)


def _group_attrs(path: str) -> dict:
    """The attributes of the group at `path`; raises a _NOT_FOUND error when it
    has no zarr.json."""
    return dict(open_zarr_group(path).attrs)


def has_tables_group(image_path: str) -> bool:
    """Whether an image group has its own ``tables`` group (one GET)."""
    try:
        open_zarr_group(_join(image_path, "tables"))
    except Exception:
        return False
    return True


def _label_time_size(label_path: str) -> "int | None":
    """The t size of a label image's level 0, from its zarr.json and its level-0
    array's metadata; 1 when it has no time axis, None when unreadable."""
    try:
        group = open_zarr_group(label_path)
        ms = dict(group.attrs)["ome"]["multiscales"][0]
        axes = ms["axes"]
        t_index = next(
            (
                i
                for i, a in enumerate(axes)
                if a.get("type") == "time" or str(a.get("name", "")).lower() == "t"
            ),
            None,
        )
        if t_index is None:
            return 1
        return int(group[ms["datasets"][0]["path"]].shape[t_index])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Column checks
# ---------------------------------------------------------------------------


def _label_stats(pf: pq.ParquetFile) -> "tuple[object, int] | None":
    """(min, null count) of the ``label`` column from the row-group statistics,
    or None when any row group lacks them."""
    md = pf.metadata
    idx = next(
        (i for i in range(md.num_columns) if md.schema.column(i).path == "label"),
        None,
    )
    if idx is None:
        return None
    low, nulls = None, 0
    for rg in range(md.num_row_groups):
        if md.row_group(rg).num_rows == 0:
            continue
        stats = md.row_group(rg).column(idx).statistics
        if stats is None or not stats.has_min_max or not stats.has_null_count:
            return None
        low = stats.min if low is None else min(low, stats.min)
        nulls += stats.null_count
    return (1 if low is None else low), nulls


def _bad_label_count(pf: pq.ParquetFile, column: "pa.ChunkedArray | None") -> int:
    """Rows whose ``label`` is null or <= 0."""
    if column is None:
        stats = _label_stats(pf)
        if stats is not None and stats[0] > 0 and stats[1] == 0:
            return 0
        column = pf.read(columns=["label"]).column(0)
    nonpositive = pc.sum(pc.less_equal(column, 0)).as_py() or 0
    return column.null_count + nonpositive


def _is_integer(dtype: pa.DataType) -> bool:
    if pa.types.is_dictionary(dtype):
        dtype = dtype.value_type
    return pa.types.is_integer(dtype)


def _check_condition_table(
    pf: pq.ParquetFile,
    image_groups: "Collection[str] | None",
    image_groups_complete: bool,
) -> list[Issue]:
    loc = ("tables", "obs")
    if "zarr_path" not in pf.schema_arrow.names:
        return [_err(loc, "tables/obs has no 'zarr_path' column")]
    if image_groups is None:
        rows = pf.metadata.num_rows
        if rows != 1:
            return [
                _err(
                    loc,
                    f"tables/obs has {rows} rows; a store whose root is its image "
                    f"group has one image group, so one row",
                )
            ]
        return []
    values = pf.read(columns=["zarr_path"]).column(0).to_pylist()
    issues: list[Issue] = []
    duplicates = [v for v, n in Counter(values).items() if n > 1]
    if duplicates:
        issues.append(
            _err(
                loc + ("zarr_path",),
                f"tables/obs has more than one row for zarr_path {_some(duplicates)}",
            )
        )
    present = set(values)
    missing = set(image_groups) - present
    if missing:
        issues.append(
            _err(
                loc + ("zarr_path",),
                f"tables/obs has no row for {len(missing)} image group(s): "
                f"{_some(missing)}",
            )
        )
    if image_groups_complete:
        extra = present - set(image_groups)
        if extra:
            issues.append(
                _err(
                    loc + ("zarr_path",),
                    f"tables/obs has rows for {len(extra)} zarr_path value(s) that "
                    f"are not image groups of the store: {_some(extra)}",
                )
            )
    return issues


def _check_object_table(
    name: str,
    table_type: str,
    pf: pq.ParquetFile,
    *,
    is_plate: bool,
    label_t_size: "int | None",
) -> list[Issue]:
    loc = ("tables", name)
    schema = pf.schema_arrow
    columns = set(schema.names)
    issues: list[Issue] = []
    for key in ("zarr_path", "label"):
        if key not in columns:
            issues.append(
                _err(loc + (key,), f"{table_type} tables/{name} has no '{key}' column")
            )

    label_ok = "label" in columns and _is_integer(schema.field("label").type)
    if "label" in columns and not label_ok:
        issues.append(
            _err(
                loc + ("label",),
                f"'label' of tables/{name} must be an integer column, "
                f"got {schema.field('label').type}",
            )
        )

    if ("parent_label_name" in columns) != ("parent_label" in columns):
        issues.append(
            _err(
                loc,
                f"tables/{name} has only one of 'parent_label_name' and "
                f"'parent_label'; they are given together or not at all",
            )
        )

    if is_plate:
        missing = [c for c in _PLATE_KEY_COLUMNS if c not in columns]
        if missing:
            issues.append(
                _warn(
                    loc,
                    f"tables/{name} in a plate should have the key columns "
                    f"{missing}",
                )
            )

    if "t" not in columns and label_t_size is not None and label_t_size > 1:
        issues.append(
            _err(
                loc + ("t",),
                f"tables/{name} has no 't' column, but its label image has "
                f"{label_t_size} timepoints",
            )
        )

    label_column = None
    if table_type == "masking_roi_table" and {"zarr_path", "label"} <= columns:
        keys = [k for k in ("zarr_path", "t", "label") if k in columns]
        key_table = pf.read(columns=keys)
        distinct = key_table.group_by(keys).aggregate([]).num_rows
        repeated = key_table.num_rows - distinct
        if repeated:
            issues.append(
                _err(
                    loc,
                    f"object index tables/{name} has {repeated} row(s) that repeat "
                    f"another row's ({', '.join(keys)})",
                )
            )
        label_column = key_table.column("label")

    if label_ok:
        bad = _bad_label_count(pf, label_column)
        if bad:
            issues.append(
                _err(
                    loc + ("label",),
                    f"tables/{name} has {bad} row(s) whose 'label' is null or <= 0",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _region_label(attrs: dict) -> "str | None":
    region = attrs.get("region")
    path = region.get("path") if isinstance(region, dict) else None
    if isinstance(path, str) and path.startswith(_REGION_PREFIX):
        label = path[len(_REGION_PREFIX) :]
        if label and "/" not in label:
            return label
    return None


def _check_listing(tables_path: str, present: list[str]) -> list[Issue]:
    loc = ("tables", "zarr.json")
    try:
        attrs = _group_attrs(tables_path)
    except _NOT_FOUND:
        return [
            _err(
                loc,
                "tables/ has no zarr.json; it must list every table of the store "
                "in a 'tables' attribute",
            )
        ]
    listed = attrs.get("tables")
    if not isinstance(listed, list) or not all(isinstance(n, str) for n in listed):
        return [_err(loc, "tables/zarr.json must have a 'tables' list of names")]
    issues = []
    absent = set(listed) - set(present)
    if absent:
        issues.append(
            _err(
                loc,
                f"tables/zarr.json lists table(s) with no group: {_some(absent)}",
            )
        )
    unlisted = set(present) - set(listed)
    if unlisted:
        issues.append(
            _err(
                loc,
                f"tables/zarr.json does not list table group(s): {_some(unlisted)}",
            )
        )
    return issues


def _model_issues(name: str, attrs: dict) -> list[Issue]:
    try:
        StoreTableV0_3.model_validate(attrs)
    except pydantic.ValidationError as exc:
        return [
            _err(
                ("tables", name, *e.get("loc", ())),
                f"tables/{name}: {e.get('msg', '').removeprefix('Value error, ')}",
            )
            for e in exc.errors(include_url=False)
        ]
    return []


def check_store_tables(
    store_path: str,
    *,
    is_plate: bool,
    image_groups: "Collection[str] | None",
    image_groups_complete: bool,
    labels_by_image: "dict[str, set[str]]",
) -> list[Issue]:
    """Check a v0.3 store's tables (see the module docstring).

    image_groups: the store's image groups as ``zarr_path`` values, relative to the
        store root (an HCS plate's declared on-disk fields, e.g. ``A/1/0``); None for
        a store whose root is its single image group.
    image_groups_complete: whether image_groups is every image group of the store;
        when False only their presence in the condition table is checked.
    labels_by_image: label image names of each validated image group, keyed by
        ``zarr_path`` (``""`` for a store whose root is the image group).
    """
    try:
        return _check_store_tables(
            store_path,
            is_plate=is_plate,
            image_groups=image_groups,
            image_groups_complete=image_groups_complete,
            labels_by_image=labels_by_image,
        )
    except _obs_exc.BaseError as exc:
        return [
            _err(
                ("tables", "s3_io"),
                f"S3 access error (network/throttling/auth) reading tables: {exc}",
            )
        ]


def _check_store_tables(
    store_path: str,
    *,
    is_plate: bool,
    image_groups: "Collection[str] | None",
    image_groups_complete: bool,
    labels_by_image: "dict[str, set[str]]",
) -> list[Issue]:
    issues: list[Issue] = []
    tables_path = _join(store_path, "tables")
    present = _list_child_dirs(tables_path)
    if present is not None:
        issues.extend(_check_listing(tables_path, present))

    indexed_labels: set[str] = set()
    obs_ok = False
    for name in present or []:
        table_path = _join(tables_path, name)
        loc = ("tables", name)
        try:
            attrs = _group_attrs(table_path)
        except _NOT_FOUND:
            issues.append(_err(loc, f"tables/{name}/ has no zarr.json"))
            continue
        except Exception as exc:
            issues.append(_err(loc, f"tables/{name}/zarr.json is unreadable: {exc}"))
            continue
        issues.extend(_model_issues(name, attrs))
        table_type = attrs.get("type")

        if name == "obs":
            if table_type != "condition_table":
                issues.append(
                    _err(
                        loc + ("type",),
                        f"tables/obs is reserved for the condition table; its type "
                        f"must be 'condition_table', got {table_type!r}",
                    )
                )
            obs_ok = True
        elif table_type == "condition_table":
            issues.append(
                _err(
                    loc + ("type",),
                    f"tables/{name} is a condition_table; a store has exactly one, "
                    f"tables/obs",
                )
            )

        parquet_path = _join(table_path, "table.parquet")
        try:
            pf = _open_parquet(parquet_path)
        except _NOT_FOUND:
            if name == "obs" and _exists(_join(table_path, "obs.parquet")):
                message = (
                    "tables/obs has the v0.2 obs.parquet instead of table.parquet; "
                    "a v0.3 condition table keeps its data in "
                    "tables/obs/table.parquet, the file ngio reads"
                )
            else:
                message = f"tables/{name} has no table.parquet"
            issues.append(_err(loc + ("table.parquet",), message))
            continue
        except _obs_exc.BaseError:
            raise
        except Exception as exc:
            issues.append(
                _err(
                    loc + ("table.parquet",),
                    f"tables/{name}/table.parquet is unreadable: {exc}",
                )
            )
            continue

        if name == "obs":
            issues.extend(
                _check_condition_table(pf, image_groups, image_groups_complete)
            )
        if table_type in OBJECT_TABLE_TYPES:
            label = _region_label(attrs)
            if table_type == "masking_roi_table" and label is not None:
                indexed_labels.add(label)
            sample = next(
                (g for g, names in labels_by_image.items() if label in names), None
            )
            t_size = (
                _label_time_size(_join(store_path, sample, "labels", label))
                if sample is not None
                else None
            )
            issues.extend(
                _check_object_table(
                    name, table_type, pf, is_plate=is_plate, label_t_size=t_size
                )
            )

    if not obs_ok:
        issues.append(
            _err(
                ("tables", "obs"),
                "the store has no tables/obs condition table; every v0.3 store "
                "has one, with one row per image group",
            )
        )

    all_labels = set().union(*labels_by_image.values()) if labels_by_image else set()
    for label in sorted(all_labels - indexed_labels):
        issues.append(
            _warn(
                ("tables", f"{label}_ROI_table"),
                f"label image '{label}' has no object index (a masking_roi_table "
                f"with region '../labels/{label}'); every instance label image "
                f"must have one",
            )
        )
    return issues
