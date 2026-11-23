"""
Builders for v0.3 store-level tables (docs/v0.3/tables.rst), for the table tests.

``add_condition_table`` writes ``tables/obs`` and the ``tables/zarr.json`` listing;
``add_table`` writes any other table group with its ``table.parquet``. Both rewrite
the listing to name every table group present, so a test breaks the listing only on
purpose.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CONDITION_ATTRS = {
    "type": "condition_table",
    "table_version": "1",
    "backend": "parquet",
    "index_key": "zarr_path",
    "index_type": "str",
}


def object_attrs(label: str, table_type: str = "masking_roi_table") -> dict:
    return {
        "type": table_type,
        "table_version": "1",
        "backend": "parquet",
        "index_key": "label",
        "index_type": "int",
        "instance_key": "label",
        "region": {"path": f"../labels/{label}"},
    }


def _write_group(path: Path, attrs: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attrs})
    )


def write_listing(store: Path, names: "list[str] | None" = None) -> None:
    """Write tables/zarr.json; by default it lists every table group present."""
    tables = store / "tables"
    if names is None:
        names = sorted(p.name for p in tables.iterdir() if p.is_dir())
    _write_group(tables, {"tables": names})


def add_table(
    store: Path,
    name: str,
    attrs: dict,
    columns: "dict[str, list] | pa.Table | None",
    *,
    filename: str = "table.parquet",
) -> None:
    """Write tables/<name>/ (zarr.json + Parquet data) and refresh the listing.
    ``columns`` None writes no data file."""
    path = store / "tables" / name
    _write_group(path, attrs)
    if columns is not None:
        table = columns if isinstance(columns, pa.Table) else pa.table(columns)
        pq.write_table(table, path / filename)
    write_listing(store)


def add_condition_table(store: Path, zarr_paths: list[str], **kwargs) -> None:
    add_table(store, "obs", CONDITION_ATTRS, {"zarr_path": zarr_paths}, **kwargs)


def object_rows(
    zarr_paths: list[str], labels: list, *, plate: bool = True, **extra: list
) -> dict[str, list]:
    """Key columns of a table of objects: zarr_path, label, t (all 0), and the
    plate's row / column / path_in_well derived from zarr_path."""
    rows: dict[str, list] = {"zarr_path": zarr_paths, "label": labels}
    rows["t"] = [0] * len(labels)
    if plate:
        parts = [p.split("/") for p in zarr_paths]
        rows["row"] = [p[0] for p in parts]
        rows["column"] = [p[1] for p in parts]
        rows["path_in_well"] = [p[2] for p in parts]
    rows.update(extra)
    return rows
