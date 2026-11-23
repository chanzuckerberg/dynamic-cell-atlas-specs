"""
Pydantic model for the attributes of a v0.3 table group (docs/v0.3/tables.rst).

``StoreTableV0_3`` mirrors ``docs/v0.3/schema.json`` ``$defs/StoreTable``: ngio's
table attributes with the Parquet backend, the conditional keys of a table of
objects (``feature_table`` / ``masking_roi_table``) and of the condition table, and
no ``dca`` block. ``tests/validation/versions/v0_3/test_tables_model.py`` checks that
the two agree.

Fields that the schema declares but does not require default to ``None`` without
being validated, so an explicit ``null`` is rejected (as the schema's ``type`` does)
while an absent key is accepted. Other ngio attributes (e.g. ``categorical_columns``)
are allowed, as the schema allows them.

The checks that read the store (listing, Parquet columns, the condition table's rows)
live in ``validation/tables.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TableType = Literal[
    "feature_table",
    "masking_roi_table",
    "roi_table",
    "condition_table",
    "generic_table",
]

# Tables of objects: they describe the objects of one label image (tables.rst).
OBJECT_TABLE_TYPES: frozenset[str] = frozenset({"feature_table", "masking_roi_table"})


class StoreTableRegionV0_3(BaseModel):
    """``region`` of a table of objects: the label image, ``../labels/<label_name>``."""

    model_config = ConfigDict(extra="allow")

    path: str = Field(pattern=r"^\.\./labels/[^/]+$")


class StoreTableV0_3(BaseModel):
    """Attributes of ``<store>/tables/<name>/zarr.json``."""

    model_config = ConfigDict(extra="allow")

    type: TableType
    table_version: Literal["1"]
    backend: Literal["parquet"]
    index_key: str = Field(default=None, min_length=1)
    index_type: Literal["int", "str"] = None
    instance_key: str = Field(default=None, min_length=1)
    region: StoreTableRegionV0_3 = None

    @model_validator(mode="before")
    @classmethod
    def _no_dca_block(cls, data: object) -> object:
        if isinstance(data, dict) and "dca" in data:
            raise ValueError(
                "a table group carries ngio's attributes only, not a 'dca' block"
            )
        return data

    @model_validator(mode="after")
    def _conditional_keys(self) -> "StoreTableV0_3":
        if self.type in OBJECT_TABLE_TYPES:
            required = {"index_key", "index_type", "instance_key", "region"}
            consts = {"index_key": "label", "instance_key": "label"}
        elif self.type == "condition_table":
            required = {"index_key", "index_type"}
            consts = {"index_key": "zarr_path", "index_type": "str"}
        else:
            return self
        missing = sorted(k for k in required if getattr(self, k) is None)
        if missing:
            raise ValueError(f"a {self.type} requires {missing}")
        wrong = [
            f"{k} must be {v!r}, got {getattr(self, k)!r}"
            for k, v in consts.items()
            if getattr(self, k) != v
        ]
        if wrong:
            raise ValueError(f"{self.type}: " + "; ".join(wrong))
        return self
