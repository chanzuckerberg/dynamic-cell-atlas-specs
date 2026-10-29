"""Facts a store carries that a config can read as ``{ column = ... }`` leaves.

Pipelines fill these reserved columns from their own stores, including ``store_name_date``
from their own naming convention.
"""

#: Reserved column names every pipeline fills for each of its stores.
STORE_COLUMNS: frozenset[str] = frozenset(
    {
        "store_dataset_name",
        "store_group_id",
        "store_name_date",
        "store_time_increment_s",
    }
)
