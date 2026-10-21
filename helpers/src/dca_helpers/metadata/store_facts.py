"""Facts a store carries that a config can read as ``{ column = ... }`` leaves."""

import re
from datetime import date

#: Reserved column names the engine supplies for every store.
STORE_COLUMNS: frozenset[str] = frozenset(
    {
        "store_dataset_name",
        "store_group_id",
        "store_name_date",
        "store_time_increment_s",
    }
)

_LEADING_DATE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_")
_TRAILING_DATE = re.compile(r"_(\d{4})(\d{2})(\d{2})$")


def name_date(dataset_name: str) -> str | None:
    """Return the ISO date in *dataset_name*, or None when it has no clear one.

    Only two forms count: a leading ``YYYY_MM_DD_`` prefix, or a final ``_YYYYMMDD``
    token. A token that is not a real calendar date returns None.
    """
    match = _LEADING_DATE.match(dataset_name) or _TRAILING_DATE.search(dataset_name)
    if match is None:
        return None
    try:
        return date(*(int(part) for part in match.groups())).isoformat()
    except ValueError:
        return None
