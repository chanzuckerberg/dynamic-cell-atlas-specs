"""Shared DCA metadata engine: resolve a ``[dca.*]`` config block per group."""

from dca_helpers.metadata.attributes import (
    merge_dca_attributes,
    write_normalization_statistics,
)
from dca_helpers.metadata.config import load_dca_config
from dca_helpers.metadata.resolve import (
    DCA_KEY,
    MISSING,
    DcaValidationError,
    dca_read_columns,
    resolve,
    resolve_dca,
    validate_dca,
)

__all__ = [
    "DCA_KEY",
    "MISSING",
    "DcaValidationError",
    "dca_read_columns",
    "load_dca_config",
    "merge_dca_attributes",
    "resolve",
    "resolve_dca",
    "validate_dca",
    "write_normalization_statistics",
]
