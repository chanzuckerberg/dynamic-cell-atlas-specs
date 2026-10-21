"""Shared DCA metadata engine: resolve a ``[dca.*]`` config block per group."""

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
    "resolve",
    "resolve_dca",
    "validate_dca",
]
