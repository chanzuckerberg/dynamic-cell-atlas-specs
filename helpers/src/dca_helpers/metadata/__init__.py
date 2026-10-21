"""Shared DCA metadata engine: resolve a ``[dca.*]`` config block per group."""

from dca_helpers.metadata.attributes import (
    DCA_SCHEMA_URL,
    merge_dca_attributes,
    write_normalization_statistics,
)
from dca_helpers.metadata.config import load_dca_config
from dca_helpers.metadata.propose import (
    Proposal,
    propose,
    search_cellosaurus,
    search_ncbitaxon,
)
from dca_helpers.metadata.resolve import (
    DCA_KEY,
    MISSING,
    DcaValidationError,
    dca_read_columns,
    resolve,
    resolve_dca,
    validate_dca,
)
from dca_helpers.metadata.store_facts import STORE_COLUMNS

__all__ = [
    "DCA_KEY",
    "DCA_SCHEMA_URL",
    "MISSING",
    "STORE_COLUMNS",
    "DcaValidationError",
    "Proposal",
    "dca_read_columns",
    "load_dca_config",
    "merge_dca_attributes",
    "propose",
    "resolve",
    "resolve_dca",
    "search_cellosaurus",
    "search_ncbitaxon",
    "validate_dca",
    "write_normalization_statistics",
]
