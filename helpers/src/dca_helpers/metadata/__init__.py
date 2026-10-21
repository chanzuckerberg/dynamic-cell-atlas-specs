"""Shared DCA metadata engine that resolves a ``[dca.*]`` config block per group.

:func:`validate_dca` checks the experimental block only, not ``channels`` or the
normalization statistics. After writing a store, gate it with
``dca_helpers.validation.runner.validate(<store>, spec_version="0.2")``, which checks the
full ``dca`` block, channels and statistics included.

Pass ``warnings=`` a list to collect SHOULD findings from :func:`validate_dca` instead of
logging them, and ``run_attrs=`` to have :func:`merge_dca_attributes` record the writing
tool's version and git sha beside the ``dca`` block.
"""

from dca_helpers.metadata.attributes import (
    DCA_SCHEMA_URL,
    merge_dca_attributes,
    write_normalization_statistics,
)
from dca_helpers.metadata.config import load_dca_config
from dca_helpers.metadata.perturbation_ids import (
    ensembl_from_ncbi_gene,
    morgan_fingerprint,
    search_ensembl,
    search_pubchem,
)
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
    "ensembl_from_ncbi_gene",
    "load_dca_config",
    "merge_dca_attributes",
    "morgan_fingerprint",
    "propose",
    "resolve",
    "resolve_dca",
    "search_cellosaurus",
    "search_ensembl",
    "search_ncbitaxon",
    "search_pubchem",
    "validate_dca",
    "write_normalization_statistics",
]
