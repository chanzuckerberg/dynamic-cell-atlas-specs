# Dynamic Cell Atlas Specifications

This repo defines standards for storing diverse 5D fluorescence microscopy datasets (3D+time+channels) in a unified format for AI model training, using Zarr v3, OME-ngff 0.5 and additional metadata.

For more detailed information on scope, use cases etc. refer to the [documentation](https://chanzuckerberg.github.io/dynamic-cell-atlas-specs/).

## Contents

- [Specification](docs/v0.2/) and JSON schema
- [Data Validator](helpers/src/dca_helpers/validation) using Pydantic
- [Helper tooling](helpers/src/dca_helpers) for parts of the specification that go beyond general OME-ngff

## A pipeline validates every FOV, then writes the DCA block

A pipeline resolves one `[dca.*]` TOML block per FOV with `dca_helpers.metadata`, validates every FOV, and only then writes. The engine exports `load_dca_config`, `resolve_dca`, `validate_dca`, `DcaValidationError`, `merge_dca_attributes`, `write_normalization_statistics`, `DCA_SCHEMA_URL` and `STORE_COLUMNS`. `SPEC_VERSION` lives in `dca_helpers.metadata.attributes`.

`base.toml` holds what is known up front. A leaf is a literal or `{ column = "..." }`, which is read from the per-FOV row (`STORE_COLUMNS` lists the columns the store itself supplies). Each channel catalogue entry needs a `name`.

```toml
[dca.study]
source = "Example Lab"
study_type = ["Screen"]
description = "U2OS live-cell imaging example"
year_imaged = "2026"
experimenter = [{ first_name = "A", last_name = "B" }]
dataset_name = { column = "store_dataset_name" }
release_date = { column = "store_name_date" }
license = "CC-BY-4.0"

[dca.biosample]
group_id = { column = "store_group_id" }
organism = [{ ontology_term_id = "NCBITaxon:9606", label = "Homo sapiens" }]
tissue_type = "cell line"
cell_line_name = { column = "cell_line" }
tissue = [{ ontology_term_id = "CVCL:0042", label = "U2OS" }]
disease = [{ ontology_term_id = "MONDO:0009807", label = "osteosarcoma" }]
development_stage = [{ ontology_term_id = "unknown", label = "unknown" }]

[dca.specimen]
specimen_state = "live"

[dca.study_component]
imaging_method = [{ ontology_term_id = "FBbi:00000246", label = "fluorescence microscopy" }]
assay = [{ ontology_term_id = "EFO:0007550", label = "high content screen" }]

[dca.channels."raw GFP"]
name = "raw GFP"
description = "GFP fluorescence"
channel_type = "fluorescence"
biological_annotation = { marker = { column = "marker" }, marker_type = "protein" }

[dca.provenance]
```

`confirmed.toml` carries the human-confirmed values. `load_dca_config` merges it over the base.

```toml
[dca.study]
release_date = "2026-11-01"

[dca.provenance.study.release_date]
source_kind = "human"
```

```python
from dca_helpers.metadata import (
    DcaValidationError, load_dca_config, merge_dca_attributes,
    resolve_dca, validate_dca, write_normalization_statistics,
)

block = load_dca_config("base.toml", "confirmed.toml")
row = {"store_dataset_name": "example_plate", "store_group_id": "example_plate/A/1",
       "store_name_date": "", "cell_line": "U2OS", "marker": "SEC61B"}
keys = ["raw GFP"]

warnings = []
payload = validate_dca(resolve_dca(block, row, channel_keys=keys, where="A/1/0"),
                       where="A/1/0", warnings=warnings)
attrs = {}
merge_dca_attributes(attrs, {k: v for k, v in payload.items() if k != "channels"},
                     payload["channels"], run_attrs={"my-pipeline": {"version": "1.0"}})
stats = {"p1": 0.1, "p5": 0.2, "p95": 0.8, "p99": 0.9}
write_normalization_statistics(attrs, {"0": {"dataset_statistics": stats}})

try:
    validate_dca(resolve_dca(load_dca_config("base.toml"), row | {"cell_line": ""},
                             channel_keys=keys, where="A/1/0"), where="A/1/0")
except DcaValidationError as err:
    print(err.fields)
```

A field the row leaves empty and no TOML fills is reported by path in `err.fields` (`['study.release_date', 'biosample']` for the call above, since `base.toml` alone has no release date and the row has no cell line). Pass a list as `warnings=` to collect SHOULD findings (here, one per field that lacks a `[dca.provenance]` record). Without it, `validate_dca` logs one warning per FOV listing every finding, so a plate repeats each missing provenance record once per FOV. A real base config gives each field a provenance record, which silences these warnings. `run_attrs={"<tool>": {"version": ..., "code_git_sha": ...}}` records the writing tool as a sibling of `dca`. `merge_dca_attributes` replaces `attributes.dca` on each write and keeps only `normalization_statistics`, so a re-run drops a section the config no longer has.

Validate every FOV before writing any. A bad FOV found after half the store is written leaves a store that fails the gate. `validate_dca` checks the experimental block only, not channels or normalization statistics. Gate the written store with `dca_helpers.validation.runner.validate(<store>, spec_version="0.2")` or `dca-validate --spec-version 0.2`.

### dca-propose suggests ontology terms for unresolved fields

`dca-propose` suggests values for a human to confirm. It writes nothing. Each `--field` is matched by a rule, and `--hint KEY=VALUE` supplies the input by DCA field path.

| Rule | Fills | Reads hint |
| --- | --- | --- |
| `cellosaurus` | `biosample.cell_line_id`, `biosample.tissue` | `biosample.cell_line_name`; `biosample.tissue` also needs `biosample.tissue_type` to be `cell line` |
| `ols_label` | any `*.label` | the sibling `*.ontology_term_id` |

```bash
dca-propose --field biosample.cell_line_id --hint biosample.cell_line_name=U2OS
```

`--offline` skips registry lookups and reports each field as skipped. The Cellosaurus lookup is importable from `dca_helpers.metadata` as `search_cellosaurus(name)`, which returns `(CURIE, label)` pairs and caches results per name.

## Building the Documentation

The specifications are written in reStructuredText and built using Sphinx. To build and view the documentation locally:

### Prerequisites

- Python 3.11 or later
- pip

### Installation

```bash
# Install documentation dependencies
pip install ".[docs]"
```

### Local Development Server

To start a local development server with automatic rebuilding:

```bash
# Start live development server
sphinx-autobuild docs docs/_build/html
```

This will start a server at `http://localhost:8000` that automatically rebuilds and refreshes when you make changes to the source files.

### Build Static Documentation

To build the documentation once:

```bash
# Build HTML documentation
cd docs && make html
```

The built documentation will be available in `docs/_build/html/index.html`.

### Other Build Formats

```bash
# Build PDF (requires LaTeX)
cd docs && make latexpdf

# Build EPUB
cd docs && make epub

# Clean build files
cd docs && make clean
```

## Testing

The repository includes a comprehensive test suite to validate documentation quality and build process.

### Running Tests

```bash
# Run all tests (recommended)
python test.py

# Run only unit tests (fast)
python test.py --type unit

# Run only integration tests
python test.py --type integration

# Run with verbose output
python test.py --verbose

# Or use pytest directly
pytest tests/ -v
```

### Test Types

- **Unit Tests**: Fast tests that validate configuration, content structure, and formatting
- **Integration Tests**: Tests that build documentation and validate output
- **Slow Tests**: Extended tests including link checking and multiple format builds

### Continuous Integration

Tests run automatically on every push and pull request via GitHub Actions. The CI pipeline:

- Validates documentation builds without errors
- Checks content structure and formatting
- Performs link checking
- Tests multiple output formats

## Accessing the Documentation

For local development and testing, see the "Local Development Server" section above.

Public hosting via ReadTheDocs is configured (`.readthedocs.yaml`) but not yet active.


## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.
Please report unacceptable behavior to [opensource@biohub.org](mailto:opensource@biohub.org).

## Security 

Please disclose security issues responsibly by contacting security@biohub.org.
