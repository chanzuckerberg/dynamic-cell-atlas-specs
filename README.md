# Dynamic Cell Atlas Specifications

This repo defines standards for storing diverse 5D fluorescence microscopy datasets (3D+time+channels) in a unified format for AI model training, using Zarr v3, OME-ngff 0.5 and additional metadata.

For more detailed information on scope, use cases etc. refer to the [documentation](https://chanzuckerberg.github.io/dynamic-cell-atlas-specs/).

## Contents

- [Specification](docs/v0.2/) and JSON schema
- [Data Validator](helpers/src/dca_helpers/validation) using Pydantic
- [Helper tooling](helpers/src/dca_helpers) for parts of the specification that go beyond general OME-ngff

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
