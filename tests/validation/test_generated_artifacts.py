"""Drift guard: the committed generated artifacts must match `make schema`.

The LinkML schema is the single source of truth; the JSON Schema and Pydantic
models are generated and committed. If someone edits the YAML without
regenerating (or hand-edits a generated file), CI fails here.

Skipped when linkml is not installed — it is a dev-only tool, not a runtime
dependency, so the artifacts are validated wherever the dev toolchain exists.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# Relative paths, run from REPO_ROOT — must match the Makefile exactly.
# gen-pydantic embeds the schema path it is given into the module's `source_file`
# metadata, so the invocation has to mirror `make schema` to be reproducible.
VERSIONS = ["0.2", "0.3"]


def _schema(version: str) -> str:
    return f"schema/v{version}/dca_experimental_metadata.yaml"


def _json_out(version: str) -> Path:
    return (
        REPO_ROOT
        / "schema"
        / f"v{version}"
        / "generated"
        / "dca_experimental_metadata.schema.json"
    )


def _pydantic_out(version: str) -> Path:
    return (
        REPO_ROOT
        / "helpers"
        / "src"
        / "dca_helpers"
        / "validation"
        / "versions"
        / f"v{version.replace('.', '_')}"
        / "_generated_models.py"
    )


pytestmark = pytest.mark.skipif(
    shutil.which("gen-json-schema") is None or shutil.which("gen-pydantic") is None,
    reason="linkml generators not installed (dev-only tool)",
)


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    return proc.stdout


@pytest.mark.parametrize("version", VERSIONS)
def test_json_schema_in_sync_with_yaml(version):
    regenerated = _run(["gen-json-schema", "--no-metadata", _schema(version)])
    committed = _json_out(version).read_text()
    assert regenerated == committed, (
        f"schema/v{version}/generated/dca_experimental_metadata.schema.json is stale. "
        "Run `make schema` and commit the result."
    )


@pytest.mark.parametrize("version", VERSIONS)
def test_pydantic_models_in_sync_with_yaml(version):
    regenerated = _run(["gen-pydantic", _schema(version)])
    committed = _pydantic_out(version).read_text()
    assert regenerated == committed, (
        f"{_pydantic_out(version).relative_to(REPO_ROOT)} is stale. "
        "Run `make schema` and commit the result."
    )
