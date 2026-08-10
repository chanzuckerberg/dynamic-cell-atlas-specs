"""
CLI entry point for DCA validation.

Usage
-----
    dca-validate PATH [--spec-version VERSION] [--output {text,json}] [--strict]
                      [--show-warnings]
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Literal

import click

from dca_helpers import CURRENT_SPEC_VERSION
from dca_helpers.validation import validate
from dca_helpers.validation.result import Severity, ValidationRun

# Suppress ResourceWarnings from unclosed aiohttp sessions/connectors emitted
# on GC by zarr/s3fs internals — not actionable from user code.
# Two paths: warnings.warn (filtered here) and loop.call_exception_handler
# (silenced via the asyncio logger).
warnings.filterwarnings("ignore", category=ResourceWarning, message="Unclosed.*")
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


def _print_text(run: ValidationRun, show_warnings: bool = False) -> None:
    for r in run:
        if r.skipped:
            continue
        issues = r.issues if show_warnings else r.errors
        if not issues:
            continue
        status = "PASS" if r.passed else "FAIL"
        click.echo(f"{status} {r.node_path}")
        for issue in issues:
            prefix = "WARNING" if issue.severity == Severity.WARNING else "ERROR"
            click.echo(f"  [{prefix}] {issue.message}")

    summary = run.summary
    warnings_count = sum(len(r.warnings) for r in run if not r.skipped)
    failed_lines = ""
    for r in run:
        if r.skipped:
            continue
        if not r.passed:
            failed_lines += f"\n    {r.node_path}"
            for err in r.errors:
                failed_lines += f"\n      - {err.message}"
    skipped_line = (
        f"\n  {summary.stores_skipped} skipped (GEFF)" if summary.stores_skipped else ""
    )
    click.echo(
        f"\nValidated {summary.stores_validated} stores "
        f"({summary.level_zarr_jsons_validated} scale-level zarr.json files):\n"
        f"  {summary.stores_passed} passed\n"
        f"  {summary.stores_failed} failed{failed_lines}{skipped_line}\n"
        f"  {warnings_count} warning(s)\n"
        f"  {summary.duration_seconds:.2f}s elapsed"
    )


def _print_json(run: ValidationRun, show_warnings: bool = False) -> None:
    output = [
        {
            "node_path": r.node_path,
            "parent_path": r.parent_path,
            "spec_version": r.spec_version,
            "ngff_version": r.ngff_version,
            "passed": r.passed,
            "skipped": r.skipped,
            "issues": [
                {
                    "severity": i.severity.value,
                    "loc": list(i.loc),
                    "message": i.message,
                }
                for i in (r.issues if show_warnings else r.errors)
            ],
        }
        for r in run
    ]
    click.echo(json.dumps(output, indent=2))


@click.command()
@click.argument("path")
@click.option(
    "--spec-version",
    default=CURRENT_SPEC_VERSION,
    show_default=True,
    help="DCA spec version to validate against.",
)
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Exit 1 if any SHOULD warnings are present.",
)
@click.option(
    "--show-warnings",
    is_flag=True,
    default=False,
    help="Show SHOULD warnings in output (hidden by default).",
)
@click.option(
    "--strategy",
    type=click.Choice(["glob", "walk"]),
    default="glob",
    show_default=True,
    help=(
        "S3 discovery strategy. glob: single flat scan (better for many scattered images). "
        "walk: BFS with early pruning (better for single/few plates)."
    ),
)
@click.option(
    "--check-ontology",
    is_flag=True,
    default=False,
    help=(
        "Resolve every ontology_term_id against its source registry (EBI OLS4, "
        "or the Cellosaurus API for CVCL cell lines) and check its label. For "
        "preparation_method and imaging_method, also check the term descends "
        "from the required FBbi ancestor. Makes network requests; unresolvable "
        "ids, label mismatches and wrong-subtree terms are errors, an "
        "unreachable registry is a warning."
    ),
)
@click.pass_context
def main(
    ctx: click.Context,
    path: str,
    spec_version: str,
    output: str,
    strict: bool,
    show_warnings: bool,
    strategy: Literal["glob", "walk"],
    check_ontology: bool,
) -> None:
    """Validate Zarr stores at or under PATH against the DCA spec.

    PATH may be a local filesystem path or an s3://bucket/prefix URL.
    If PATH contains zarr.json at its top level it is validated directly;
    otherwise all Zarr stores found under PATH are validated.
    """
    run = validate(
        path,
        spec_version=spec_version,
        strategy=strategy,
        show_progress=output == "text",
        check_ontology=check_ontology,
    )

    if output == "json":
        _print_json(run, show_warnings=show_warnings)
    else:
        _print_text(run, show_warnings=show_warnings)

    has_errors = run.summary.stores_failed > 0
    has_warnings = any(r.warnings for r in run if not r.skipped)

    if has_errors or (strict and has_warnings):
        ctx.exit(1)
