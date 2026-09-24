"""
CLI entry point for DCA validation.

Usage
-----
    dca-validate PATH [--spec-version VERSION] [--output {text,json,parquet}]
                      [--output-directory PATH] [--strict] [--show-warnings]
                      [--wells W1,W2] [--field-offset N] [--field-limit N]
                      [--list-wells]
"""

import json
import logging
import warnings
from typing import Literal

import click

from dca_helpers import CURRENT_SPEC_VERSION
from dca_helpers.validation import reporter, validate
from dca_helpers.validation.result import Severity, ValidationRun
from dca_helpers.validation.runner import list_plate_wells

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
    if summary.is_partial:
        click.echo(
            f"\n  PARTIAL: {summary.wells_selected}/{summary.wells_total} wells, "
            f"{summary.fields_selected}/{summary.fields_total} fields selected"
        )


def _print_json(run: ValidationRun, show_warnings: bool = False) -> None:
    output = []
    for r in run:
        entry = {
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
        if r.subset_coverage is not None:
            # Only the plate-root entry carries these keys (see #178) —
            # additive, does not change the existing flat-list shape for
            # other consumers. "partial" is JSON's equivalent of the text
            # output's PARTIAL marker line.
            entry["partial"] = True
            entry["subset_coverage"] = {
                "wells_total": r.subset_coverage.wells_total,
                "wells_selected": r.subset_coverage.wells_selected,
                "fields_total": r.subset_coverage.fields_total,
                "fields_selected": r.subset_coverage.fields_selected,
            }
        output.append(entry)
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
    type=click.Choice(["text", "json", "parquet"]),
    default="text",
    show_default=True,
    help=(
        "Output format. text/json print to stdout; parquet writes a "
        "validation report file instead (see --output-directory)."
    ),
)
@click.option(
    "--output-directory",
    default=None,
    metavar="PATH",
    help=(
        "Directory to write the Parquet report to, with --output parquet "
        "(defaults to the current directory). Invalid with any other "
        "--output value."
    ),
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
@click.option(
    "--wells",
    default=None,
    help=(
        "Comma-separated list of well paths to validate, matching "
        "plate.wells[].path exactly (e.g. 'A/1,B/2'). Requires PATH to point "
        "directly at a single HCS plate store, not a discovery root."
    ),
)
@click.option(
    "--field-offset",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Skip this many fields (in declaration order) within each selected well.",
)
@click.option(
    "--field-limit",
    type=click.IntRange(min=0),
    default=None,
    help="Validate at most this many fields (after --field-offset) per selected well.",
)
@click.option(
    "--list-wells",
    is_flag=True,
    default=False,
    help=(
        "List PATH's wells and their declared field counts, then exit without "
        "validating. Requires PATH to point directly at a single HCS plate store."
    ),
)
@click.pass_context
def main(
    ctx: click.Context,
    path: str,
    spec_version: str,
    output: str,
    output_directory: str | None,
    strict: bool,
    show_warnings: bool,
    strategy: Literal["glob", "walk"],
    check_ontology: bool,
    wells: str | None,
    field_offset: int,
    field_limit: int | None,
    list_wells: bool,
) -> None:
    """Validate Zarr stores at or under PATH against the DCA spec.

    PATH may be a local filesystem path or an s3://bucket/prefix URL.
    If PATH contains zarr.json at its top level it is validated directly;
    otherwise all Zarr stores found under PATH are validated.
    """
    if output_directory is not None and output != "parquet":
        raise click.UsageError("--output-directory requires --output parquet")

    if list_wells:
        if output == "parquet":
            raise click.UsageError(
                "--list-wells cannot be combined with --output parquet — it "
                "always prints a listing, not a report."
            )
        if wells is not None or field_offset != 0 or field_limit is not None:
            raise click.UsageError(
                "--list-wells cannot be combined with --wells/--field-offset/"
                "--field-limit — it always lists every well in the plate."
            )
        try:
            wells_and_counts = list_plate_wells(path)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        total_fields = sum(count for _, count in wells_and_counts)
        if output == "json":
            click.echo(
                json.dumps(
                    {
                        "wells": [
                            {"well": w, "fields": c} for w, c in wells_and_counts
                        ],
                        "wells_total": len(wells_and_counts),
                        "fields_total": total_fields,
                    },
                    indent=2,
                )
            )
        else:
            for well_path, count in wells_and_counts:
                click.echo(f"{well_path}  {count} fields")
            click.echo(f"\n{len(wells_and_counts)} wells, {total_fields} fields total")
        ctx.exit(0)

    try:
        run = validate(
            path,
            spec_version=spec_version,
            strategy=strategy,
            show_progress=output == "text",
            check_ontology=check_ontology,
            wells=wells,
            field_offset=field_offset,
            field_limit=field_limit,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    if output == "json":
        _print_json(run, show_warnings=show_warnings)
    elif output == "parquet":
        report_path = reporter.write_local(run, output_dir=output_directory)
        click.echo(f"Wrote validation report to {report_path}", err=True)
    else:
        _print_text(run, show_warnings=show_warnings)

    has_errors = run.summary.stores_failed > 0
    has_warnings = any(r.warnings for r in run if not r.skipped)

    if has_errors or (strict and has_warnings):
        ctx.exit(1)
