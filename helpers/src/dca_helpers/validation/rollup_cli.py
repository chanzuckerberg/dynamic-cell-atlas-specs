"""Metadata manifest creation and offline HCS report rollup."""

import os
import tempfile
from pathlib import Path

import click

from dca_helpers import CURRENT_SPEC_VERSION
from dca_helpers.validation import reporter
from dca_helpers.validation.cli import _print_json, _print_text
from dca_helpers.validation.hcs_facts import HCSManifest
from dca_helpers.validation.rollup import create_hcs_manifest, rollup_reports


@click.group()
def main() -> None:
    """Validate cross-chunk HCS consistency against an explicit snapshot manifest."""


@main.command("manifest")
@click.argument("path")
@click.option(
    "--snapshot-id", required=True, help="Caller-managed immutable dataset revision."
)
@click.option("--spec-version", default=CURRENT_SPEC_VERSION, show_default=True)
@click.option("--output-file", required=True, type=click.Path(path_type=Path))
def manifest_command(
    path: str, snapshot_id: str, spec_version: str, output_file: Path
) -> None:
    """Record all declared fields from PATH's plate/well metadata, without opening images."""
    try:
        manifest = create_hcs_manifest(path, snapshot_id, spec_version)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    # A partial manifest file must never be mistaken for a complete inventory.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_file.parent, delete=False
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(manifest.model_dump_json(indent=2) + "\n")
        os.replace(tmp_name, output_file)
    except OSError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
    click.echo(f"Wrote HCS snapshot manifest to {output_file}", err=True)


@main.command("check")
@click.argument(
    "manifest_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "reports",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    type=click.Choice(["text", "json", "parquet"]),
    default="text",
    show_default=True,
)
@click.option("--output-directory", type=click.Path(path_type=Path), default=None)
@click.option(
    "--show-warnings", is_flag=True, help="Show SHOULD findings in text/JSON output."
)
@click.option(
    "--strict", is_flag=True, help="Exit 1 for SHOULD warnings as well as errors."
)
@click.pass_context
def check_command(
    ctx: click.Context,
    manifest_file: Path,
    reports: tuple[Path, ...],
    output: str,
    output_directory: Path | None,
    show_warnings: bool,
    strict: bool,
) -> None:
    """Combine REPORTS for MANIFEST_FILE. Reads reports only, never the source plate."""
    if output_directory is not None and output != "parquet":
        raise click.UsageError("--output-directory requires --output parquet")
    try:
        manifest = HCSManifest.model_validate_json(
            manifest_file.read_text(encoding="utf-8")
        )
        run = rollup_reports(reports, manifest)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    if output == "parquet":
        report_path = reporter.write_local(run, output_dir=output_directory)
        click.echo(f"Wrote HCS rollup report to {report_path}", err=True)
    elif output == "json":
        _print_json(run, show_warnings=show_warnings)
    else:
        _print_text(run, show_warnings=show_warnings)
    if run.summary.stores_failed or (strict and any(r.warnings for r in run)):
        ctx.exit(1)
