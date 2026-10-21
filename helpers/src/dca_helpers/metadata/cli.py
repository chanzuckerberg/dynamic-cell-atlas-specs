"""
CLI entry point for DCA field proposals.

Usage
-----
    dca-propose --dataset-name NAME [--channel NAME ...] --field PATH [--field PATH ...]
                [--hint KEY=VALUE ...] [--offline] [--output {text,json}]
"""

import json
from dataclasses import asdict
from typing import Literal

import click

from dca_helpers.metadata.propose import Proposal, propose


def _parse_hint(
    ctx: click.Context, param: click.Parameter, values: tuple[str, ...]
) -> dict:
    hints = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise click.BadParameter(f"{item!r} must be KEY=VALUE", param=param)
        hints[key] = value
    return hints


def _print_text(proposals: list[Proposal]) -> None:
    for p in proposals:
        value = "-" if p.value is None else json.dumps(p.value)
        click.echo(f"{p.field}\t{value}\t{p.source}")


@click.command()
@click.option(
    "--dataset-name", required=True, help="Dataset name, used for date rules."
)
@click.option(
    "--channel", "channels", multiple=True, help="Channel name, in index order."
)
@click.option(
    "--field", "fields", multiple=True, required=True, help="Unresolved field path."
)
@click.option(
    "--hint",
    "hints",
    multiple=True,
    callback=_parse_hint,
    help="Lookup hint as KEY=VALUE, e.g. biosample.cell_line_name=A549.",
)
@click.option("--offline", is_flag=True, help="Skip registry lookups.")
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def main(
    dataset_name: str,
    channels: tuple[str, ...],
    fields: tuple[str, ...],
    hints: dict[str, str],
    offline: bool,
    output: Literal["text", "json"],
) -> None:
    """Propose values for unresolved DCA fields. Proposals are never written."""
    proposals = propose(
        fields,
        dataset_name=dataset_name,
        channel_names=channels,
        hints=hints,
        online=not offline,
    )
    if output == "json":
        click.echo(json.dumps([asdict(p) for p in proposals], indent=2))
    else:
        _print_text(proposals)
