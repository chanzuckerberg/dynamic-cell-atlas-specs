"""
CLI entry point for DCA proposals.

Lookups: Cellosaurus, NCBITaxon and OLS labels for taxonomy fields, Ensembl gene
ids and PubChem CIDs for perturbations. The Morgan fingerprint of a compound is
computed from a SMILES hint with RDKit and works with ``--offline``.

Usage
-----
    dca-propose --field PATH [--field PATH ...] [--hint KEY=VALUE ...]
                [--offline] [--output {text,json}]

Example
-------
    dca-propose --field perturbation.definition.chemical.morgan_fingerprint \
        --hint perturbation.definition.chemical.smiles='CC(=O)Oc1ccccc1C(=O)O'
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
        if p.value is None:
            value = "-"
        elif isinstance(p.value, str):
            value = p.value
        else:
            value = json.dumps(p.value)
        click.echo(f"{p.field}\t{value}\t{p.source}")


@click.command()
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
    fields: tuple[str, ...],
    hints: dict[str, str],
    offline: bool,
    output: Literal["text", "json"],
) -> None:
    """Propose values for unresolved DCA fields. Proposals are never written.

    Each rule reads its inputs from hints named by DCA field path, e.g.
    perturbation.definition.genetic.gene_symbol=TP53 (or .ncbi_gene_id, plus an
    optional species=mus_musculus), perturbation.definition.chemical.smiles,
    .inchikey, or perturbation.definition.name.
    """
    proposals = propose(
        fields,
        hints=hints,
        online=not offline,
    )
    if output == "json":
        click.echo(json.dumps([asdict(p) for p in proposals], indent=2))
    else:
        _print_text(proposals)
