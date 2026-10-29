import json

from click.testing import CliRunner

from dca_helpers.metadata.cli import main


def test_cli_json_output_offline():
    r = CliRunner().invoke(
        main,
        [
            "--field", "biosample.cell_line_id",
            "--hint", "biosample.cell_line_name=A549",
            "--offline",
            "--output", "json",
        ],
    )  # fmt: skip
    assert r.exit_code == 0
    assert json.loads(r.output) == [
        {
            "field": "biosample.cell_line_id",
            "value": None,
            "source": "lookup skipped (offline)",
            "rule": "cellosaurus",
        },
    ]


def test_cli_text_output_is_tab_separated():
    r = CliRunner().invoke(
        main,
        ["--field", "biosample.cell_line_id", "--offline"],
    )
    assert r.exit_code == 0
    assert r.output == "biosample.cell_line_id\t-\tno hint: biosample.cell_line_name\n"


def test_cli_rejects_a_malformed_hint():
    r = CliRunner().invoke(
        main,
        ["--field", "biosample.cell_line_id", "--hint", "nokey"],
    )
    assert r.exit_code == 2 and "KEY=VALUE" in r.output


def test_cli_replays_a_recorded_lookup(replay_registry):
    r = CliRunner().invoke(
        main,
        [
            "--field", "biosample.cell_line_id",
            "--hint", "biosample.cell_line_name=A549",
            "--output", "json",
        ],
    )  # fmt: skip
    assert r.exit_code == 0
    assert json.loads(r.output) == [
        {
            "field": "biosample.cell_line_id",
            "value": "CVCL:0023",
            "source": "Cellosaurus CVCL:0023",
            "rule": "cellosaurus",
        },
    ]
