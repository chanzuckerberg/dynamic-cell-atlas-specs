import json

from click.testing import CliRunner

from dca_helpers.metadata.cli import main


def test_cli_json_output_offline():
    r = CliRunner().invoke(
        main,
        [
            "--dataset-name", "2026_09_25_x",
            "--field", "study.release_date",
            "--field", "biosample.cell_line_id",
            "--hint", "biosample.cell_line_name=A549",
            "--offline",
            "--output", "json",
        ],
    )  # fmt: skip
    assert r.exit_code == 0
    assert json.loads(r.output) == [
        {
            "field": "study.release_date",
            "value": "2026-09-25",
            "source": "dataset name",
            "rule": "name_date",
        },
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
        [
            "--dataset-name",
            "x",
            "--channel",
            "Phase3D",
            "--field",
            "channels.0.channel_type",
        ],
    )
    assert r.exit_code == 0
    assert r.output == 'channels.0.channel_type\t"labelfree"\tchannel name\n'


def test_cli_rejects_a_malformed_hint():
    r = CliRunner().invoke(
        main,
        ["--dataset-name", "x", "--field", "study.release_date", "--hint", "nokey"],
    )
    assert r.exit_code == 2 and "KEY=VALUE" in r.output
