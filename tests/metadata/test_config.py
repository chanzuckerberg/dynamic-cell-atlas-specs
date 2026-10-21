from pathlib import Path

import pytest

from dca_helpers.metadata import load_dca_config


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_later_file_wins_at_a_scalar_leaf(tmp_path):
    base = _write(
        tmp_path,
        "base.toml",
        '[dca.study]\nlicense = "CC-BY-4.0"\ndataset_name = { column = "store_dataset_name" }\n',
    )
    over = _write(tmp_path, "over.toml", '[dca.study]\nlicense = "CC0-1.0"\n')
    assert load_dca_config(base, over)["study"] == {
        "license": "CC0-1.0",
        "dataset_name": {"column": "store_dataset_name"},
    }


def test_a_literal_replaces_a_column_leaf_whole(tmp_path):
    base = _write(
        tmp_path,
        "b.toml",
        '[dca.biosample]\ncell_line_name = { column = "cell_line" }\n',
    )
    over = _write(tmp_path, "o.toml", '[dca.biosample]\ncell_line_name = "A549"\n')
    assert load_dca_config(base, over)["biosample"]["cell_line_name"] == "A549"


def test_a_value_leaf_replaces_a_column_leaf_whole(tmp_path):
    base = _write(
        tmp_path,
        "b.toml",
        '[dca.biosample]\ncell_line_name = { column = "cell_line" }\n',
    )
    over = _write(
        tmp_path, "o.toml", '[dca.biosample]\ncell_line_name = { value = "A549" }\n'
    )
    assert load_dca_config(base, over)["biosample"]["cell_line_name"] == {
        "value": "A549"
    }


def test_a_column_leaf_replaces_a_value_leaf_whole(tmp_path):
    base = _write(
        tmp_path, "b.toml", '[dca.biosample]\ncell_line_name = { value = "A549" }\n'
    )
    over = _write(
        tmp_path,
        "o.toml",
        '[dca.biosample]\ncell_line_name = { column = "cell_line" }\n',
    )
    assert load_dca_config(base, over)["biosample"]["cell_line_name"] == {
        "column": "cell_line"
    }


def test_an_array_of_tables_is_replaced_not_appended(tmp_path):
    base = _write(
        tmp_path,
        "b.toml",
        '[[dca.study.experimenter]]\nname = "A"\n[[dca.study.experimenter]]\nname = "B"\n',
    )
    over = _write(tmp_path, "o.toml", '[[dca.study.experimenter]]\nname = "C"\n')
    assert load_dca_config(base, over)["study"]["experimenter"] == [{"name": "C"}]


def test_a_quoted_channel_key_survives(tmp_path):
    path = _write(
        tmp_path,
        "c.toml",
        '[dca.channels."raw GFP EX488 EM525-45"]\ndescription = "d"\n',
    )
    assert "raw GFP EX488 EM525-45" in load_dca_config(path)["channels"]


def test_a_file_without_a_dca_table_is_an_error(tmp_path):
    (tmp_path / "a.toml").write_text("[other]\nx = 1\n")
    with pytest.raises(ValueError, match=r"a\.toml.*\[dca\]"):
        load_dca_config(tmp_path / "a.toml")


def test_inputs_are_not_mutated_across_calls(tmp_path):
    base = _write(tmp_path, "b.toml", '[dca.study]\nlicense = "CC-BY-4.0"\n')
    over = _write(tmp_path, "o.toml", '[dca.study]\nlicense = "CC0-1.0"\n')
    first = load_dca_config(base, over)
    first["study"]["license"] = "changed"
    second = load_dca_config(base, over)
    assert second == {"study": {"license": "CC0-1.0"}}
    assert first is not second
