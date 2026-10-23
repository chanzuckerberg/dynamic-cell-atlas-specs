"""Tests for ``[dca.*]`` config resolution.

:func:`resolve` on its own (the grammar: ``column`` / ``value`` / ``map``, recursion,
one-item-list collapse), then :func:`resolve_dca` and :func:`validate_dca`.
"""

import copy

import pytest

from dca_helpers.metadata.resolve import (
    DcaValidationError,
    _every_line_is_should,
    resolve,
    resolve_dca,
    validate_dca,
)
from tests.conftest import ASPIRIN_MORGAN_FINGERPRINT, ASPIRIN_PUBCHEM_CID
from tests.metadata.conftest import min_dca_block

METADATA = {
    "pert_type": "DMSO_control",
    "mmoles_per_liter": 0.5,
    "conc_text": "0.5 uM",
    "well": "A1",
}


def test_value_is_used_directly():
    assert resolve({"value": "LINCS Cell Painting"}, METADATA) == "LINCS Cell Painting"


def test_column_reads_the_group_s_metadata():
    assert resolve({"column": "well"}, METADATA) == "A1"


@pytest.mark.parametrize(
    "base",
    [
        pytest.param({"column": "pert_type"}, id="column"),
        pytest.param({"value": "DMSO_control"}, id="value"),
    ],
)
def test_map_translates_a_column_value(base):
    """`map` post-processes the base value however it was resolved — a column read
    and a literal alike; it is not a parallel case to column/value."""
    assert resolve({**base, "map": {"DMSO_control": "negcon"}}, METADATA) == "negcon"


def test_a_value_missing_from_map_is_an_error():
    node = {"column": "well", "map": {"DMSO_control": "negcon"}}
    with pytest.raises(ValueError, match="not in its map"):
        resolve(node, METADATA)


@pytest.mark.parametrize("cell", [2, 2.0], ids=["int", "integral-float"])
def test_map_canonicalises_a_numeric_cell_to_its_string_key(cell):
    """Regression: a numeric column with a `map` aborted the whole manifest.

    A base read from a `column` keeps its dtype, so `binning = 2` (int, or an integral
    float) never matched the string key `"2"` the config authored — the miss raised "not
    in its map" (`config/jump_cp.toml`'s `binning`). `_map_lookup` canonicalises to `str`.
    """
    node = {"column": "binning", "map": {"1": "none", "2": "binned_2x2"}}
    assert resolve(node, {"binning": cell}) == "binned_2x2"


def test_column_and_value_together_is_an_error():
    with pytest.raises(ValueError, match="both 'column' and 'value'"):
        resolve({"column": "well", "value": "x"}, METADATA)


def test_an_absent_column_is_an_error():
    with pytest.raises(ValueError, match="not in the group's metadata"):
        resolve({"column": "nope"}, METADATA)


def test_resolve_dca_reports_an_absent_column_as_a_plain_value_error():
    """Row-shaped errors carry no FOV; callers catch ``ValueError`` and add ``where``."""
    block = min_dca_block()
    block["biosample"]["cell_line_name"] = {"column": "cell_line"}
    with pytest.raises(ValueError, match="'cell_line'") as excinfo:
        resolve_dca(block, {"well": "A1"}, where="A/1/0")
    assert type(excinfo.value) is ValueError
    assert "A/1/0" not in str(excinfo.value)


def test_plain_dicts_and_literals_recurse_untouched():
    node = {"study": {"license": "CC0-1.0", "dataset_name": {"column": "well"}}}
    assert resolve(node, METADATA) == {
        "study": {"license": "CC0-1.0", "dataset_name": "A1"}
    }


def test_a_nested_spec_inside_value_resolves():
    assert resolve({"value": {"column": "well"}}, METADATA) == "A1"


# ---------------------------------------------------------------------------
# The one-item-list collapse — `[[dca.a.b]]` parses to a list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        pytest.param([{"value": "LINCS"}], "LINCS", id="one-item-leaf-collapses"),
        pytest.param(
            [{"value": "a"}, {"value": "b"}], ["a", "b"], id="multi-item-stays"
        ),
        pytest.param([{"value": ["Other"]}], ["Other"], id="one-element-list-value"),
        pytest.param(
            [{"value": [{"ontology_term_id": "na", "label": "na"}]}],
            [{"ontology_term_id": "na", "label": "na"}],
            id="one-element-ontology-list",
        ),
        pytest.param(
            [{"label": "DAPI"}], [{"label": "DAPI"}], id="one-entry-plain-table-list"
        ),
    ],
)
def test_a_one_item_leaf_list_collapses(node, expected):
    """Only the `[[dca.a.b]]` wrapper collapses: a one-item list whose item is a
    leaf spec. A genuine one-element list VALUE (`study_type = ["Other"]`, an
    ontology-term list) stays a list — collapsing it would hand the spec a string
    where it requires a list — and so does a one-entry list of plain tables, which
    is a genuine list, not a wrapped leaf."""
    assert resolve(node, METADATA) == expected


def test_a_schema_object_with_a_value_field_is_not_a_leaf():
    # Concentration is {value, unit, text}: `value` is a real field name here, so
    # membership alone would misread the whole object as a leaf and collapse it to
    # the concentration number.
    node = {
        "unit": "uM",
        "text": {"column": "conc_text"},
        "value": [{"column": "mmoles_per_liter"}],
    }
    assert resolve(node, METADATA) == {
        "unit": "uM",
        "text": "0.5 uM",
        "value": 0.5,
    }
    # An all-literal concentration passes through untouched the same way.
    assert resolve({"value": 10.0, "unit": "uM"}, METADATA) == {
        "value": 10.0,
        "unit": "uM",
    }


# ---------------------------------------------------------------------------
# Blank cells decline, and take their subtree with them
# ---------------------------------------------------------------------------

# The LINCS shape: a definition whose id and mechanism come from columns, beside a
# literal `modality` that the spec makes conditional on a `chemical` block.
DEFINITION = {
    "perturbation_id": {"column": "broad_sample"},
    "mechanism": {"column": "moa"},
    "modality": "compound",
}


def test_a_node_whose_every_column_is_blank_is_pruned_with_its_literals():
    """All-blank prunes the node whole, literals included; a populated column keeps it.

    The literal is the point: leaving `modality = "compound"` behind would demand a
    `chemical` block the control well does not have.
    """
    assert resolve({"definition": DEFINITION}, {"broad_sample": "", "moa": ""}) == {}
    # A partly populated node keeps the node (literal included) and drops only the
    # blank leaf.
    assert resolve(DEFINITION, {"broad_sample": "DMSO", "moa": ""}) == {
        "perturbation_id": "DMSO",
        "modality": "compound",
    }
    # And pruning stops at the smallest enclosing node.
    tree = {"control_class": {"column": "pert_type"}, "definition": DEFINITION}
    out = resolve(tree, {"pert_type": "negcon", "broad_sample": "", "moa": ""})
    assert out == {"control_class": "negcon"}, "control_class is known and must survive"


def test_the_top_level_is_never_pruned_even_when_every_column_is_blank():
    """Pruning applies inside the tree, never to the tree itself.

    A group that declines every source column — a control well, a group the platemap
    omits — must still get its literal `study` / `biosample`. Aggregating the all-blank
    rule across independent top-level blocks would leave the store with no DCA metadata
    at all, which is a worse outcome than one missing a perturbation.
    """
    block = {
        "study": {"license": "CC0-1.0"},
        "perturbation": {
            "control_class": {"column": "pert_type"},
            "definition": {"perturbation_id": {"column": "broad_sample"}},
        },
    }
    out = resolve(block, {"pert_type": "", "broad_sample": ""})
    assert out == {"study": {"license": "CC0-1.0"}}


@pytest.mark.parametrize(
    "blank", ["", "  ", "nan", "NaN", "none", "NULL", "n/a", "na", None]
)
def test_the_blank_vocabulary(blank):
    assert resolve(
        {"a": {"column": "c"}, "b": {"column": "d"}}, {"c": blank, "d": "kept"}
    ) == {"b": "kept"}


def test_a_literal_na_is_always_the_sentinel():
    # Missing-ness is tested only on `column` cells, so a config that writes `na`
    # means the DCA does-not-apply sentinel. No opt-out keyword needed.
    assert resolve({"ontology_term_id": "na", "label": "na"}, {}) == {
        "ontology_term_id": "na",
        "label": "na",
    }


# ---------------------------------------------------------------------------
# Digit-keyed sections become lists
# ---------------------------------------------------------------------------


def test_digit_keys_become_a_list_in_numeric_order():
    # TOML has no integer keys, so `[dca.a.0]` parses to the string "0". The digit map
    # is a VALUE one level below the block's root — which is where the rule fires, and
    # the only place it should.
    node = {"a": {str(i): {"name": f"c{i}"} for i in range(11)}}
    assert resolve(node, {}) == {
        "a": [{"name": f"c{i}"} for i in range(11)]
    }, "not lexical order"


def test_a_digit_keyed_block_root_is_not_turned_into_a_list():
    """Regression: `resolve()` applied the digit rule to the root as well.

    A block whose top-level keys happened to be digits became a list, and `resolve_dca`
    then raised AttributeError reaching for `.get("channels")`. The root of a `[dca]`
    block is a container of named blocks; it is never an indexed list.
    """
    block = {"0": {"a": 1}, "1": {"a": 2}}
    assert resolve(block, {}) == block
    assert resolve_dca(block, {}) == block, "resolve_dca must not raise"


def test_resolve_dca_selects_channels_by_key_in_the_declared_order():
    """The catalogue is a map; `channel_keys` is what orders it and derives `index`."""
    block = {
        "channels": {
            "ch_dna": {"name": "DAPI"},
            "ch_ab": {"name": {"column": "antibody"}},
        }
    }
    assert resolve_dca(
        block, {"antibody": "anti-COPA"}, channel_keys=["ch_dna", "ch_ab"]
    )["channels"] == [
        {"name": "DAPI", "index": 0},
        {"name": "anti-COPA", "index": 1},
    ]


def test_resolve_dca_is_permutation_invariant_in_the_catalogue():
    """Reordering the catalogue's sections cannot change which metadata lands where.

    The whole reason for keying it. Under the removed positional catalogue one integer
    was both the binding and the C order, so a renumber silently did both at once.
    """
    forwards = {"a": {"name": "A"}, "b": {"name": "B"}}
    backwards = {"b": {"name": "B"}, "a": {"name": "A"}}
    keys = ["a", "b"]
    assert (
        resolve_dca({"channels": forwards}, {}, channel_keys=keys)
        == resolve_dca({"channels": backwards}, {}, channel_keys=keys)
        == {"channels": [{"name": "A", "index": 0}, {"name": "B", "index": 1}]}
    )


def test_resolve_dca_rejects_a_positional_catalogue():
    block = {"channels": {"0": {"name": "DAPI"}, "1": {"name": "WGA"}}}
    with pytest.raises(ValueError, match="keyed by position"):
        resolve_dca(block, {}, channel_keys=["0", "1"])


def test_resolve_dca_requires_a_channel_order():
    """No `channel_keys` means nothing says which catalogue entry is which plane."""
    with pytest.raises(ValueError, match="no channel order reached the resolver"):
        resolve_dca({"channels": {"a": {"name": "A"}}}, {})


def test_resolve_dca_reports_a_key_with_no_catalogue_block():
    block = {"channels": {"a": {"name": "A"}}}
    with pytest.raises(
        ValueError, match=r"channel 'b' has no \[dca.channels.b\] block"
    ):
        resolve_dca(block, {}, channel_keys=["a", "b"])


@pytest.mark.parametrize(
    ("channel", "metadata"),
    [
        pytest.param(
            {"name": {"column": "label"}, "channel_type": {"column": "kind"}},
            {"label": "", "kind": "fluorescence"},
            id="blank-name-column",
        ),
        pytest.param({"name": ""}, {}, id="empty-literal-name"),
    ],
)
def test_resolve_dca_requires_every_channel_to_keep_a_name(channel, metadata):
    """A blank name column must not yield a nameless entry: the omero projection
    reads entry["name"] and would KeyError deep inside a convert task, or the plane
    would land unlabeled. A literal `name = ""` ships an unlabeled plane the same
    way — the spec's `name: str` has no min_length and the store-level dca-validate
    passes it too — so content is checked here, not just presence. Fail at manifest
    build, naming the channel."""
    with pytest.raises(ValueError, match="channel 'a' resolved without a usable name"):
        resolve_dca({"channels": {"a": channel}}, metadata, channel_keys=["a"])


# ---------------------------------------------------------------------------
# Provenance: nested tables flatten to the spec's dotted-field-path map
# ---------------------------------------------------------------------------

_RECORD = {"source_kind": "human"}


def test_nested_provenance_flattens_to_dotted_keys():
    """`[dca.provenance.biosample.tissue]` and the quoted one-key form mean the same."""
    nested = {"provenance": {"biosample": {"tissue": dict(_RECORD)}}}
    quoted = {"provenance": {"biosample.tissue": dict(_RECORD)}}
    assert (
        resolve_dca(nested, {})
        == resolve_dca(quoted, {})
        == {"provenance": {"biosample.tissue": _RECORD}}
    )


def test_nested_provenance_keeps_a_record_s_own_subtables():
    """Recursion stops at `source_kind`, so `citation` is a field, not a path segment."""
    record = {"source_kind": "agent", "citation": {"source_url": "https://x"}}
    out = resolve_dca({"provenance": {"study": {"license": record}}}, {})
    assert out["provenance"] == {"study.license": record}


def test_nested_provenance_accepts_a_quoted_segment():
    """An `additional_metadata` item name with spaces is quoted as ONE segment."""
    out = resolve_dca(
        {
            "provenance": {
                "biosample": {
                    "additional_metadata": {"Derived from site": dict(_RECORD)}
                }
            }
        },
        {},
    )
    assert out["provenance"] == {
        "biosample.additional_metadata.Derived from site": _RECORD
    }


def test_a_provenance_entry_without_source_kind_is_an_error():
    """A record missing its required `source_kind` reads as a mistyped path — say so."""
    with pytest.raises(ValueError, match=r"study\.license.*source_kind"):
        resolve_dca(
            {"provenance": {"study": {"license": {"citation": {"quote": "q"}}}}}, {}
        )


def test_a_provenance_path_extending_a_record_must_be_quoted():
    """`a.b` and `a.b.c` both carrying provenance cannot nest; the error says to quote."""
    with pytest.raises(
        ValueError, match=r'\[dca\.provenance\."study\.license\.extra"\]'
    ):
        resolve_dca(
            {"provenance": {"study": {"license": {**_RECORD, "extra": dict(_RECORD)}}}},
            {},
        )


def test_an_empty_provenance_table_stays_empty():
    """`[dca.provenance]` with no entries is how a config says "nothing agent-curated"."""
    assert resolve_dca({"provenance": {}}, {}) == {"provenance": {}}


# The idr0022 empty-well shape: two column-gated records that blank together beside
# literal records for fields that never prune. The gates are load-bearing — a record
# whose field pruned is a MUST error — but before the per-record exemption their
# blanks aggregated up and pruned the WHOLE map, literal records included, failing
# validation with `provenance: Field required` (the anchor-gate workaround's target).
GATED_PROVENANCE = {
    "provenance": {
        "study": {"license": {"source_kind": "human"}},
        "perturbation": {
            "control_class": {"source_kind": "original_data"},
            "definition": {
                "source_kind": {
                    "column": "pert_modality",
                    "map": {"sirna": "original_data"},
                }
            },
            "concentration": {
                "source_kind": {"column": "conc_nm", "map": {"50": "agent"}}
            },
        },
    }
}


def test_a_declined_gate_prunes_its_record_not_the_provenance_map():
    """A blanked gate prunes THAT record; sibling records and the map survive.

    `provenance` is a map of independent records, so the top-level exemption applies
    one level deeper: two gated records blanking together (an empty well's
    `definition` + `concentration`) must not take the literal `study` / `biosample` /
    `control_class` records — or the required map itself — down with them.
    """
    out = resolve_dca(GATED_PROVENANCE, {"pert_modality": "", "conc_nm": ""})
    assert out["provenance"] == {
        "study.license": {"source_kind": "human"},
        "perturbation.control_class": {"source_kind": "original_data"},
    }


def test_a_populated_gate_keeps_its_record_resolved():
    out = resolve_dca(GATED_PROVENANCE, {"pert_modality": "sirna", "conc_nm": "50"})
    assert out["provenance"] == {
        "study.license": {"source_kind": "human"},
        "perturbation.control_class": {"source_kind": "original_data"},
        "perturbation.definition": {"source_kind": "original_data"},
        "perturbation.concentration": {"source_kind": "agent"},
    }


def test_a_declined_record_prunes_whole_with_its_citation_literals():
    """Record-level blank-decline is unchanged: the gate takes the record's own
    literal subtables (its citation) with it, exactly as any node prunes whole."""
    block = {
        "provenance": {
            "study": {"license": {"source_kind": "human"}},
            "perturbation": {
                "concentration": {
                    "source_kind": {"column": "conc_nm", "map": {"50": "agent"}},
                    "citation": {"source_url": "https://x", "quote": "50nM"},
                }
            },
        }
    }
    out = resolve_dca(block, {"conc_nm": ""})
    assert out["provenance"] == {"study.license": {"source_kind": "human"}}


def test_a_segment_whose_every_record_declined_is_dropped_not_an_error():
    """An all-pruned nesting segment vanishes rather than reaching the flatten as an
    empty non-record table, and a map whose every record declined stays an empty map
    rather than pruning into `provenance: Field required`."""
    gated = {
        "source_kind": {"column": "conc_nm", "map": {"50": "agent"}},
    }
    block = {"provenance": {"perturbation": {"concentration": dict(gated)}}}
    assert resolve_dca(block, {"conc_nm": ""}) == {"provenance": {}}
    kept = {
        "provenance": {
            "study": {"license": {"source_kind": "human"}},
            "perturbation": {"concentration": dict(gated)},
        }
    }
    assert resolve_dca(kept, {"conc_nm": ""})["provenance"] == {
        "study.license": {"source_kind": "human"}
    }


def test_a_declined_group_still_validates_with_its_literal_provenance():
    """The manifest-build symptom the exemption fixes: on a group whose only gated
    records declined, the map pruned whole and validation aborted the build with
    `provenance: Field required`. The literal records must ride through instead."""
    block = min_dca_block(
        perturbation={"control_class": "empty"},
        provenance={
            "study": {"license": {"source_kind": "human"}},
            "perturbation": {
                "definition": {
                    "source_kind": {
                        "column": "pert_modality",
                        "map": {"sirna": "original_data"},
                    }
                }
            },
        },
    )
    resolved = resolve_dca(block, {"pert_modality": ""})
    validated = validate_dca(resolved, where="empty well")
    assert validated["provenance"].keys() == {"study.license"}
    assert validated["provenance"]["study.license"]["field_path"] == "study.license"


def test_the_same_path_authored_quoted_and_nested_is_an_error():
    """TOML parses the two spellings as different keys, so both records arrive;
    letting one silently win would lose whichever the author meant to keep."""
    both = {
        "provenance": {
            "biosample.tissue": {"source_kind": "human"},
            "biosample": {"tissue": {"source_kind": "agent"}},
        }
    }
    with pytest.raises(ValueError, match="authored twice"):
        resolve_dca(both, {})


# ---------------------------------------------------------------------------
# resolve_dca and validate_dca
# ---------------------------------------------------------------------------

# A valid, minimal DCA v0.2 block in [dca.*] shape (the shared factory), plus the
# column-fed perturbation the validate tests read through METADATA_FOR_BLOCK.
DCA_BLOCK = min_dca_block(
    perturbation={
        "control_class": [
            {
                "column": "Metadata_pert_type",
                "map": {"DMSO_control": "negcon", "compound": "treatment"},
            }
        ],
        "definition": {
            "perturbation_id": {"column": "Metadata_broad_sample"},
            "modality": "other",
        },
    }
)


@pytest.mark.parametrize(
    ("field", "term", "marker"),
    [
        # `tissue` is the authoritative slot for cell-line identity, so an anatomical
        # term here is the SHOULD.
        pytest.param(
            "tissue",
            {"ontology_term_id": "UBERON:0000955", "label": "brain"},
            "CVCL",
            id="tissue-without-cvcl",
        ),
        # `development_stage` in the wrong ontology for the organism is the second
        # SHOULD.
        pytest.param(
            "development_stage",
            {"ontology_term_id": "MmusDv:0000001", "label": "mouse stage"},
            "MmusDv",
            id="development-stage-wrong-ontology",
        ),
    ],
)
def test_a_should_severity_finding_warns_and_the_build_proceeds(
    caplog, field, term, marker
):
    """A SHOULD must not abort the build, and must not cost us normalisation.

    Two of v0.2.4's rules are SHOULDs that `dca_helpers` implements as a *raising* model
    validator. `validate_dca` used to construct the model to normalise the block and
    treat any raise as fatal, so those two aborted the whole manifest build — on JUMP-CP,
    on a human-reviewed `tissue` term, with a message reading `[SHOULD]`. Both rules live
    in the same raising validator, so covering only one would let a narrower fix pass —
    hence the parametrization over both.

    This test replaces one that asserted the opposite (`a_model_only_failure_still_names_
    the_group`). Its premise was false: it claimed the model is "stricter than the issue
    list", but `validate_experimental_metadata_v0_2` runs the *same* `model_validate` and
    maps `[SHOULD]` to WARNING, so the issue list saw the finding all along — the fixture
    "cleared" it only because the test filtered to severity == "error".
    """
    from dca_helpers.validation.versions.v0_2.experimental_metadata import (
        validate_experimental_metadata_v0_2,
    )

    # Keep a column-fed leaf in the block so normalisation has something to do and its
    # survival is observable.
    block = {
        **DCA_BLOCK,
        "biosample": {**DCA_BLOCK["biosample"], field: [term]},
        # One provenance entry, so the `field_path` injection has something to inject
        # into and the survival of normalisation is observable below.
        "provenance": {"study.license": {"source_kind": "human"}},
    }
    resolved = resolve(block, METADATA_FOR_BLOCK)

    severities = {
        getattr(i.severity, "value", i.severity)
        for i in validate_experimental_metadata_v0_2(
            {k: v for k, v in resolved.items() if k != "channels"}
        )
        if marker in i.message
    }
    assert severities == {
        "warning"
    }, f"fixture must trigger the {field} SHOULD, got {severities}"

    with caplog.at_level("WARNING"):
        out = validate_dca(resolved, where="group A9_s1")

    assert out["biosample"][field][0]["ontology_term_id"] == term["ontology_term_id"]
    assert marker in caplog.text and "group A9_s1" in caplog.text, caplog.text
    # Normalisation still ran: `field_path` is injected from the provenance map key, and
    # would be absent if we had fallen back to an unvalidated passthrough.
    assert out["provenance"]["study.license"]["field_path"] == "study.license"


def _block_with_a_should_finding() -> dict:
    """A resolved block whose ``tissue`` is not a CVCL term, a SHOULD finding."""
    tissue = {"ontology_term_id": "UBERON:0000955", "label": "brain"}
    block = {**DCA_BLOCK, "biosample": {**DCA_BLOCK["biosample"], "tissue": [tissue]}}
    return resolve(block, METADATA_FOR_BLOCK)


def test_a_warnings_list_collects_should_findings_and_logs_nothing(caplog):
    collected: list[str] = []
    with caplog.at_level("WARNING"):
        out = validate_dca(
            _block_with_a_should_finding(), where="group A9_s1", warnings=collected
        )

    assert out["biosample"]["tissue"][0]["ontology_term_id"] == "UBERON:0000955"
    cvcl = [w for w in collected if "CVCL" in w]
    assert len(cvcl) == 1 and cvcl[0].startswith("biosample: "), collected
    assert caplog.records == []


@pytest.mark.parametrize("slot", ["definition", "cotreatment"])
def test_a_missing_beta_perturbation_id_is_a_warning_not_an_error(slot):
    chemical = {"pubchem_cid": ASPIRIN_PUBCHEM_CID}
    perturbation = {
        "control_class": "treatment",
        "definition": {"perturbation_id": "P1", "modality": "compound"},
        slot: {"perturbation_id": "P2", "modality": "compound", "chemical": chemical},
    }
    if slot == "cotreatment":
        perturbation["definition"]["chemical"] = {
            **chemical,
            "morgan_fingerprint": ASPIRIN_MORGAN_FINGERPRINT,
        }
    collected: list[str] = []

    out = validate_dca(
        min_dca_block(perturbation=perturbation), where="A/1/0", warnings=collected
    )

    assert out["perturbation"][slot]["chemical"] == chemical
    assert [w for w in collected if "morgan_fingerprint" in w] == [
        f"perturbation: {slot}.chemical.morgan_fingerprint is missing for modality "
        "'compound' (beta in v0.2; required from v0.3)"
    ]


def test_a_cotreatment_warning_does_not_hide_a_missing_definition():
    perturbation = {
        "control_class": "treatment",
        "cotreatment": {
            "perturbation_id": "P2",
            "modality": "crispr",
            "genetic": {"gene_symbol": "TP53"},
        },
    }
    collected: list[str] = []
    with pytest.raises(DcaValidationError) as exc:
        validate_dca(
            min_dca_block(perturbation=perturbation), where="A/1/0", warnings=collected
        )
    assert "definition is required" in str(exc.value)
    assert exc.value.fields == ["perturbation"]


def test_without_a_warnings_list_should_findings_are_logged(caplog):
    with caplog.at_level("WARNING"):
        validate_dca(_block_with_a_should_finding(), where="group A9_s1")

    assert "CVCL" in caplog.text and "group A9_s1" in caplog.text, caplog.text


def test_a_must_raised_alongside_a_should_is_still_fatal():
    """The tolerance is per-message, not per-validator.

    `_every_line_is_should` splits a newline-joined accumulation the same way
    `_convert_errors` does, so a validator that reports a MUST and a SHOULD together
    cannot be waved through on the strength of its second line.
    """
    assert _every_line_is_should("[SHOULD] a\n[SHOULD] b")
    assert not _every_line_is_should("[SHOULD] a\nhard failure")
    assert not _every_line_is_should("hard failure")
    assert not _every_line_is_should("")


#: The columns DCA_BLOCK's `{ column = ... }` leaves read.
METADATA_FOR_BLOCK = {
    "Metadata_pert_type": "DMSO_control",
    "Metadata_broad_sample": "DMSO",
}


@pytest.mark.parametrize(
    ("study_type", "extra_columns"),
    [
        pytest.param([{"column": "st"}], {"st": "Screen"}, id="column-fed"),
        pytest.param("Screen", {}, id="plainly-authored-scalar"),
        pytest.param(["Screen"], {}, id="real-list-left-alone"),
    ],
)
def test_a_list_field_fed_from_a_column_stays_a_list(study_type, extra_columns):
    """Regression: the one-item-list collapse flattened genuine list fields.

    `[[dca.study.study_type]]` and `[[dca.study.dataset_name]]` are the same TOML shape,
    but one is `list[...]` and the other a string. A type-blind resolve collapses both,
    so a column-fed `study_type` would reach the spec as a bare scalar and fail with
    "Input should be a valid list" — making that config un-writable. The production
    `resolve_dca` walks the schema, so its declared list type wraps the scalar up front
    (the obvious lenient thing — a plainly authored scalar at a list field is what
    authors mean) while a real list, a real scalar and a dict field are left alone. This
    exercises the real production composition (`validate_dca(resolve_dca(...))`); the
    up-front shaping is what makes the old `_relist` repair step unnecessary.
    """
    block = {**DCA_BLOCK, "study": {**DCA_BLOCK["study"], "study_type": study_type}}
    out = validate_dca(
        resolve_dca(block, {**METADATA_FOR_BLOCK, **extra_columns}), where="group A1_s1"
    )
    assert out["study"]["study_type"] == ["Screen"]
    assert out["study"]["dataset_name"] == "D", "a scalar must not be wrapped"
    assert isinstance(out["provenance"], dict), "dict fields are not lists"


# ---------------------------------------------------------------------------
# Schema-driven structural dispatch: resolve_dca walks the model, so the field's
# declared type — not the TOML shape — decides leaf-vs-list-vs-model. Three edge
# cases the type-blind grammar had to guess at stop being possible by construction.
# ---------------------------------------------------------------------------


def test_concentration_authored_only_as_value_column_is_a_model_not_a_leaf():
    """A `Concentration` authored with just `value = {column = ...}` stays an object.

    `{value: {column: ...}}` has keys `⊆ {column, value, map}`, so `_is_leaf` reads it as a
    leaf and the type-blind resolver collapses it to a bare number. Walking the schema,
    `concentration` is a model, so `value` is its `value` slot — object kept and normalised.
    """
    block = min_dca_block(
        perturbation={
            "control_class": "treatment",
            "definition": {"perturbation_id": "X", "modality": "other"},
            "concentration": {"value": {"column": "conc"}, "unit": "uM"},
        }
    )
    resolved = resolve_dca(block, {"conc": "0.5"})
    assert resolved["perturbation"]["concentration"] == {
        "value": "0.5",
        "unit": "uM",
    }, "schema-driven resolve must keep it a Concentration object, not a bare scalar"
    out = validate_dca(resolved, where="g")
    assert out["perturbation"]["concentration"]["value"] == 0.5
    assert isinstance(out["perturbation"]["concentration"]["value"], float)


def test_a_model_field_misauthored_as_a_bare_leaf_falls_back_to_the_leaf_grammar():
    """A model field written as a bare leaf (no key the model declares) resolves as a leaf.

    `concentration = {column = "conc"}` is invalid either way, but the schema-driven walk
    must not turn it into `{"column": <cell>}` (iterating `"column"` as a model key). Sharing
    no key with `Concentration.model_fields`, it defers to the leaf grammar → recognisable
    scalar. Narrow guard: it must NOT catch `{value = {column}}`, so the fix above stands.
    """
    node = {"column": "conc"}
    block = min_dca_block(
        perturbation={
            "control_class": "treatment",
            "definition": {"perturbation_id": "X", "modality": "other"},
            "concentration": node,
        }
    )
    resolved = resolve_dca(block, {"conc": 0.5})
    assert (
        resolved["perturbation"]["concentration"] == 0.5
    ), "a bare-leaf model field resolves to the cell, not {'column': <cell>}"


def test_a_list_field_fed_a_scalar_is_a_list_straight_out_of_resolve_dca():
    """The list-vs-scalar case that once needed a repair pass is decided up front instead.

    `study_type` (`list[str]`) and `dataset_name` (`str`) are authored the same TOML way;
    a type-blind resolve collapses both and would leave a separate pass to re-wrap
    `study_type`. Against the schema, its declared list type wraps the scalar up front, so
    `resolve_dca` alone returns a list — no repair round-trip — while the scalar is left
    alone.
    """
    block = min_dca_block()
    block["study"] = {**block["study"], "study_type": "Screen"}
    resolved = resolve_dca(block, {})
    assert resolved["study"]["study_type"] == [
        "Screen"
    ], "a scalar at a list field is a list straight out of resolve_dca, no repair pass"
    assert resolved["study"]["dataset_name"] == "D", "the scalar field is left alone"
    # And a genuine one-element list value is preserved, not double-wrapped.
    block["study"]["study_type"] = ["Screen"]
    assert resolve_dca(block, {})["study"]["study_type"] == ["Screen"]


def test_a_list_field_authored_as_a_digit_keyed_subtable_resolves_to_a_flat_list():
    """A `list[T]` field authored `[dca.study.study_type.0]` flattens to `["Screen"]`.

    Regression: the single-pass list branch wraps a non-list node into `[node]` before
    recursing, so a digit-keyed `{"0": "Screen"}` sub-table became `[{"0": "Screen"}]` ->
    `_indexed_to_list` inside -> the doubly-nested `[["Screen"]]` the spec rejects. The old
    two-phase path ran `_indexed_to_list` first; the list branch now does too. No live
    config uses this form (all author `[[...]]`/inline arrays), so this was latent.
    """
    block = min_dca_block()
    block["study"] = {**block["study"], "study_type": {"0": "Screen"}}
    assert resolve_dca(block, {})["study"]["study_type"] == [
        "Screen"
    ], "a digit-keyed sub-table at a list field flattens to a list, not a nested list"
    # Numeric ordering preserved, still no double-nesting, with more than one entry.
    block["study"]["study_type"] = {"0": "Screen", "1": "Other"}
    assert resolve_dca(block, {})["study"]["study_type"] == ["Screen", "Other"]


def test_resolve_dca_output_is_model_shaped_so_validate_needs_no_repair():
    """End-state: `resolve_dca` alone yields a block `validate_dca` accepts unrepaired.

    The single schema walk shapes the ENTIRE experimental block — including `provenance`,
    routed through its `dict[str, FieldProvenance]` type — so what reaches `validate_dca`
    is already model-shaped. A scalar-authored list field is a list straight out of
    `resolve_dca`, and nested provenance flattens and gets its `field_path` injected,
    with no intervening re-listing pass.
    """
    block = min_dca_block()
    # A list field authored as a bare scalar — the case the old `_relist` repaired.
    block["study"] = {**block["study"], "study_type": "Screen"}
    # Provenance authored nested; it must flatten and normalise straight through.
    block["provenance"] = {"study": {"license": {"source_kind": "human"}}}

    resolved = resolve_dca(block, {})
    assert resolved["study"]["study_type"] == [
        "Screen"
    ], "list field is wrapped by the schema walk, pre-validate"
    assert resolved["provenance"] == {
        "study.license": {"source_kind": "human"}
    }, "provenance is resolved through the schema walk and still flattened by resolve_dca"

    out = validate_dca(resolved, where="g")
    assert out["study"]["study_type"] == ["Screen"]
    assert (
        out["provenance"]["study.license"]["field_path"] == "study.license"
    ), "normalisation ran: field_path injected from the map key"


# ---------------------------------------------------------------------------
# Whole-block cases through resolve + validate TOGETHER
# ---------------------------------------------------------------------------
#
# Each case runs a full block through `_resolved` (resolve_dca then validate_dca), so
# what they pin is the interaction the grammar tests above cannot: pruning that leaves
# a block the spec then rejects. (The collapse meeting the validator's re-listing is
# pinned by the collapse test, the list-field repair test and
# `test_build_manifest_attaches_a_resolved_block_per_group`.) Grammar rules on their
# own are covered once, above.

_MIN = min_dca_block()

_PERT = {
    "control_class": {"column": "cls", "map": {"neg": "negcon", "cmpd": "treatment"}},
    "definition": {
        "perturbation_id": {"column": "pid"},
        "modality": "compound",
        "chemical": {
            "inchikey": "KYRVNWMVYQXFEU-UHFFFAOYSA-N",
            "pubchem_cid": ASPIRIN_PUBCHEM_CID,
            "morgan_fingerprint": ASPIRIN_MORGAN_FINGERPRINT,
        },
    },
    "solvent": {"column": "solv"},
}


def _resolved(block, metadata):
    # In a real run the order comes from the config's `channel_column_names`; these cases
    # are about the resolver, so the catalogue's own key order stands in for it.
    keys = list(block.get("channels") or []) or None
    return validate_dca(resolve_dca(block, metadata, channel_keys=keys), where="g")


#: id -> (block, metadata, check(result))
ADVERSARIAL_OK = {
    "populated": (
        {**_MIN, "perturbation": _PERT},
        {"cls": "cmpd", "pid": "X", "solv": "DMSO"},
        lambda r: (
            r["perturbation"]["solvent"] == "DMSO"
            and r["perturbation"]["control_class"] == "treatment"
        ),
    ),
}

#: id -> (block, metadata, substring the error must contain)
ADVERSARIAL_ERRORS = {
    # Pruning `perturbation_id` leaves `definition` unpopulated, and the spec requires
    # it whenever control_class is not "empty" — so a partial block fails loudly.
    "blank-prunes-subtree": (
        {**_MIN, "perturbation": _PERT},
        {"cls": "cmpd", "pid": "", "solv": "DMSO"},
        "definition is required",
    ),
    # NaN is blank, so this is the same failure by a different route.
    "nan-is-blank": (
        {**_MIN, "perturbation": _PERT},
        {"cls": "cmpd", "pid": float("nan"), "solv": "DMSO"},
        "definition is required",
    ),
}


@pytest.mark.parametrize("case_id", sorted(ADVERSARIAL_OK))
def test_adversarial_case_resolves(case_id):
    block, metadata, check = ADVERSARIAL_OK[case_id]
    assert check(_resolved(block, metadata)), case_id


@pytest.mark.parametrize("case_id", sorted(ADVERSARIAL_ERRORS))
def test_adversarial_case_raises(case_id):
    block, metadata, expected = ADVERSARIAL_ERRORS[case_id]
    with pytest.raises(ValueError) as excinfo:
        _resolved(block, metadata)
    assert expected in str(excinfo.value), f"{case_id}: {excinfo.value}"


def test_neither_resolve_nor_validate_mutates_its_inputs():
    """Both alias in places; a caller that reused a group's metadata would be bitten."""
    block, metadata, _ = ADVERSARIAL_OK["populated"]
    b, m = copy.deepcopy(block), copy.deepcopy(metadata)
    _resolved(b, m)
    assert m == metadata, "the group's metadata was mutated"
    assert b == block, "the config block was mutated"


def test_validation_error_lists_every_must_finding_with_its_field():
    block = min_dca_block()
    del block["study"]["release_date"]
    del block["biosample"]["organism"]
    with pytest.raises(DcaValidationError) as exc:
        validate_dca(resolve_dca(block, {}), where="A/1/0")
    err = exc.value
    assert err.where == "A/1/0"
    assert sorted(err.fields) == ["biosample.organism", "study.release_date"]
    assert all(issue.startswith(field) for issue, field in zip(err.issues, err.fields))
    assert isinstance(err, ValueError)
