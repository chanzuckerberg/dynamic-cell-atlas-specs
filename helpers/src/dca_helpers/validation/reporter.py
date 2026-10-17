"""
Builds a Parquet validation report from a `ValidationRun`.

`build_report` is the single public entry point: given a `ValidationRun`, it
returns the report as Parquet bytes. This module owns the report's schema and
serialization only; it never decides where the bytes end up. `write_local` is
a thin convenience on top of `build_report` for the CLI's `--output parquet`
mode (atomic local file write). Callers that store the report elsewhere
should call `build_report` directly.

One row per node, not one row per run: a batch `validate()` call over many
Zarr stores produces one Parquet file with one row per `ZarrNodeValidationResult`,
plus a single `record_type="validation_run_summary"` row. Every field on
that row is otherwise identical to what's on every node row (`input_path`,
`dca_spec_version`, `start_timestamp`, `completion_timestamp`, etc.) -- its
job is purely structural, guaranteeing at least one row exists even for a
zero-store run, not carrying exclusive data. `ValidationSummary`'s other
aggregate fields (stores_passed/failed, nodes_by_type, etc.) are dropped
from the persisted report entirely -- each is a plain aggregate query over
the node rows, so storing it a second time here would just be a second
source of truth that could silently disagree with the first. `passed`/
`node_type`/`parent_path` are plain top-level columns, not nested inside an
array, so a consumer filters directly (e.g. `WHERE parent_path IS NULL AND
passed = false` for "which store roots failed"; field and label rows carry a
parent_path, so a store whose only defect is in a field or label has a passing
root row) instead of unnesting, and
Parquet row-group pruning has real per-row stats to act on once a report
spans many stores -- unlike the single-row-per-run design this replaced.

There is no `run_id` column: rows from one `validate()` call always land in
one file (one `build_report()` call = one file), so a consumer can already
group by source file without a redundant stored key. If
report files are ever compacted across runs, this assumption needs
revisiting.
"""

import dataclasses
import os
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dca_helpers.validation.result import (
    Issue,
    Severity,
    ValidationRun,
    ZarrNodeValidationResult,
)

_REPORTING_FORMAT_VERSION = "1.0.0"

_RECORD_TYPE_VALIDATION_RUN_SUMMARY = "validation_run_summary"
_RECORD_TYPE_NODE = "node"

_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_SLUG_LENGTH = 50
_SLUG_HEAD_LENGTH = 20
_SLUG_TRUNCATION_MARKER = "--"


def _slugify_path(path: str) -> str:
    """Turn a local path or s3:// URI into a short, human-identifying
    filename prefix.

    Collision-safety is handled entirely by `_generate_filename`'s random
    suffix, not by this function -- so this is intentionally just sanitize
    and truncate, no digest, no anti-collision cleverness.

    Truncation keeps a bucket-identifying head plus as much of the tail as
    fits, not just the head: a field/FOV path's distinguishing part (the
    plate name) sits near the head, with non-distinguishing well/field
    coordinates trailing off the end, but a store-root path's
    distinguishing part (the dataset/plate name) *is* the tail, behind a
    long, generic bucket/prefix head. Keeping both ends handles both shapes
    without needing to detect which one a given path is. The head is
    capped at `_SLUG_HEAD_LENGTH` and taken from the first path segment
    (the bucket name) rather than a raw character count, so it lands on a
    segment boundary instead of cutting a bucket name off mid-word.

    A leading "-" survives sanitization (it's a legal filename character)
    but is stripped anyway: a filename starting with "-" can be misread as
    a flag by CLI tools operating on it later (e.g. `rm -foo.parquet`)
    unless the caller quotes it or prefixes "./".
    """
    slug = path.rstrip("/").split("://", 1)[-1]
    slug = _SLUG_UNSAFE.sub("_", slug).strip("_").lstrip("-")
    if len(slug) <= _MAX_SLUG_LENGTH:
        return slug or "run"
    head = slug.split("_", 1)[0][:_SLUG_HEAD_LENGTH]
    tail_length = _MAX_SLUG_LENGTH - len(head) - len(_SLUG_TRUNCATION_MARKER)
    return head + _SLUG_TRUNCATION_MARKER + slug[-tail_length:]


def _generate_filename(run: ValidationRun, timestamp: datetime) -> str:
    """Filename is `<slug>__<timestamp>__<random>.parquet`.

    `timestamp` is threaded in from the caller rather than computed here, so
    `write_local` can derive both the filename and the record's own
    `completion_timestamp` from the same instant -- computing them
    separately risked the two drifting apart (e.g. a UTC-second or
    -midnight tick landing between the two calls).

    The random suffix carries all of this scheme's collision-safety, since the
    slug alone doesn't: several runs can validate the same path (for example,
    parallel runs that each take a subset of one large store with
    --wells/--field-offset/--field-limit), so they share a slug and may finish
    within the same second. 8 hex characters make a collision negligible; a
    collision would silently overwrite another report, because the write is an
    atomic `os.replace`.
    """
    slug = _slugify_path(run.path)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{slug}__{stamp}__{suffix}.parquet"


def _completion_timestamp(timestamp: datetime | None = None) -> str:
    return (timestamp or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


# Fields dropped from ZarrNodeValidationResult's persisted per-node columns --
# not schema-agnostic in intent, just the one known case where a dataclass
# field must NOT carry through automatically via _jsonify below.
_NODE_RESULT_EXCLUDED_FIELDS = frozenset(
    {
        "spec_version",  # redundant per node -- see the run-level dca_spec_version
    }
)


def _jsonify(value):
    """
    Recursively convert dataclasses/enums into plain Python values suitable
    for pyarrow ingestion against an explicit schema (see REPORT_SCHEMA).

    Used instead of hand-enumerating each field a dataclass carries, so a
    new ZarrNodeValidationResult field is included automatically rather
    than silently dropped from the persisted report until this module is
    updated to know about it by name. That alone isn't sufficient for
    Parquet, though -- a genuinely new field also needs a matching column
    added to REPORT_SCHEMA, or pyarrow will reject it (see the comment above
    REPORT_SCHEMA for why the schema is explicit).
    tests/validation/test_reporter.py's schema/dataclass consistency tests
    assert the two stay in sync.

    Enum conversion matches this module's pre-existing, established
    conventions: Severity by `.value` (a lowercase word), any other enum
    (e.g. ZarrNodeType) by `.name`.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _jsonify(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, Severity):
        return value.value
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {_jsonify(k): _jsonify(v) for k, v in value.items()}
    return value


def _issue_to_dict(issue: Issue) -> dict:
    # loc is a mixed int/str tuple (mirrors Pydantic's error-location tuple)
    # -- REPORT_SCHEMA types it list<string>, so every element is stringified
    # rather than left for pyarrow to reject as a mixed-type list.
    return {
        "severity": issue.severity.value,
        "loc": [str(x) for x in issue.loc],
        "message": issue.message,
    }


def _result_to_dict(result: ZarrNodeValidationResult) -> dict:
    """Serialize `result`'s fields as-is, except `_NODE_RESULT_EXCLUDED_FIELDS`."""
    record = {
        f.name: _jsonify(getattr(result, f.name))
        for f in dataclasses.fields(result)
        if f.name not in _NODE_RESULT_EXCLUDED_FIELDS
    }
    record["issues"] = [_issue_to_dict(i) for i in result.issues]
    return record


def _shared_fields(run: ValidationRun, completion_timestamp: str) -> dict:
    """Provenance columns present on every row, regardless of `record_type`."""
    return {
        "input_path": run.path,
        "dca_spec_version": run.spec_version,
        "dca_validator_version": run.validator_version,
        "reporting_format_version": _REPORTING_FORMAT_VERSION,
        "start_timestamp": run.summary.start_timestamp,
        "completion_timestamp": completion_timestamp,
    }


def _validation_run_summary_row(run: ValidationRun, completion_timestamp: str) -> dict:
    """
    The one row per report guaranteeing at least one row exists even for a
    zero-store run. Every field on it is otherwise identical to what's on
    every node row -- it carries no data a node row couldn't also carry.
    Without it, a zero-store run would produce a valid but genuinely empty
    Parquet file, and none of that run's provenance would be recoverable
    from its contents at all.
    """
    return {
        "record_type": _RECORD_TYPE_VALIDATION_RUN_SUMMARY,
        **_shared_fields(run, completion_timestamp),
        "node_path": None,
        "parent_path": None,
        "passed": None,
        "ngff_version": None,
        "multiscale_level_count": None,
        "issues": None,
        "node_type": None,
        "skipped": None,
        "subset_coverage": None,
    }


def _node_row(
    result: ZarrNodeValidationResult, run: ValidationRun, completion_timestamp: str
) -> dict:
    return {
        "record_type": _RECORD_TYPE_NODE,
        **_shared_fields(run, completion_timestamp),
        **_result_to_dict(result),
    }


def to_rows(run: ValidationRun, completion_timestamp: str) -> list[dict]:
    """Serialize a `ValidationRun` into the rows persisted to Parquet: one
    validation_run_summary row (see `_validation_run_summary_row`) followed
    by one row per node result.

    This always includes every issue (errors and warnings) -- the persisted
    report is the archival record, not a display format, so it isn't gated
    by display-only flags the CLI's text/json output has (e.g.
    --show-warnings).

    `completion_timestamp` is threaded in rather than computed here so a
    caller deriving another value from it (a file name, as `write_local`
    does) gets the exact same timestamp that ends up in every row --
    computing it twice risks a UTC-midnight tick putting the two out of sync.
    """
    return [_validation_run_summary_row(run, completion_timestamp)] + [
        _node_row(r, run, completion_timestamp) for r in run.results
    ]


# --- Explicit Parquet schema -------------------------------------------
#
# Explicit rather than inferred from data: pyarrow's schema inference from a
# handful of rows can guess a wrong/incompatible type for a column that
# happens to be all-None in one particular run (e.g. a zero-store run's
# node columns, which are only ever populated by the validation_run_summary
# row's nulls), and tools that read many report files together need each
# column to have the same type in every file -- a column typed differently
# (or as `null`) in one file than another breaks a merged-schema read. The
# cost is that a genuinely new dataclass field
# needs this schema updated by hand too; see `_jsonify`'s docstring and the
# consistency tests below.

_ISSUE_TYPE = pa.struct(
    [
        pa.field("severity", pa.string()),
        pa.field("loc", pa.list_(pa.string())),
        pa.field("message", pa.string()),
    ]
)

_SUBSET_COVERAGE_TYPE = pa.struct(
    [
        pa.field("wells_total", pa.int64()),
        pa.field("wells_selected", pa.int64()),
        pa.field("fields_total", pa.int64()),
        pa.field("fields_selected", pa.int64()),
    ]
)

REPORT_SCHEMA = pa.schema(
    [
        # Discriminator: _RECORD_TYPE_VALIDATION_RUN_SUMMARY or _RECORD_TYPE_NODE.
        pa.field("record_type", pa.string()),
        # Shared provenance -- present on every row, identical across all
        # rows from one run. start_timestamp/completion_timestamp together
        # let a query derive duration without a separately-stored value.
        pa.field("input_path", pa.string()),
        pa.field("dca_spec_version", pa.string()),
        pa.field("dca_validator_version", pa.string()),
        pa.field("reporting_format_version", pa.string()),
        pa.field("start_timestamp", pa.string()),
        pa.field("completion_timestamp", pa.string()),
        # Node-only -- null on the validation_run_summary row. Mirrors
        # ZarrNodeValidationResult's fields (minus _NODE_RESULT_EXCLUDED_FIELDS).
        pa.field("node_path", pa.string()),
        pa.field("parent_path", pa.string()),
        pa.field("passed", pa.bool_()),
        pa.field("ngff_version", pa.string()),
        pa.field("multiscale_level_count", pa.int64()),
        pa.field("issues", pa.list_(_ISSUE_TYPE)),
        pa.field("node_type", pa.string()),
        pa.field("skipped", pa.bool_()),
        pa.field("subset_coverage", _SUBSET_COVERAGE_TYPE),
    ]
)


def _to_table(run: ValidationRun, completion_timestamp: str) -> pa.Table:
    rows = to_rows(run, completion_timestamp)
    return pa.Table.from_pylist(rows, schema=REPORT_SCHEMA)


def build_report(
    run: ValidationRun, *, completion_timestamp: str | None = None
) -> bytes:
    """
    Build the Parquet report for `run` and return it as bytes.

    The single entry point for anyone -- this module's own `write_local`,
    or an external caller -- that wants `run`'s report without writing or
    knowing anything about how it eventually gets persisted. Serializes
    entirely in memory (via a pyarrow BufferOutputStream, not a temp file):
    there is no partial-write/atomicity concern here the way there is for
    `write_local`'s on-disk file, since nothing is visible
    to another process until the caller does something with the returned
    bytes.

    `completion_timestamp` defaults to `None`, in which case one is computed
    internally. Pass it explicitly when the value baked into the returned
    report must match one computed elsewhere (for example, a file name). Compute
    it once and reuse the same string, rather than computing it twice and
    risking the two drifting apart (e.g. a UTC-midnight tick landing between
    the computations). See `to_rows`'s docstring for the same rationale.
    """
    if completion_timestamp is None:
        completion_timestamp = _completion_timestamp()
    table = _to_table(run, completion_timestamp)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def write_local(run: ValidationRun, output_dir: str | Path | None = None) -> Path:
    """
    Build `run`'s report and atomically write it under `output_dir` (the
    current working directory if not given).

    Writes to a `.tmp` file first and `os.replace`s it into place so a
    process killed mid-write (for example by a time limit or an out-of-memory
    kill) never leaves a half-written file visible under its final name: a
    file with its final name is always complete.
    `os.replace` (rather than `os.rename`) also overwrites an existing
    destination atomically on every platform; `os.rename` only does that
    on POSIX -- it raises on Windows if the destination already exists.

    This is a convenience for `dca-validate --output parquet`. Callers that
    persist the report themselves should call `build_report` directly; the
    collision-safe naming and atomic write here apply only to this local file.

    The filename's timestamp segment and the record's `completion_timestamp`
    are derived from the same `datetime.now()` call, not computed
    independently -- see `_generate_filename`'s docstring.
    """
    directory = Path(output_dir) if output_dir is not None else Path.cwd()
    directory.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    final_path = directory / _generate_filename(run, now)
    tmp_path = final_path.with_name(final_path.name + ".tmp")

    try:
        tmp_path.write_bytes(
            build_report(run, completion_timestamp=_completion_timestamp(now))
        )
        os.replace(tmp_path, final_path)
    except Exception:
        # Broad on purpose: pyarrow raises its own exception types (e.g.
        # ArrowInvalid, ArrowTypeError) that are not OSError subclasses, and
        # any failure here must still clean up the `.tmp` file to preserve
        # the "file exists => complete" invariant.
        tmp_path.unlink(missing_ok=True)
        raise
    return final_path
