Validator
=========

The DCA validator checks whether a Zarr dataset conforms to the DCA v0.2 specification.
Given a local path or S3 URL, it validates each store against OME-NGFF v0.5 structure
and then applies DCA-specific MUST and SHOULD requirements.

Multiple stores are validated in parallel.

Install
-------

Clone the repository and install from source:

.. code-block:: bash

   git clone https://github.com/chanzuckerberg/dynamic-cell-atlas-specs
   cd dynamic-cell-atlas-specs
   uv pip install -e .

CLI
---

.. code-block:: text

   dca-validate PATH [OPTIONS]

   Arguments:
     PATH  Local filesystem path or s3://bucket/prefix URL.

   Options:
     --spec-version TEXT            DCA spec version to validate against [default: 0.1]
     --output [text|json]           Output format [default: text]
     --strict                       Exit 1 if any SHOULD warnings are present
     --strategy [glob|walk]         S3 discovery strategy [default: glob]
     --check-ontology               Resolve every ontology_term_id against its
                                    source registry and check its label. Makes
                                    network requests; off by default.
     --wells TEXT                    Comma-separated well paths to validate,
                                    matching plate.wells[].path exactly (e.g.
                                    'A/1,B/2'). Requires PATH to point directly
                                    at a single HCS plate store.
     --field-offset INTEGER RANGE   Skip this many fields (in declaration
                                    order) within each selected well [default: 0]
     --field-limit INTEGER RANGE    Validate at most this many fields (after
                                    --field-offset) per selected well
     --list-wells                   List PATH's wells and their declared field
                                    counts, then exit without validating.
     --help                         Show this message and exit.

Pass ``--spec-version 0.2`` to validate against the v0.2 rules.

**Examples**

.. code-block:: bash

   # Validate a single store against v0.2
   dca-validate path/to/image.zarr --spec-version 0.2

   # Validate all stores under a directory
   dca-validate path/to/experiment/ --spec-version 0.2

   # Validate an S3 plate (use walk strategy for a single plate root)
   dca-validate s3://my-bucket/data/plate.ome.zarr --spec-version 0.2 --strategy walk

   # Validate many scattered images on S3 (use glob strategy)
   dca-validate s3://my-bucket/data/ --spec-version 0.2 --strategy glob

   # Emit JSON output
   dca-validate s3://my-bucket/data/plate.ome.zarr --spec-version 0.2 --output json

   # Fail on SHOULD warnings as well as MUST errors
   dca-validate path/to/image.zarr --spec-version 0.2 --strict

   # Additionally resolve every ontology term against its registry (network)
   dca-validate path/to/image.zarr --spec-version 0.2 --check-ontology

   # List a plate's wells and their field counts without validating
   dca-validate s3://my-bucket/data/plate.ome.zarr --spec-version 0.2 --list-wells

   # Validate only well A/1
   dca-validate s3://my-bucket/data/plate.ome.zarr --spec-version 0.2 --wells A/1

   # Chunk a huge well: validate fields 5000-9999 of well A/1
   dca-validate s3://my-bucket/data/plate.ome.zarr --spec-version 0.2 \
       --wells A/1 --field-offset 5000 --field-limit 5000

**Well/field subsetting**

For a single HCS plate whose per-well field count is too large to validate in
one run (a single well can be far larger than a typical whole plate),
``--wells``, ``--field-offset``, and ``--field-limit`` scope one invocation to
a subset of the plate's work:

* ``--wells`` restricts which wells are opened at all. Wells not named are
  never opened — no checks run against them, and they are not treated as
  failures.
* ``--field-offset``/``--field-limit`` slice each *selected* well's own field
  list independently (per well, not globally across wells), in the order
  fields are declared in that well's metadata. This is what makes chunking
  work *within* one oversized well, not just across wells.
* ``--list-wells`` is a dry run: it prints every well's declared field count
  without validating anything, so you can decide how to split the work before
  committing to a chunking scheme. It always lists every well — combining it
  with ``--wells``/``--field-offset``/``--field-limit`` is a usage error
  rather than silently ignoring the filter. Respects ``--output json``.

These options require ``PATH`` to point directly at a single HCS plate store
— not a discovery root — since subsetting within one already-identified plate
is orthogonal to, and mechanically incompatible with, discovering multiple
stores under a parent prefix in the same invocation.

When any of these narrow what was validated, both output formats carry an
explicit partial-run marker (with wells/fields selected vs. the total in
scope — on-disk fields across every well that got opened, which can be fewer
than a plate's raw declared count if a field is missing or duplicated) so a
partial result is never mistaken for full plate coverage. Text output prints
a ``PARTIAL: ...`` summary line; JSON output adds ``"partial": true`` and a
``"subset"`` object to the plate-root entry. Partial-ness is
informational only: it does not change exit-code semantics — the exit code
still reflects whether errors were found among whatever was actually
validated.

**S3 discovery strategies**

Two strategies are available for discovering Zarr stores under an S3 prefix:

* ``glob`` — performs a single flat scan for all ``zarr.json`` files under the prefix.
  Better for many scattered image stores across a large prefix.
* ``walk`` — BFS traversal with early pruning (stops descending once a Zarr root is found).
  Better for a single plate or a small number of known plate roots.

**Exit codes**

* ``0`` — all stores pass (SHOULD warnings permitted unless ``--strict``)
* ``1`` — one or more stores fail, or ``--strict`` and warnings are present

Python API
----------

.. code-block:: python

   from dca_helpers.validation import validate
   from dca_helpers.validation.result import ValidationRun

   run: ValidationRun = validate("path/to/experiment/", spec_version="0.2")

   summary = run.summary
   print(
       f"{summary.stores_passed}/{summary.stores_validated} stores passed "
       f"({summary.level_zarr_jsons_validated} scale-level zarr.json files) "
       f"in {summary.duration_seconds:.2f}s"
   )

   for r in run:
       status = "PASS" if r.passed else "FAIL"
       print(f"{status} {r.node_path}")
       for issue in r.issues:
           print(f"  [{issue.severity.name}] {issue.message}")

``validate`` accepts any path in the hierarchy — a single store, a subdirectory, or an
ancestor containing multiple stores. If the path has a ``zarr.json`` at its top level it
is validated directly; otherwise all Zarr stores found under the path are validated in
parallel.

The ``strategy`` parameter selects the S3 discovery strategy (``"glob"`` or ``"walk"``);
it is ignored for local paths.

.. code-block:: python

   # Validate a plate on S3 using the walk strategy
   run = validate(
       "s3://my-bucket/data/plate.ome.zarr",
       spec_version="0.2",
       strategy="walk",
   )

Interpreting Results
--------------------

Each entry in the ``ValidationRun`` corresponds to one validated Zarr node:

* ``r.passed`` — ``True`` when there are no ``ERROR``-severity issues.
* ``r.issues`` — list of :class:`Issue` objects, each with a ``severity``
  (``ERROR`` or ``WARNING``) and a ``message``.
* ``r.errors`` — convenience filter: only ``ERROR``-severity issues.
* ``r.warnings`` — convenience filter: only ``WARNING``-severity issues.

**Errors** correspond to MUST violations — the store fails validation.
**Warnings** correspond to SHOULD violations — informational only, the store still passes.

.. code-block:: python

   from dca_helpers.validation.result import Severity

   for r in run:
       print(f"{r.node_path}: {len(r.errors)} error(s), {len(r.warnings)} warning(s)")

What v0.2 Validates
--------------------

The v0.2 validator applies all v0.1 array-level checks and adds new checks for
multiresolution rules and channel/normalization metadata.

Array-level checks (unchanged from v0.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These rules apply to every scale level (Zarr array) in both image and label stores:

.. list-table::
   :header-rows: 1
   :widths: 10 55 35

   * - Level
     - Rule
     - Spec section
   * - MUST
     - Axes must be ``[t, c, z, y, x]``
     - array-standard.rst § Dimensions
   * - MUST
     - Uncompressed chunk size ≥ 512 KB (or equal to array size if smaller)
     - array-standard.rst § Chunk Size
   * - MUST
     - Compression codec must be ``zstd``, ``lz4``, or ``blosc``
     - array-standard.rst § Compression
   * - MUST
     - Shard size < 5 TB
     - array-standard.rst § Shard Size
   * - MUST
     - If array ≥ 1 GB, shard size must be ≥ 1 GB
     - array-standard.rst § Shard Size
   * - MUST
     - Downsampling factor for ``t`` and ``c`` dimensions must be 1
     - array-standard.rst § Multiresolution
   * - SHOULD
     - Uncompressed chunk size ≥ 1 MB
     - array-standard.rst § Chunk Size
   * - SHOULD
     - Spatial chunk sizes should be 128 × 128 × 128 (z × y × x)
     - array-standard.rst § Chunk Size
   * - SHOULD
     - Time and channel chunk sizes should be 1
     - array-standard.rst § Chunk Size
   * - SHOULD
     - Shard size < 5 GB
     - array-standard.rst § Shard Size
   * - SHOULD
     - Spatial shard dimensions should be ≤ 2048
     - array-standard.rst § Shard Size
   * - SHOULD
     - Time shard dimension should be ≥ 16
     - array-standard.rst § Shard Size
   * - SHOULD
     - Integer data should use zstd (not blosc), compression level ≤ 3
     - array-standard.rst § Compression
   * - SHOULD
     - Float data should use byte shuffle + zstd (not blosc), level ≤ 3
     - array-standard.rst § Compression
   * - SHOULD
     - Raw image arrays should use ``uint8`` or ``uint16``
     - array-standard.rst § Data Types
   * - SHOULD
     - Label image arrays should use ``uint32``
     - array-standard.rst § Data Types

Multiresolution rule (changed from v0.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

v0.1 required at least 3 scale levels. v0.2 replaces this with a size-based rule:

.. list-table::
   :header-rows: 1
   :widths: 10 55 35

   * - Level
     - Rule
     - Spec section
   * - MUST
     - At least one scale level must have every spatial dimension ≤ 2048
     - array-standard.rst § Multiresolution
   * - SHOULD
     - Downsampling factor for spatial dimensions should be ≈ 2
     - array-standard.rst § Multiresolution

DCA metadata (new in v0.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

v0.2 requires a ``dca`` object in the ``zarr.json`` of each image group.
Label images are exempt from ``dca`` metadata requirements.

.. list-table::
   :header-rows: 1
   :widths: 10 55 35

   * - Level
     - Rule
     - Spec section
   * - MUST
     - ``dca`` key must be present in the image group ``zarr.json``
     - channel-metadata.rst § Location in Zarr Store
   * - MUST
     - ``dca`` must contain ``version``, ``url``, ``channels``, and ``normalization_statistics``
     - channel-metadata.rst, schema.json ``dca.required``
   * - MUST
     - ``dca.version`` must be ``"0.2"``
     - channel-metadata.rst, schema.json
   * - SHOULD
     - ``dca.channels`` should include one entry per channel in the C axis
     - channel-metadata.rst § Location in Zarr Store

Channel metadata (new in v0.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each entry in ``dca.channels`` is validated as follows:

.. list-table::
   :header-rows: 1
   :widths: 10 55 35

   * - Level
     - Rule
     - Spec section
   * - MUST
     - Each channel must have ``name`` (string)
     - channel-metadata.rst § Required Fields
   * - MUST
     - Each channel must have ``index`` (integer ≥ 0)
     - channel-metadata.rst § Required Fields
   * - MUST
     - Each channel must have ``description`` (string)
     - channel-metadata.rst § Required Fields
   * - MUST
     - Channel ``index`` values must be unique across all channels
     - channel-metadata.rst § Required Fields
   * - SHOULD
     - ``channel_type`` should be one of ``fluorescence``, ``chromogenic``, ``labelfree``, ``predicted``
     - channel-metadata.rst § Guidance on Channel Type
   * - SHOULD
     - ``biological_annotation`` should be provided for ``fluorescence`` and ``predicted`` channels
     - channel-metadata.rst § Optional Fields
   * - SHOULD
     - ``biological_annotation.marker_type`` should be one of ``endogenous_tag``, ``live_cell_dye``, ``fixed_dye``, ``antibody``
     - channel-metadata.rst § Guidance on Marker Type Vocabulary
   * - SHOULD
     - ``biological_annotation.cpg_labeled_structure`` should match the CellPainting Gallery ``Label_Structure`` vocabulary
     - channel-metadata.rst § Biological Annotation Entries
   * - SHOULD
     - ``biological_annotation.cpg_labeled_molecule`` should match the CellPainting Gallery ``Label_Molecule`` vocabulary
     - channel-metadata.rst § Biological Annotation Entries

Normalization statistics (new in v0.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dca.normalization_statistics`` is validated as follows:

.. list-table::
   :header-rows: 1
   :widths: 10 55 35

   * - Level
     - Rule
     - Spec section
   * - MUST
     - Keys must be channel index strings (e.g. ``"0"``, ``"1"``) or ``"metadata"``
     - normalization-statistics.rst, schema.json ``additionalProperties: false``
   * - MUST
     - Each channel entry must have ``dataset_statistics``
     - normalization-statistics.rst § Channel Normalization Fields
   * - MUST
     - ``dataset_statistics`` must contain ``p1``, ``p5``, ``p95``, ``p99``
     - normalization-statistics.rst § Required Fields

Experimental metadata (new in v0.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The v0.2 ``dca`` block also carries the experimental-metadata record, mirrored from the
curated Parquet (the source of truth). Its **required floor** (``study``,
``biosample``, ``specimen``, ``study_component``, ``provenance``) is the validated
part; recommended and optional fields are carried in the same block and are not
enforced. The validator extracts the floor
blocks from the ``dca`` object and validates them against the generated **Pydantic
models** (``helpers/src/dca_helpers/validation/versions/v0_2/_generated_models.py``), which are
produced from the
:download:`LinkML schema <../../schema/v0.2/dca_experimental_metadata.yaml>` — the single
source of truth for the experimental metadata. The generated base enforces the required
floor, types, enums, and CURIE-shape patterns; the conditional if/then rules below
(``gen-pydantic`` records these only as inert metadata) are re-stated as ``model_validator``
subclasses in ``validation/versions/v0_2/experimental_metadata.py``. Most violations are
MUST (ERROR), reported under a ``dca`` location prefix; the one exception is
provenance-coverage (see below). Label images are exempt.

.. note::

   The committed JSON Schema (``…/dca_experimental_metadata.schema.json``) carries the
   identical rules and is exercised by the test suite as a cross-check, but it is **not**
   on the runtime path — the validator enforces the Pydantic models. Editing the JSON
   Schema by hand has no effect on validation; change the LinkML source and run
   ``make schema``.

.. list-table::
   :header-rows: 1
   :widths: 10 55 35

   * - Level
     - Rule
     - Spec section
   * - MUST
     - Structure — object nesting, field types, and numeric bounds
     - experimental-metadata.rst § Conformance
   * - MUST
     - Required floor present — ``study`` / ``biosample`` / ``specimen`` / ``study_component`` / ``provenance`` and their required fields. ``perturbation`` is **not** part of this floor; see the next row
     - experimental-metadata.rst § Conformance
   * - MUST
     - ``perturbation`` **contents**, whenever the block is present — the structural, enum, CURIE, and conditional rules below all apply to it. Its **presence is never required**: an absent ``perturbation`` is always accepted, because no check can distinguish an unperturbed dataset from a screen whose curator omitted the block (see the limitations below)
     - experimental-metadata.rst § Conformance
   * - MUST
     - ``well_perturbation`` **contents**, whenever a well group's ``dca`` carries the block — dispatched to ``validate_well_perturbation_v0_2`` rather than the image-group entry point (which would filter the block out and misreport the image-only required floor as missing). Same presence rule as ``perturbation``: an absent ``well_perturbation`` is always accepted
     - experimental-metadata.rst § Conformance
   * - MUST
     - Cross-placement agreement — every field of view under one well must carry an identical ``perturbation`` value, checked whether or not the well has a ``well_perturbation`` row at all. Only once the fields agree is that consensus compared against the row, if one is present; a well with no row is unaffected, since the row is an optional, additive index. The image-group block is authoritative, so a mismatch is repaired by re-deriving the well row, never the reverse
     - experimental-metadata.rst § Where it is written
   * - MUST
     - Controlled-vocabulary enums (``modality``, ``control_class``, ``tissue_type``, ``source_kind``, the instrument enums)
     - experimental-metadata.rst § Conformance
   * - MUST
     - CURIE shape on ontology-id fields (e.g. ``^NCBITaxon:\d+$``, ``^EFO:\d+$``; the ``unknown`` / ``na`` / ``unavailable`` sentinels are permitted where the slot allows them). A few "no value" ontology terms are rejected so each sentinel has one spelling: ``ZFS:0000000``, ``FBdv:00007012`` and ``ZFA:0001093``
     - experimental-metadata.rst § Conformance
   * - MUST
     - ``modality: compound`` requires the ``chemical`` block; ``crispr`` / ``orf`` / ``shrna`` / ``mirna`` / ``sirna`` require the ``genetic`` block
     - experimental-metadata.rst § Conformance
   * - MUST
     - ``tissue_type: "cell line"`` requires ``cell_line_name``
     - experimental-metadata.rst § Conformance
   * - MUST
     - a JUMP-CP ``dataset_name`` requires ``cp_version``
     - experimental-metadata.rst § Conformance
   * - MUST
     - ``study_type`` including ``Other`` requires ``study_type_other``
     - experimental-metadata.rst § Conformance
   * - MUST
     - a non-``empty`` ``control_class`` requires a ``definition`` (``empty`` is exempt). Checked on the image group's ``perturbation`` block, and on a ``WellPerturbation``, which inherits the rule
     - experimental-metadata.rst § Conformance
   * - MUST
     - ``source_kind: agent`` requires a ``citation``
     - experimental-metadata.rst § Conformance
   * - MUST
     - Provenance coverage — an orphan ``provenance`` ``field_path`` (a key matching no populated field) is rejected
     - experimental-metadata.rst § Conformance
   * - SHOULD
     - Provenance coverage — every populated field should carry a ``provenance`` record (required for agent-curated values); a missing record is a WARNING
     - experimental-metadata.rst § Conformance
   * - SHOULD
     - A sentinel ``ontology_term_id`` constrains its ``label``: ``unknown`` / ``na`` repeat the sentinel, ``unavailable`` carries free text. A real term's label is checked against the registry only under ``--check-ontology``
     - experimental-metadata.rst § What is and is not machine-checked
   * - SHOULD
     - ``development_stage`` should use the stage ontology for the ``organism`` (HsapDv / MmusDv / ZFS / FBdv / WBls, or a UBERON life-cycle term). Checked only when exactly one organism is listed, and only for those five; sentinels are exempt
     - experimental-metadata.rst § What is and is not machine-checked
   * - SHOULD
     - For ``tissue_type: "cell line"``, ``tissue`` should carry a ``CVCL:`` accession, and ``cell_line_id`` should match it. Two warnings: ``tissue`` holding no accession at all, and the mirror disagreeing with it
     - experimental-metadata.rst § What is and is not machine-checked
   * - SHOULD
     - Beta in v0.2, required from v0.3: a ``compound`` perturbation's ``chemical`` block should carry ``pubchem_cid`` and ``morgan_fingerprint``, and a genetic one's ``genetic`` block should carry ``ensembl_gene_id``. One WARNING per missing identifier, on the image group's ``definition`` / ``cotreatment``, the same on a ``WellPerturbation``, and on a bare codebook entry. ``NA`` counts as present
     - experimental-metadata.rst § Perturbation identifiers are beta in v0.2

Provenance coverage is a cross-field check (the set of ``provenance`` keys vs. the
populated fields) that LinkML rules cannot express, so it is computed by the validator
alongside the model rules rather than by the schema. The two cross-slot SHOULDs above
are there for the same reason: a LinkML rule is a per-slot predicate and cannot compare
one slot against another. Both compare fields already present on the same ``Biosample``,
so neither needs to resolve a term.

``time_increment_s`` is also conditionally required (for time-lapse data), but its
condition is the ``zarr.json`` array shape rather than a metadata field, so it is checked
by the validator directly rather than by the generated schema — the one conditional
requirement outside the JSON Schema.

**What the validator does not check**

* **Term existence** — the CURIE patterns check shape (correct prefix, well-formed id),
  not that the term is real; confirming a term against pinned ontology versions is a
  deferred layer.
* **Mappings** — the ``exact_mappings`` are cross-references for interop / projection,
  not constraints.
* **Free-text fields** — unconstrained.
* **Grounding re-verification** — ``source_kind: agent`` requires a ``citation`` (enforced
  above), but re-checking that the quote actually appears in the source needs the raw
  source pages (not shipped with the store), so it is deferred; downstream trusts the
  carried ``match_status``.
* **Whether a ``perturbation`` or ``well_perturbation`` block should be there at all** —
  "this dataset has perturbations" is a fact about the experiment, not about the store,
  so no check can distinguish an unperturbed dataset from a screen whose curator forgot
  the block. The validator enforces each block's *contents* whenever it is present and
  never requires its presence; an absent block is always accepted at either placement
  (a possible future opt-in requirement).

Regenerating the schema artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The JSON Schema and Pydantic models are generated from the LinkML source and committed.
After editing ``schema/v0.2/dca_experimental_metadata.yaml``, regenerate both:

.. code-block:: bash

   make schema

This runs ``gen-json-schema --no-metadata`` (→
``schema/v0.2/generated/dca_experimental_metadata.schema.json``, the test-suite cross-check)
and ``gen-pydantic`` (→
``helpers/src/dca_helpers/validation/versions/v0_2/_generated_models.py``, the models the
validator enforces at runtime, can also be used by an agent and downstream consumers;
committed inside the package so it ships with any install). CI fails if the committed
artifacts drift from the schema, so commit the regenerated files alongside any schema
change.

S3 Access
---------

Set AWS credentials before calling ``validate``:

.. code-block:: bash

   AWS_PROFILE=my-profile dca-validate s3://my-bucket/data/plate.ome.zarr --spec-version 0.2

Or in Python:

.. code-block:: python

   import os
   os.environ["AWS_PROFILE"] = "my-profile"

   from dca_helpers.validation import validate
   run = validate("s3://my-bucket/data/plate.ome.zarr", spec_version="0.2")

.. note::

   For large HCS plates with thousands of wells, point ``validate`` at the plate root
   directly and use ``strategy="walk"`` rather than a parent directory.
   Recursive S3 discovery over thousands of ``zarr.json`` files can exhaust session
   credentials before completing. Validating the plate root also runs structural checks
   (e.g. verifying all declared wells exist as valid Zarr groups).
