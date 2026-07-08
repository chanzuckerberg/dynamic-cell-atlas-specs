Experimental Metadata Specification
====================================

Overview
--------

This specification defines how **experimental metadata** (i.e. the acquisition,
study, sample, and perturbation context surrounding a dataset) is recorded for
Dynamic Cell Atlas datasets. It complements the :doc:`channel-metadata`
specification (per-channel semantics) and :doc:`normalization-statistics`.

Experimental metadata is curated into a **sibling Parquet within the Zarr
store** by an agent or human curation from source text
(papers, portals, READMEs, file headers). This is the complete
record, the required fields and the optional ones alike, each with inline provenance;
the Parquet is the **source of truth**.

A small **required floor** (see :ref:`Conformance <experimental-conformance>`) is
additionally **mirrored into** the ``dca`` block in ``zarr.json``, alongside the channel
and normalization-statistics metadata it already carries (see :doc:`dca-metadata`), so
these stable, every-dataset fields are compliant with the `OME-NGFF storage format
<https://ngff.openmicroscopy.org/0.5/#storage-format>`_, which places group-level
metadata in the ``zarr.json`` attributes. That
``zarr.json`` copy is derived from the Parquet; **optional or not-yet-standardized**
fields stay in the Parquet alone until they are
:ref:`promoted <experimental-promotion>`.

Every curated value carries its **provenance as plain columns alongside it** (see
:ref:`Provenance <experimental-evidence>`), so each field is tied to its source and
independently verifiable. Because metadata is often derived from prose, keeping it in a
re-curatable Parquet lets values be corrected as the source understanding improves.

This document organizes the fields by REMBI component: **Study**, **Study Component**
(imaging method + assay class), **Biosample**, **Specimen** (sample preparation),
**Image Acquisition** (instrument), and **Image Data**, plus the DCA-native
**Perturbation** model, all pulling from existing standards.

.. seealso::

   See :doc:`experimental-metadata-example` for a worked example (an example cultured
   human iPSC dataset) of how a 3D+t live-imaging dataset is filled in,
   including which concepts still land in ``additional_metadata``.

.. _experimental-standards:

Standards Repurposed
--------------------

This specification is, by design, mostly an **assembly of existing community
standards**. Its overall structure — ``Study`` / ``StudyComponent`` / ``Biosample`` /
``Specimen`` / ``AcquisitionMetadata`` / ``ImageData`` — follows the `REMBI reporting guideline
<https://www.nature.com/articles/s41592-021-01166-8>`_; definitions of each REMBI
component these classes correspond to are on the `BioImage Archive REMBI Overview
<https://www.ebi.ac.uk/bioimage-archive/rembi-help-overview/>`_ and in the
`REMBI metadata table (paper supplement)
<https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-021-01166-8/MediaObjects/41592_2021_1166_MOESM1_ESM.pdf>`_.
Within that, every
field is anchored on a more specific established schema or ontology wherever one exists,
and DCA-native fields are added **only** where no standard covers the concept. Each field records its source as an ``exact_mappings`` (or
``close_mappings``) entry in the :download:`LinkML schema
<../../schema/v0.2/dca_experimental_metadata.yaml>`, which is the machine-readable record
of this table. The DCA spec is meant to be interoperable with the broader community.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Standard
     - What it contributes to this spec
   * - `Bio-Formats / OME <https://bio-formats.readthedocs.io/en/v8.5.0/metadata-summary.html>`_
     - Instrument & acquisition (microscope, objective, detector, imaging
       environment, time increment); study ``Experiment`` + people
       (``Experimenter``); the OME ``Reagent`` anchor for the ``Perturbation`` model.
   * - `CellPainting Gallery / Broad <https://github.com/broadinstitute/cellpainting-gallery-metadata/blob/main/harmonized_ontology.json>`_
     - Cell-line identity (name / type / modification); study fields (source, DOI,
       year imaged, CellProfiler version → ``Study.cp_version``); perturbation treatment
       fields, control classes and modalities. Also harmonizes the associated compound /
       ORF / CRISPR perturbation tables (one linkout).
   * - `CZI cross-modality v1.1.0 (CELLxGENE-aligned) <https://github.com/chanzuckerberg/data-guidance/blob/main/standards/cross-modality/1.1.0/schema.md>`_
     - Sample context as ``ontology_term_id`` + label pairs: organism, tissue,
       tissue_type, disease, development stage, assay. Incorporates Cellosaurus
       (``CVCL:`` terms, see ontologies below) for **cell lines** (``tissue_type``
       "cell line").
   * - `schema.org <https://schema.org/>`_
     - Dataset-level ``dataset_name`` / ``license`` / ``related_publication`` /
       ``release_date`` (``name`` / ``license`` / ``citation`` / ``datePublished``).
   * - `OPS Data Standard v0.1.0 <https://github.com/chanzuckerberg/ops-schema/blob/main/standards/ops/0.1.0/experimental-metadata.md>`_
     - Cell culture setup: ``Specimen.growth_conditions`` (→ ``cellular.growth_conditions``)
       and ``AcquisitionMetadata.plate_type`` (→ ``cellular.plate_type``). The
       cross-modality sample fields (``experiment.*``) share the same CZI
       ontology terms as DCA ``Biosample``.
   * - `CZI Scientific Dataset Catalog <https://github.com/chanzuckerberg/dataset-catalog>`_
     - The dataset-level container this metadata populates: ``dataset_type``
       (raw/processed) + a Lineage Edge ``transformed_from`` (→
       ``ImageData.processing_variant``); the ``sample`` /
       ``experiment`` ``{label, ontology_id}`` shape (→ DCA ``Biosample`` term-id +
       label pairs); and ``governance.license`` / ``doi``.
   * - Ontologies —
       `NCBITaxon <https://obofoundry.org/ontology/ncbitaxon>`_,
       `UBERON <https://obofoundry.org/ontology/uberon>`_,
       `CL <https://obofoundry.org/ontology/cl>`_,
       `MONDO <https://obofoundry.org/ontology/mondo>`_,
       `EFO <https://www.ebi.ac.uk/ols4/ontologies/efo>`_,
       `FBbi <https://obofoundry.org/ontology/fbbi>`_,
       `HsapDv <https://obofoundry.org/ontology/hsapdv>`_,
       `PATO <https://obofoundry.org/ontology/pato>`_,
       `Cellosaurus <https://www.cellosaurus.org/>`_,
       `NCBI Gene <https://www.ncbi.nlm.nih.gov/gene>`_,
       `HGNC <https://www.genenames.org/>`_,
       `Ensembl <https://www.ensembl.org/>`_,
       `PubChem <https://pubchem.ncbi.nlm.nih.gov/>`_
     - Term ids underpinning the cross-modality, imaging-method, cell-line and
       perturbation fields, enabling queries and cross-spec (e.g. OPS) projection.

.. _experimental-dca-native:

DCA extensions
--------------

The following are the only concepts this spec **adds**, each at a
genuine gap no referenced standard covers:

* **Per-value provenance (the main extension).** The inline ``Provenance`` columns —
  ``citation`` (``source_url`` / ``quote`` / ``match_status`` / ``char_start`` /
  ``char_end``),
  ``grounded``, ``source_kind`` — carried alongside *every* curated value. The
  catalog records dataset-level ``data_quality`` but nothing per-field; no referenced
  standard models per-value grounding/citation this way (it comes from the agent's
  provenance layer).
* **Storage + promotion model.** The full curated record in a sibling Parquet
  (source of truth), the required floor mirrored into ``zarr.json`` (alongside the
  channel / normalization-statistics metadata), and the
  :ref:`promotion <experimental-promotion>` path that extends that mirror to more fields.
  The catalog is storage-agnostic (its ``metadata`` is opaque JSON on the Dataset
  record); how that metadata is physically laid out in the store is a DCA convention.
* **``additional_metadata``** — the promotable, evidence-backed free-text bucket. The
  catalog permits extra key-value pairs, but the promotion + provenance mechanism is
  DCA-specific.
* **``Biosample.group_id``** — the sample↔image linkage key, reusing the Parquet (``tables/obs``) row key.
* **Structured, modality-agnostic ``Perturbation``** — the catalog carries an opaque
  ``sample.perturbation`` (``[json]``); DCA gives it structure, anchored on OME
  ``Reagent`` and generalized to chemical / genetic / physical / biological. REMBI
  scopes treatment as a single (unstructured) Biosample "extrinsic variable"; DCA
  instead models it as separate ``Perturbation`` (a reusable per-construct codebook) and
  ``WellPerturbation`` (the per-well assignment with dosing / control / timing) classes,
  because the grain differs from a single ``Biosample`` and a free-text entry cannot
  capture it. The link is by key: ``Biosample.group_id`` ↔ ``WellPerturbation.plate`` /
  ``well`` (the OME-NGFF HCS path).
* **``Specimen.specimen_state``** — the coarse specimen state (``live`` / ``fixed`` /
  ``unknown``), a DCA controlled vocabulary. No referenced standard structures the
  live-vs-fixed distinction cleanly across specimen types (OME / Bio-Formats has no
  sample-preparation fields; FBbi has only specific specimen-preparation terms, not a
  clean state axis), so it is DCA-native (REMBI-derived). The specific preparation
  methods (such as fixation subtype, sectioning, whole-mount, clearing, expansion) live in
  ``Specimen.preparation_method`` as FBbi ``{ontology_term_id, label}`` terms.

A few descriptive fields have **no machine-readable term source** and are carried as
DCA-native: ``Experimenter.orcid`` and the nested ``Publication`` fields (publication
title, authors, year). Everything else maps to a referenced standard or the catalog
via ``exact_mappings`` / ``close_mappings`` in the LinkML schema.

.. _experimental-conformance:

Conformance: Required, Recommended, Optional
--------------------------------------------

All experimental metadata lives in the Parquet; the **required** fields are
additionally mirrored into ``zarr.json`` (the ``dca`` block). The table below is the
field reference by category; for nested blocks (``microscope`` / ``objective`` /
``detector`` / ``imaging_environment``, ``experimenter``, ``related_publication``) it
names the block rather than each leaf — the :download:`LinkML schema
<../../schema/v0.2/dca_experimental_metadata.yaml>` is the exhaustive, authoritative list:

* **Required** — the field MUST be **present** and is mirrored into ``zarr.json``. Where
  a value does not exist or apply, state it with a sentinel (see the sentinel definitions
  below) rather than dropping the field, so a missing value is always *stated*, never
  silently omitted or fabricated. Most required fields apply **unconditionally** (the
  cross-modality term annotations, ``year_imaged``, ``imaging_method``); others are
  **required only when a
  condition holds** and omitted otherwise (the perturbation fields when a dataset has
  perturbations, ``cell_line_name`` for ``tissue_type`` "cell line", ``cp_version`` for a
  JUMP-CP ``dataset_name``, ``time_increment_s`` for time-lapse data).

  The sentinels, used in place of a value rather than dropping the field:

  * ``unknown`` — the value is not available to the curator (absent from the source data).
  * ``na`` — the field does not apply to this experiment or use case.
  * ``unavailable`` — (ontology-term fields) the value is known but no ontology term exists
    for it; the ``label`` then carries the free-text value.

  For an ontology-term field, when the ``ontology_term_id`` is ``unknown`` or ``na`` the
  ``label`` repeats that sentinel; when it is ``unavailable`` the ``label`` carries the
  free-text value. Which sentinels a field accepts is per field (e.g. ``imaging_method``
  accepts only a real FBbi term or ``unavailable``).
* **Recommended** — **SHOULD** be populated whenever the information exists (e.g.
  ``related_publication``, ``accession_id``, ``protocol_url``). Lives in the Parquet, not
  the ``zarr.json`` floor, and is not enforced. Unlike Optional, omitting it when the
  value *is* available is discouraged.
* **Optional** — **MAY** be populated; do so when available, and simply **omit**
  otherwise. Lives in the Parquet; an absent Optional field is fine and needs no
  sentinel.

The **Recommended** and **Optional** fields (the Parquet layer, outside the ``zarr.json``
floor) are **promotion candidates**: a future spec version may reclassify one as
**Required** (unconditionally or when a condition holds) and mirror it into the floor,
once it is verifiable, schema-stable, and consistently scoped (see
:ref:`Promotion <experimental-promotion>`).
``additional_metadata`` goes one step further: its free-text entries can first become
defined fields before being promoted.

.. list-table:: Experimental metadata fields
   :header-rows: 1
   :widths: 14 32 18 36

   * - Group
     - Field(s)
     - Required?
     - Standard / ontology
   * - Study
     - ``accession_id``
     - Recommended
     - present when deposited; omitted for in-house data
   * - Study
     - ``dataset_name``
     - Required
     - schema.org (``sc:name``) / catalog ``Dataset.name``; the data's own name,
       distinct from ``Publication.title``
   * - Study
     - ``description``
     - Required
     - OME ``Experiment.Description``
   * - Study
     - ``study_type``
     - Required
     - OME ``Experiment.Type``
   * - Study
     - ``source``
     - Required
     - CellPainting Gallery ``Source``
   * - Study
     - ``related_publication``
     - Recommended
     - schema.org (``sc:citation``); each entry self-contained, per-entry ``doi`` =
       CPG ``DOI_to_Cite``
   * - Study
     - ``year_imaged``
     - Required (``unknown`` if not determinable)
     - CPG ``Year_Imaged``
   * - Study
     - ``cp_version``
     - Required when JUMP-CP
     - CPG ``CP_Version``
   * - Study
     - ``protocol_url``
     - Optional
     - DCA-native (protocols.io)
   * - Study
     - ``release_date``
     - Required
     - schema.org (``sc:datePublished``)
   * - Study
     - ``license``
     - Required
     - schema.org (``sc:license``)
   * - Study
     - ``experimenter`` (name / email / institution)
     - Required
     - OME ``Experimenter``
   * - Sample
     - ``organism`` (``ontology_term_id`` + ``label``)
     - Required
     - cross-modality · NCBITaxon
   * - Sample
     - ``tissue_type``
     - Required
     - cross-modality (``tissue`` / ``cell culture`` / ``cell line`` / ``organoid``)
   * - Sample
     - ``tissue`` (``ontology_term_id`` + ``label``)
     - Required
     - cross-modality · UBERON / CL
   * - Sample
     - ``disease`` (``ontology_term_id`` + ``label``)
     - Required
     - cross-modality · MONDO / PATO
   * - Sample
     - ``development_stage`` (``ontology_term_id`` + ``label``)
     - Required
     - cross-modality · HsapDv
   * - Study component
     - ``assay`` (``ontology_term_id`` + ``label``; assay / experiment class)
     - Required
     - cross-modality · **EFO only**
   * - Specimen
     - ``specimen_state`` (coarse live/fixed state; FBbi
       ``preparation_method`` terms for the specifics)
     - Required
     - DCA controlled vocab (REMBI-derived) · FBbi optional
   * - Sample
     - ``cell_line_name`` / ``cell_line_id`` / ``cell_line_type`` / ``cell_line_modification`` / ``parent_cell_line_id``
     - ``cell_line_name`` required when ``tissue_type`` = "cell line"; the others
       recommended (a Cellosaurus ``cell_line_id`` may be absent for iPSC / primary
       lines, and the authoritative cell-line identity is the ``tissue`` term)
     - CellPainting Gallery + Cellosaurus (``CVCL:``)
   * - Specimen
     - ``growth_conditions``
     - Optional
     - OPS ``cellular.growth_conditions``
   * - Acquisition
     - ``plate_type``
     - Optional
     - OPS ``cellular.plate_type``
   * - Perturbation
     - ``modality`` / ``control_class`` (+ the per-well ``perturbation`` reference);
       ``perturbation_identifier`` and ``concentration`` optional
     - Required when perturbed
     - OME ``Reagent`` + CellPainting Gallery
   * - Perturbation
     - chemical (``inchikey`` / ``inchi`` / ``smiles`` / ``pubchem_cid`` / ``chembl_id``)
     - Required when chemical
     - CellPainting Gallery
   * - Perturbation
     - genetic (``gene_symbol`` / ``ncbi_gene_id`` / ``vector`` …)
     - Required when CRISPR / ORF
     - CellPainting Gallery (JUMP) + HGNC / NCBIGene
   * - Perturbation
     - per-well dosing / timing (``solvent`` / ``duration_h`` /
       ``timepoint_primary_h`` / ``timepoint_secondary_h`` / ``secondary_perturbation``)
     - Optional
     - CellPainting Gallery (``WellPerturbation``)
   * - Study component
     - ``imaging_method`` (``ontology_term_id`` + ``label``; acquisition modality)
     - Required (``unavailable`` only if no FBbi term fits)
     - **FBbi only**
   * - Acquisition
     - ``time_increment_s``
     - Required when time-lapse
     - OME ``Image.TimeIncrement``
   * - Acquisition
     - ``timepoint_acquisition``
     - Optional
     - CellPainting Gallery ``Timepoint_Acquisition``
   * - Image data
     - ``processing_variant``
     - Optional
     - DCA-native · zarr-local mirror of the Scientific Dataset Catalog
       ``dataset_type`` (``raw`` / ``processed``)
   * - Acquisition
     - ``microscope`` / ``objective`` / ``detector`` / ``imaging_environment`` / ``acquisition_date``
     - Optional
     - OME (Bio-Formats)
   * - Extra
     - ``additional_metadata`` — extra study-provided values not yet defined in this spec
     - Optional
     - DCA-native · agent-minted, each with inline provenance columns. Lives in the
       Parquet and can be **elevated** to a defined (and ultimately required,
       ``zarr.json``) field once it stabilizes — see
       :ref:`Promotion <experimental-promotion>`.

The :ref:`provenance columns <experimental-evidence>` are **REQUIRED for
agent-curated values** and **OPTIONAL for manually entered values** (set
``source_kind: human``; the citation MAY be blank).

.. note::

   Some CellPainting Gallery fields are deliberately **out of scope here** because
   they belong to another layer. The per-channel staining/optics details
   (``Label_Mechanism`` / ``Label_Molecule`` / ``Label_Reagent`` /
   ``Label_Fluorophore`` / ``Label_Structure`` and the excitation/emission
   peaks/widths) belong with **channel metadata** (see :doc:`channel-metadata`);
   they are not defined as structured fields there, so they are usually recorded in
   the channel's free-text ``description``. The structural fields
   (``File*``, ``Batch`` / ``Plate`` / ``Well`` / ``Site``, ``Plate_Size``,
   ``Image_*``, ``Microscope_Pixel_Size``) live in the array / HCS structural layer
   (``zarr.json`` dimensions, dtype, scale, plate layout), not in experimental
   metadata.

.. note::

   **Identity, authority, and image linkage.** Several facts are reachable through
   more than one field; each has a single **authoritative** slot, with the others as
   convenience mirrors:

   * Cell-line identity → ``tissue``'s ``ontology_term_id`` (a Cellosaurus ``CVCL:``
     term) is authoritative; ``cell_line_id`` mirrors it.
   * Publication → each ``related_publication`` entry is a self-contained citation
     (``doi`` + title + authors + year).
   * Assay vs. modality are **distinct axes, one ontology each** (not mirrors):
     ``assay`` (EFO only) is the assay / experiment class; the acquisition modality is
     the Study-Component ``imaging_method`` (FBbi only). Both are Study-Component-level
     fields, each an ``ontology_term_id`` + ``label`` pair.

   All ontology / database identifiers are stored as **colon CURIEs** (e.g.
   ``CVCL:0030``, ``NCBIGene:7157``, ``PubChem:2244``) that expand to a resolvable URL
   via the schema ``prefixes`` block, so a stored value names its own source database
   rather than relying on the slot name. Sample-to-image linkage is carried by
   ``Biosample.group_id`` — the row key of the Parquet (``tables/obs``) table,
   with one row per image group: a single-image store has one row, an HCS plate one row
   per well (which ``WellPerturbation.plate`` / ``well`` also key on).

.. _experimental-promotion:

Promotion to ``zarr.json``
--------------------------

The required floor is already mirrored into ``zarr.json``; every *optional* or
*recommended* field stays in the Parquet alone. This keeps agent-extracted,
unstandardized, or experiment-specific fields queryable as soon as they are curated, so
consumers do not have to wait for a spec revision before they are available. An optional
or recommended field **MAY** be promoted into the ``zarr.json`` ``dca`` block in a future
spec version (joining the required floor), but only when **all** of the following hold:

1. **Verifiable** — the value can be confirmed deterministically (measured from the
   array, or via a reproducible grounded / cross-source check), not solely a
   model's prose extraction.
2. **Schema-stable** — its shape and vocabulary are settled and not expected to
   churn as taxonomies / ontologies update.
3. **Consistently scoped** — its applicability is well defined, so a consumer
   knows when to expect it. This can be either:

   * **corpus-wide** — it applies to essentially all datasets (written uniformly,
     ``null`` / ``"unknown"`` where unavailable) and joins the unconditional
     required floor; or
   * **experiment-class conditional** — it is *optional corpus-wide but required
     for a defined class of experiments* (e.g. perturbation fields for screens,
     ``cell_line_*`` when ``tissue_type`` is "cell line"). Such a field is promoted
     to ``zarr.json`` and made **conditionally required** on that class, rather than
     left optional — so consumers of that experiment type can rely on it.

Promotion is a **per-field specification decision made via a version bump**. For a
corpus-wide field it applies uniformly; for an experiment-class field it applies
to that class. When a field is promoted, the Parquet remains the **source of truth**
and the ``zarr.json`` copy is a derived, versioned snapshot.

Acquisition Metadata
--------------------

Image Acquisition is about how the images were captured. It is optional, so it lives
in the Parquet (not ``zarr.json``). The only field expected across datasets is
``time_increment_s`` (conditional, for time-lapse); see the field reference under
:ref:`Conformance <experimental-conformance>`.

A minimal, **optional** OME instrument set — ``microscope``, ``objective``,
``detector``, ``imaging_environment``, and ``acquisition_date`` — is defined in the
:download:`LinkML schema <../../schema/v0.2/dca_experimental_metadata.yaml>` (with OME
``exact_mappings``) but not enumerated here.

**Z-calibration correction factors.** Some objectives introduce a refractive-index
mismatch between the immersion medium and the sample that causes the physical Z step
to differ from the nominal (stage-commanded) value. When a correction factor is
known (e.g. a 1.43× axial-distortion correction for a 20×/0.8 NA air objective
imaging through aqueous medium per Diel et al. 2020), record it in
``AcquisitionMetadata.additional_metadata`` as
``{"name": "z_ri_correction_factor", "value": <factor>}`` alongside the nominal Z
spacing already stored in ``coordinateTransformations``. The corrected physical Z step
is ``nominal_z_step_um × z_ri_correction_factor``; downstream analyses requiring
physically-accurate axial distances should apply this multiplier.

Image Data
----------

Image Data contains image-level metadata. The one structured field here is
``ImageData.processing_variant``: the zarr-local tag for the computational variant that
produced this array (e.g. ``raw`` / ``denoised`` / ``deconvolved``); the full derivation
(parent dataset, algorithm, parameters) is a Scientific Dataset Catalog
``transformed_from`` Lineage Edge, not duplicated here. The remaining image-level fields
(format, dimensions, pixel and voxel size, channels) are not otherwise captured here:
they live in the `OME-NGFF spec <https://ngff.openmicroscopy.org/0.5/>`_ arrays and the
:doc:`Channel Metadata Spec <channel-metadata>`.

Study and Study Component
-------------------------

The sub-groups, and the standard each field draws on:

* ``study`` — experiment fields on OME ``Experiment`` (type, description) and the
  CellPainting Gallery study fields (``Source``, ``DOI_to_Cite``, ``Year_Imaged``);
  people on OME ``Experimenter`` (name, email, institution); dataset-level
  ``dataset_name`` / ``license`` / ``related_publication`` / ``release_date`` on
  **schema.org** (``name`` / ``license`` / ``citation`` / ``datePublished``).
* ``study_component`` — two experiment-class descriptors, each an
  ``ontology_term_id`` + ``label`` pair: ``assay``
  (`EFO <https://www.ebi.ac.uk/ols4/ontologies/efo>`_: the assay / experiment class,
  cross-modality / CELLxGENE-aligned) and ``imaging_method``
  (`FBbi <https://obofoundry.org/ontology/fbbi>`_: the acquisition modality, e.g.
  confocal or light-sheet). One ontology per field: EFO answers "what kind of
  experiment", FBbi "how was it imaged". Both are REQUIRED for every imaging dataset
  (``imaging_method`` uses ``unavailable`` with a free-text label only when no FBbi term
  fits; ``unknown`` and ``na`` are not permitted).

Biosample and Specimen
----------------------

**Biosample** is about what was imaged, for example the species of the organism being
imaged, whether a particular cell line is used, genetic background, etc.
**Specimen** describes how the sample was prepared for imaging.

The sub-groups, and the standard each field draws on:

* ``biosamples`` — **cell-line identity** from the CellPainting Gallery
  (``cell_line_name`` / ``cell_line_type`` / ``cell_line_modification`` + a
  Cellosaurus ``cell_line_id``; ``parent_cell_line_id`` for the parental line
  when the imaged line was derived by gene editing or reprogramming);
  **sample context** reuses the
  `CZI cross-modality schema v1.1.0
  <https://github.com/chanzuckerberg/data-guidance/blob/main/standards/cross-modality/1.1.0/schema.md>`_
  (CELLxGENE-aligned) as ``ontology_term_id`` + ``label`` object pairs: ``organism``
  (NCBITaxon), ``tissue`` (UBERON / CL) + ``tissue_type``, ``disease`` (MONDO /
  PATO), and ``development_stage`` (HsapDv) — each a list of
  ``{ontology_term_id, label}`` objects. When ``tissue_type`` is ``"cell culture"``,
  ``tissue``'s ``ontology_term_id`` takes a Cell Ontology
  (`CL <https://obofoundry.org/ontology/cl>`_) term identifying the cell type (e.g.
  ``CL:0002620`` for induced pluripotent stem cell). See the cross-modality schema for
  the full ``tissue_type`` vocabulary and term guidance.
* ``specimens`` — ``specimen_state`` records the coarse specimen state at imaging time
  (``live`` / ``fixed`` / ``unknown``), a DCA controlled vocabulary (REMBI Specimen
  module; no source ontology structures the live-vs-fixed state). The multivalued
  ``preparation_method`` carries the specific FBbi sample-preparation method(s) — a
  descendant of ``FBbi:00000001``, as an ``{ontology_term_id, label}`` term — with
  ``unavailable`` + a free-text ``label`` when no FBbi term fits (e.g. clearing,
  expansion), ``na`` when no preparation applies (e.g. a live sample), and ``unknown``
  when the source does not record it.

.. note::

   **Gene-edited cell lines (``parent_cell_line_id``).** When the imaged line was
   derived from a registered parental line by CRISPR editing or reprogramming,
   record the parental Cellosaurus accession in ``parent_cell_line_id`` (e.g.
   ``"CVCL:Y803"`` for WTC-11 hiPSC). ``cell_line_id`` then identifies the
   engineered clone.

   **hiPSC and reprogrammed cell lines (development stage).** Keep
   ``organism`` (the donor's species), but set ``development_stage``'s
   ``ontology_term_id`` to ``"na"``: a reprogrammed cell line has no organismal life
   stage (per cross-modality). Donor age, reprogramming method, and passage number go
   in ``Biosample.additional_metadata``; the cells' pluripotent identity goes in
   ``tissue``'s ``ontology_term_id`` (a Cell Ontology stem-cell term).

   **Organoids (``tissue_type = "organoid"``).** Set ``tissue``'s
   ``ontology_term_id`` to the UBERON term for the tissue of origin (not a
   ``CVCL:`` accession); the ``cell_line_*`` fields do not apply.

See the field reference under :ref:`Conformance <experimental-conformance>` for the
full Study / Sample field list with required-vs-optional status, and the
``Study``, ``Experimenter``, ``Biosample``, ``Specimen`` classes in the
:download:`LinkML schema <../../schema/v0.2/dca_experimental_metadata.yaml>` for the
authoritative field list.

.. _experimental-gap-fill:

Free-Text Gap-Fill
~~~~~~~~~~~~~~~~~~~

Free-text ``additional_metadata`` is the mechanism for concepts that have **no
stable structured schema** in this version. Examples include the **iPSC-specific
identifiers** (donor, clone, hPSCreg id, differentiation state) that the
structured ``cell_line_*`` fields do not capture, and any other field a standard
model does not cover. Rather than forcing
premature structure, such values are recorded as ``additional_metadata`` entries
available on ``study`` and ``biosamples``:

.. code-block:: json

   {
     "name": "differentiation_state",
     "value": "definitive endoderm, day 5 of directed differentiation",
     "provenance": {
       "grounded": true,
       "citation": {
         "source_url": "https://doi.org/10.1234/example",
         "quote": "hiPSCs were differentiated toward definitive endoderm over five days",
         "match_status": "exact"
       },
       "source_kind": "agent"
     }
   }

Each ``additional_metadata`` entry **MUST** carry a ``name`` and ``value`` and
**SHOULD** carry its :ref:`provenance columns <experimental-evidence>`. An agent
**MAY** mint new ``additional_metadata`` fields; as with every agent-curated value,
each **MUST** carry its extracted value, source citation, and grounding flag.
As fields here stabilize and gain a community schema, they are promoted to
first-class structured slots in a future version. ``additional_metadata`` is a
**staging area**, not a permanent home.

Perturbation Metadata
~~~~~~~~~~~~~~~~~~~~~~~

Perturbation metadata is modelled as a **modality-agnostic ``Perturbation``**
(mapped to the OME ``Reagent`` entity), specialized by a ``modality`` discriminator
into a **chemical** or **genetic** block. This generalizes beyond chemical
perturbation to **CRISPR / gene knockout / ORF** (and ``physical`` / ``biological``).
The genetic block carries the gene target that CellPainting Gallery cannot express
(its ``Treatment_Category`` enum names ``CRISPR``/``ORF`` but provides no gene
field). The fields are harmonized across three sources: OME/Bio-Formats ``Reagent``,
the CellPainting Gallery ``Treatment_*`` fields, and the JUMP compound/orf/crispr
tables.

These fields apply to **any perturbation dataset**, not just CellPainting Gallery /
JUMP ones; the ``cpg:`` / JUMP entries are alignment mappings (provenance), not a
requirement that the data originate from those projects. A custom drug screen, an
IDR dataset, or an in-house CRISPR experiment populates the **same** ``Perturbation``
/ ``WellPerturbation`` structure, with values sourced from wherever its metadata
lives (``source_kind`` records that, e.g. ``original_data`` from the source's own CSV).

**Required when present.** A dataset with no perturbations simply has no
``Perturbation`` / ``WellPerturbation``. But **if a dataset has perturbations**, the
core fields are **required** and written to ``zarr.json`` (well level):
``perturbation_id`` + ``modality`` on the ``Perturbation``, and ``plate`` / ``well``
/ ``control_class`` on the ``WellPerturbation``. ``perturbation`` is required on
every **treated or control** well (``treatment`` / ``negcon`` / ``poscon``) but is
**omitted for an** ``empty`` **well**, which has nothing to record. The ``chemical``
vs ``genetic`` block is required according to ``modality``. All three conditionals
are enforced as schema rules (see Validation).

The ``Perturbation`` *definition* (a codebook entry) is reusable across wells. Its
**assignment** is recorded per ``(plate, well)`` for **arrayed** screens as a
``WellPerturbation`` row. See the ``Perturbation``, ``ChemicalPerturbation``,
``GeneticPerturbation``, and ``WellPerturbation`` classes in the
:download:`LinkML schema <../../schema/v0.2/dca_experimental_metadata.yaml>`.

.. list-table:: Cross-source field mapping (minimal set)
   :header-rows: 1
   :widths: 26 24 25 25

   * - Concept
     - OME ``Reagent``
     - CellPainting Gallery
     - JUMP
   * - External id
     - ``ReagentIdentifier``
     - ``Treatment_Broad_Sample``
     - ``broad_sample`` / ``jcp2022``
   * - Modality
     - —
     - ``Treatment_Category``
     - ``perturbation_modality``
   * - Control class
     - —
     - ``Treatment_Control_Class``
     - control ``pert_type``
   * - Concentration
     - —
     - ``Treatment_Concentration`` (str)
     - structured value+unit
   * - Compound ids
     - (``AnnotationRef``)
     - ``InChIKey`` / ``SMILES`` / ``PubChem_CID``
     - ``inchikey`` / ``inchi`` / ``smiles``
   * - **Gene target**
     - —
     - **absent**
     - ``symbol`` + ``ncbi_gene_id`` (+ ORF ``transcript``)

The reconciled vocabularies (see the schema enums):

* ``modality`` — lowercased superset of CPG ``Treatment_Category`` and JUMP
  ``perturbation_modality``, plus non-molecular categories: ``compound`` | ``orf`` |
  ``crispr`` | ``shrna`` | ``mirna`` | ``physical`` (e.g. heat shock) |
  ``biological`` | ``other`` | ``unknown``.
* ``control_class`` — ``treatment`` | ``negcon`` | ``poscon`` | ``empty`` (CPG's
  redundant ``Control`` dropped; JUMP's ``empty`` retained).

Known gaps and contradictions:

* **Gene target gap** — CellPainting Gallery has no gene field; filled from JUMP.
* **Concentration type** — CPG records free text; DCA and JUMP use
  ``{value, unit}``. The schema keeps structured ``value``/``unit`` with a
  free-text ``text`` fallback.
* **Gene identifier** — NCBI Gene ID is the canonical key (JUMP-native); HGNC
  symbol is the label; Ensembl is optional cross-map.

.. _experimental-deferred-pooled:

Deferred: pooled / optical pooled screens (OPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``WellPerturbation`` covers **arrayed** screens (one perturbation per well). In a
**pooled / optical pooled screen (OPS)** a single well contains a library of many
sgRNAs and the perturbation is assigned **per cell** by decoding an integrated
barcode after imaging. That assignment is therefore **not** expressible per well. The
pooled-screen reagent fields (sgRNA protospacer, PAM, barcode, target locus) and the
per-cell assignment table are **deferred** to a future version, as is the broader move
from this spec's current **Parquet** tables to AnnData / SpatialData conventions for
single-cell data. The ``Perturbation`` *definition* above is unchanged for pooled
libraries; only the level at which a perturbation is assigned differs (per well for
arrayed screens, per cell for pooled).

.. _experimental-evidence:

Provenance
~~~~~~~~~~

Because metadata can be extracted by an agent, each value carries its **provenance**
(``source_kind`` + grounding ``citation`` + ``grounded`` flag). ``source_kind`` is
**always required** — it is the discriminator that tells a consumer whether a value was
``agent``-extracted, ``human``-entered, or taken from the source's own
``original_data``. The grounding ``citation`` is **required for ``agent`` values** and
**optional for ``human`` / ``original_data``** (left blank when there is nothing to
cite). This lets a consumer decide how much to trust a value without re-reading the
source. The provenance has two equivalent layouts for the two stores:

* In the **Parquet** it is **plain columns alongside each value** (one row per
  group, companion columns per field).
* In the **object form** (``zarr.json`` / the worked example) it is the
  ``ExperimentalMetadata.provenance`` **map**, keyed by each value's dotted
  ``field_path`` (e.g. ``biosample.organism``). The values stay plain
  and provenance travels as a parallel layer; a deterministic post-ingest step folds the
  two together. Free-text ``additional_metadata`` values are keyed the same way, by
  ``<block>.additional_metadata.<name>``, so all provenance lives in the one map.

``grounded`` is **computed deterministically by the agent's grounding gate, not
self-reported by the model**: a *grounding gate* checks whether the cited ``quote`` is
an (exact or fuzzy) substring of the fetched source page, recording the result as
``match_status`` (``exact`` / ``fuzzy`` / ``none``) plus the character offsets of the
match. The gate runs at curation time against the raw source pages; ``match_status``
and the offsets carry its result so the grounding stays **auditable downstream**, where
the raw corpus is no longer available.

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Field
     - Type
     - Description
   * - ``citation.source_url``
     - string (uri)
     - Where the value was read from.
   * - ``citation.quote``
     - string
     - Verbatim excerpt from the source supporting the value.
   * - ``citation.match_status``
     - enum
     - Grounding-gate result locating the quote in the source page: ``exact`` /
       ``fuzzy`` (grounded) | ``none`` (ungrounded).
   * - ``citation.char_start`` / ``char_end``
     - integer
     - Half-open span of the located quote in the source page's raw text; blank when
       ``match_status`` is ``none``.
   * - ``grounded``
     - boolean
     - ``true`` if the grounding gate located the cited quote (exact or fuzzy) in the
       fetched source text — the cited evidence is real, not fabricated. **Computed,
       not self-reported.** (Boolean rollup of ``match_status``.)
   * - ``source_kind``
     - enum
     - ``agent`` (provenance required) | ``human`` (manually entered; provenance
       optional) | ``original_data`` (from the source's own structured metadata;
       provenance optional).

Validation
----------

The :download:`LinkML schema <../../schema/v0.2/dca_experimental_metadata.yaml>` is the
authoring layer and single source of truth for all three groups; from it we generate
**Pydantic models** (enforced by the :doc:`validator <validator>`) and a committed
**JSON Schema** (interop artifact and test-suite oracle cross-check).

Authoring in LinkML (rather than hand-maintaining JSON Schema and Pydantic) also gives
machine-readable cross-spec alignment: each slot records its ``exact_mappings`` to
Bio-Formats/OME, CellPainting Gallery, CZI cross-modality, schema.org, the
Scientific Dataset Catalog, and external ontologies. The same ``Study`` / ``Biosample``
classes can therefore be validated against, or projected to, other specs that publish a
LinkML schema (e.g. OPS) without a second validator.
