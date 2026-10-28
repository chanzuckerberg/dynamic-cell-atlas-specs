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

The same record is **mirrored into** the ``dca`` block in ``zarr.json``, alongside the
channel and normalization-statistics metadata it already carries (see
:doc:`dca-metadata`), so these fields are compliant with the `OME-NGFF storage format
<https://ngff.openmicroscopy.org/0.5/#storage-format>`_, which places group-level
metadata in the ``zarr.json`` attributes. That ``zarr.json`` copy is derived from the
Parquet. A small **required floor** (see :ref:`Conformance <v03-experimental-conformance>`)
MUST be present; the recommended and optional fields MAY be omitted, and are validated
when present. Every value carries its **provenance alongside it** (see
:ref:`Provenance <v03-experimental-evidence>`), so each field is tied to its source and
independently verifiable.

This document organizes the fields by REMBI component: **Study**, **Study Component**
(imaging method + assay class), **Biosample**, **Specimen** (sample preparation),
**Image Acquisition** (instrument), and **Image Data**, plus the DCA-native
**Perturbation** model, all pulling from existing standards.

.. seealso::

   See :doc:`experimental-metadata-example` for a worked example (an example cultured
   human iPSC dataset) of how a 3D+t live-imaging dataset is filled in,
   including which concepts still land in ``additional_metadata``.

.. _v03-experimental-standards:

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
<../../schema/v0.3/dca_experimental_metadata.yaml>`, which is the machine-readable record
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
   * - `cryoET Data Portal core (cdp-common) <https://github.com/chanzuckerberg/cryoet-data-portal-backend/blob/main/schema/core/v2.0.0/common.yaml>`_
     - Sibling CZI standard. ``close_mappings`` from the biosample and specimen axes to
       its flat ``organism_taxid`` / ``organism_name``, ``tissue_id`` / ``tissue_name``,
       ``cell_type_id`` / ``cell_name``, ``cell_strain_id`` / ``cell_strain_name``,
       ``development_stage_id`` / ``development_stage_name``,
       ``preparation_sample_type`` and ``sample_preparation`` slots (cryoET splits name
       + id; DCA uses one ``ontology_term_id`` + label pair).
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
       `ZFA <https://obofoundry.org/ontology/zfa>`_,
       `MONDO <https://obofoundry.org/ontology/mondo>`_,
       `EFO <https://www.ebi.ac.uk/ols4/ontologies/efo>`_,
       `FBbi <https://obofoundry.org/ontology/fbbi>`_,
       `HsapDv <https://obofoundry.org/ontology/hsapdv>`_,
       `MmusDv <https://obofoundry.org/ontology/mmusdv>`_,
       `ZFS <https://obofoundry.org/ontology/zfs>`_,
       `FBdv <https://obofoundry.org/ontology/fbdv>`_,
       `WBls <https://obofoundry.org/ontology/wbls>`_,
       `PATO <https://obofoundry.org/ontology/pato>`_,
       `Cellosaurus <https://www.cellosaurus.org/>`_,
       `NCBI Gene <https://www.ncbi.nlm.nih.gov/gene>`_,
       `HGNC <https://www.genenames.org/>`_,
       `Ensembl <https://www.ensembl.org/>`_,
       `PubChem <https://pubchem.ncbi.nlm.nih.gov/>`_
     - Term ids underpinning the cross-modality, imaging-method, cell-line and
       perturbation fields, enabling queries and cross-spec (e.g. OPS) projection.

.. _v03-experimental-dca-native:

DCA extensions
--------------

The following are the only concepts this spec **adds**, each at a
genuine gap no referenced standard covers:

* **Per-value provenance (the main extension).** The inline ``Provenance`` fields —
  ``citation`` (``source_url`` / ``quote`` / ``match_status``) and ``source_kind`` —
  carried alongside *every* curated value. The
  catalog records dataset-level ``data_quality`` but nothing per-field; no referenced
  standard models per-value grounding/citation this way (it comes from the agent's
  provenance layer).
* **Storage + promotion model.** The full curated record in a sibling Parquet
  (source of truth), mirrored into ``zarr.json`` (alongside the
  channel / normalization-statistics metadata) with the required floor as the validated
  subset, and the :ref:`promotion <v03-experimental-promotion>` path that brings more fields
  into that subset.
* **``additional_metadata``** — the promotable, evidence-backed free-text bucket. The
  catalog permits extra key-value pairs, but the promotion + provenance mechanism is
  DCA-specific.
* **``Biosample.group_id``** — the sample↔image linkage key, reusing the Parquet (``tables/obs``) row key.
* **Structured, modality-agnostic ``Perturbation``** — the catalog carries an opaque
  ``sample.perturbation`` (``[json]``); DCA gives it structure, anchored on OME
  ``Reagent`` and generalized to chemical / genetic / physical / biological. REMBI
  scopes treatment as a single (unstructured) Biosample "extrinsic variable"; DCA
  instead models it as separate ``Perturbation`` (a reusable per-construct codebook) and
  ``PerturbationAssignment`` (the assignment with dosing / control / timing) classes,
  because the grain differs from a single ``Biosample`` and a free-text entry cannot
  capture it. The assignment travels with the image group as the ``perturbation``
  block; an arrayed HCS screen additionally indexes it per well as
  ``WellPerturbation``, linked by key: ``Biosample.group_id`` ↔
  ``WellPerturbation.plate`` / ``well`` (the OME-NGFF HCS path).
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

.. _v03-experimental-conformance:

Conformance: Required, Recommended, Optional
--------------------------------------------

All experimental metadata lives in the Parquet and is mirrored into ``zarr.json``
(the ``dca`` block); the **required** fields are the ones a store MUST carry:

* **Required** — the field MUST be **present** and is mirrored into ``zarr.json``. Where
  a value does not exist or apply, state it with a sentinel (see the sentinel definitions
  below) rather than dropping the field, so a missing value is always *stated*, never
  silently omitted or fabricated. Most required fields apply **unconditionally** (the
  cross-modality term annotations, ``year_imaged``, ``imaging_method``); others are
  **required only when a
  condition holds** and omitted otherwise (the perturbation fields when a dataset has
  perturbations, ``cell_line_name`` for ``tissue_type`` "cell line", ``cp_version`` for a
  JUMP-CP ``dataset_name``, ``study_type_other`` when ``study_type`` includes "Other",
  ``time_increment_s`` for time-lapse data, ``timestamps`` when ``time_sampling`` is
  ``variable``).

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
  ``related_publication``, ``accession_id``, ``protocol_url``). Written like every other
  field, and outside the required floor, so a store is not failed for omitting it —
  though what is present is validated. Unlike Optional, omitting it when the value
  *is* available is discouraged.
* **Optional** — **MAY** be populated; do so when available, and simply **omit**
  otherwise. Outside the required floor; an absent Optional field is fine and needs no
  sentinel, and a present one is validated like any other.

The **Recommended** and **Optional** fields (written, but outside the required floor)
are **promotion candidates**: a future spec version may reclassify one as
**Required** (unconditionally or when a condition holds) and bring it into the floor,
once it is verifiable, schema-stable, and consistently scoped (see
:ref:`Promotion <v03-experimental-promotion>`).
``additional_metadata`` goes one step further: its free-text entries can first become
defined fields before being promoted.

The table below is the field reference by category. For nested blocks (``microscope`` /
``objective`` / ``detector`` / ``imaging_environment``, ``experimenter``,
``related_publication``) it names the block rather than each leaf; the
:download:`LinkML schema <../../schema/v0.3/dca_experimental_metadata.yaml>` is the
exhaustive, authoritative list.

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
     - ``study_type_other``
     - Required when ``Other``
     - free text; used only when ``study_type`` includes ``Other``
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
       + ``unknown``
   * - Sample
     - ``tissue`` (``ontology_term_id`` + ``label``)
     - Required
     - cross-modality · UBERON / CL / ZFA
   * - Sample
     - ``disease`` (``ontology_term_id`` + ``label``)
     - Required
     - cross-modality · MONDO / PATO
   * - Sample
     - ``development_stage`` (``ontology_term_id`` + ``label``)
     - Required
     - cross-modality · HsapDv / MmusDv / ZFS / FBdv / WBls / UBERON
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
     - ``modality`` / ``control_class`` (+ the ``definition`` reference);
       ``perturbation_identifier`` and ``concentration`` optional
     - Required (``{"modality": "no perturbations"}`` when nothing was applied)
     - OME ``Reagent`` + CellPainting Gallery
   * - Perturbation
     - chemical (``inchikey`` / ``inchi`` / ``smiles`` / ``pubchem_cid`` / ``chembl_id``)
     - Required when chemical
     - CellPainting Gallery
   * - Perturbation
     - genetic (``gene_symbol`` / ``ncbi_gene_id`` / ``vector`` …)
     - Required when genetic (CRISPR / ORF / shRNA / miRNA / siRNA)
     - CellPainting Gallery + HGNC / NCBIGene
   * - Perturbation
     - dosing / timing (``solvent`` / ``duration_h`` /
       ``timepoint_primary_h`` / ``timepoint_secondary_h`` / ``cotreatment``)
     - Optional
     - CellPainting Gallery (``PerturbationAssignment``)
   * - Study component
     - ``imaging_method`` (``ontology_term_id`` + ``label``; acquisition modality)
     - Required (``unavailable`` only if no FBbi term fits)
     - **FBbi only**
   * - Acquisition
     - ``time_increment_s``
     - Required when time-lapse
     - OME ``Image.TimeIncrement``; the **nominal** cadence when sampling is variable
   * - Acquisition
     - ``time_sampling`` (``uniform`` / ``variable``)
     - Optional; absent reads as ``uniform``
     - DCA vocabulary. ``variable`` relaxes the temporal-axis unit MUST
       (:doc:`ome-zarr-metadata`) and requires ``timestamps``
   * - Acquisition
     - ``timestamps``
     - Required when ``time_sampling`` is ``variable``
     - Per-frame acquisition times as seconds since the Unix epoch (UTC), one per
       timepoint of the level-0 array, non-decreasing
   * - Acquisition
     - ``timepoint_acquisition``
     - Optional
     - CellPainting Gallery ``Timepoint_Acquisition``
   * - Image data
     - ``processing_variant``
     - Optional
     - DCA-native · ordered list of processing steps; zarr-local mirror of the
       Scientific Dataset Catalog ``dataset_type`` (``raw`` / ``processed``)
   * - Acquisition
     - ``microscope`` / ``objective`` / ``detector`` / ``imaging_environment`` / ``acquisition_date``
     - Optional
     - OME (Bio-Formats)
   * - Extra
     - ``additional_metadata`` — extra study-provided values not yet defined in this spec
     - Optional
     - DCA-native · agent-minted, each with inline provenance fields. Lives in the
       Parquet and can be **elevated** to a defined (and ultimately required,
       ``zarr.json``) field once it stabilizes — see
       :ref:`Promotion <v03-experimental-promotion>`.

The :ref:`provenance fields <v03-experimental-evidence>` are **REQUIRED for
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


Acquisition Metadata
--------------------

Image Acquisition is about how the images were captured. Its fields are optional
apart from ``time_increment_s``, which is **required when the data are time-lapse**,
and ``timestamps``, which is **required** when ``time_sampling`` is ``variable``.

``time_sampling`` states whether the timepoints are evenly spaced. Absent, it reads as
``uniform``, and the OME ``t``-axis scale describes the spacing. ``variable`` says the
scale does not, because the acquisition was interrupted or re-paced; the per-frame
``timestamps`` are authoritative instead, and ``time_increment_s`` carries the nominal
cadence the acquisition targeted. ``timestamps`` are seconds since the Unix epoch (UTC),
a single encoding so a consumer needs no companion unit field, and their length and
ordering are checked whenever they are present. Declaring ``variable`` relaxes the temporal-axis unit
MUST (see :doc:`ome-zarr-metadata`) and is checked against the array, so it cannot be
used to avoid stating a cadence that is in fact uniform.

A minimal, **optional** OME instrument set — ``microscope``, ``objective``,
``detector``, ``imaging_environment``, and ``acquisition_date`` — is defined in the
:download:`LinkML schema <../../schema/v0.3/dca_experimental_metadata.yaml>` (with OME
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
``ImageData.processing_variant``: the computational processing steps that produced this
array, as an ordered list in the order they were applied (e.g. ``["raw"]``,
``["deconvolved", "registered"]``, ``["deconvolved", "maximum intensity projection"]``).
Items **SHOULD** come from the ``ProcessingStep`` vocabulary in the LinkML schema
(``raw``, ``denoised``, ``deconvolved``, ``registered``, ``stitched``, ``cropped``,
``flat-field corrected``, ``background subtracted``, ``maximum intensity projection``,
``mean intensity projection``, ``sum intensity projection``, ``z slice``);
an item outside it is a warning. ``raw`` means no processing and **MUST** be the only
item when present. Qualifying detail (a pipeline version, a frame range) belongs in
``image_data.processing_description`` (new in v0.3, free text in the source's own
words), not in the step name. In v0.2 this field was a single string;
in v0.3 it **MUST** be a non-empty list. The full derivation (parent dataset, algorithm,
parameters) is a Scientific Dataset Catalog ``transformed_from`` Lineage Edge, not
duplicated here. The remaining image-level fields
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
  (NCBITaxon), ``tissue`` (UBERON / CL / ZFA) + ``tissue_type``, ``disease`` (MONDO /
  PATO), and ``development_stage`` — each a list of
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

   **Development stage follows the organism.** Use the stage ontology
   cross-modality names for that organism: `HsapDv
   <https://obofoundry.org/ontology/hsapdv>`_ (human), `MmusDv
   <https://obofoundry.org/ontology/mmusdv>`_ (mouse), `ZFS
   <https://obofoundry.org/ontology/zfs>`_ (zebrafish), `FBdv
   <https://obofoundry.org/ontology/fbdv>`_ (fly), `WBls
   <https://obofoundry.org/ontology/wbls>`_ (worm), and a UBERON life-cycle term for
   organisms with no dedicated stage ontology. Prefer the organism's own ontology over
   a UBERON term where one exists: only 13 of ZFS's 54 stage terms have a UBERON
   equivalent, and the two periods most whole-embryo zebrafish time-lapses cover
   (``ZFS:0000049`` segmentation, ``ZFS:0007000`` hatching) have none, so UBERON
   coarsens them to ``UBERON:0000068`` "embryo stage". Do not use an ontology's own
   "no value" term: ``ZFS:0000000`` "unknown" and ``FBdv:00007012`` "life stage" are
   rejected on ``development_stage``, as is ``ZFA:0001093`` "unspecified" on ``tissue``,
   so that ``unknown`` has one spelling (the sentinel).

   **Zebrafish anatomy versus cell type.** Cross-modality splits these across two
   fields; DCA folds both into ``tissue`` and keys the distinction off ``tissue_type``.
   In the anatomy role (``tissue_type`` ``tissue`` or ``organoid``) use a descendant of
   ``ZFA:0100000`` outside the ``ZFA:0009000`` "cell" branch; in the cell-type role
   (``tissue_type`` ``cell culture``) use a CL term, or a ``ZFA:0009000`` descendant
   where CL has no equivalent.

   **Unrecorded sample type (``tissue_type = "unknown"``).** Image corpora pooled from
   many studies without a per-image manifest (e.g. segmentation training sets) may not
   record what any given image depicts. Set ``tissue_type`` to ``unknown`` and the
   sample-context term ids to the ``unknown`` sentinel, rather than guessing a value or
   omitting the sample. ``unknown`` means the source does not record it, not that it
   has not been looked up yet.

See the field reference under :ref:`Conformance <v03-experimental-conformance>` for the
full Study / Sample field list with required-vs-optional status, and the
``Study``, ``Experimenter``, ``Biosample``, ``Specimen`` classes in the
:download:`LinkML schema <../../schema/v0.3/dca_experimental_metadata.yaml>` for the
authoritative field list.

Perturbation Metadata
---------------------

Perturbation metadata is modeled as a **modality-agnostic ``Perturbation``**
(mapped to the OME ``Reagent`` entity), specialized by a ``modality`` discriminator
into a **chemical** or **genetic** block. This generalizes beyond chemical
perturbation to **CRISPR / gene knockout / ORF** (and diet, protein, temperature,
physical and biological perturbations).
The genetic block carries the gene target that CellPainting Gallery cannot express
(its ``Treatment_Category`` enum names ``CRISPR``/``ORF`` but provides no gene
field). The fields are harmonized across two sources: OME/Bio-Formats ``Reagent``
and the CellPainting Gallery ``Treatment_*`` fields.

These fields apply to **any perturbation dataset**, not just CellPainting Gallery
ones; the ``cpg:`` entries are alignment mappings (provenance), not a
requirement that the data originate from those projects. A custom drug screen or an
in-house CRISPR experiment populates the **same** ``Perturbation``
/ ``PerturbationAssignment`` structure, with values sourced from wherever its metadata
lives (``source_kind`` records that, e.g. ``original_data`` from the source's own
manifest).

**Required on every image group.** Every image group **MUST** carry a
``perturbation`` block. When nothing was applied to its material, the block is
exactly ``{"modality": "no perturbations"}`` and **MUST NOT** carry any other field.
Otherwise ``control_class`` is **required**, and the definition's
``perturbation_id`` + ``modality`` are **required**. ``definition`` is required on
every **treated or control** assignment (``treatment`` / ``negcon`` / ``poscon``) but
is **omitted when** ``control_class`` **is** ``empty``: an untreated well that is part
of a screen's layout. The ``chemical`` vs ``genetic`` block is required according to
``modality``. All of these are enforced as schema rules (see Validation).

.. code-block:: json

   {"perturbation": {"modality": "no perturbations"}}

.. _v03-experimental-perturbation-placement:

**Where it is written.** Two things are modeled separately, because their grains
differ:

* The ``Perturbation`` *definition* (a codebook entry) is per-construct and reusable
  across the dataset. It is what the perturbagen IS.
* The ``PerturbationAssignment`` — that definition plus the dosing / control class /
  timing — is what was DONE to imaged material. It carries the definition in its
  ``definition`` slot, plus an optional ``cotreatment`` for a second definition
  applied alongside.

The assignment is the ``perturbation`` block of ``ExperimentalMetadata``, so it is
**mirrored into the image group's** ``dca`` **block like every other experimental
block** (``study`` / ``biosample`` / ``specimen`` / ``study_component``). A consumer
reads a perturbation from the image group it already has open, with no second lookup
and no HCS key to resolve — and a plateless ``image_collection`` store can express a
perturbation at all.

For an **arrayed HCS screen** the same assignment is **also** indexed per well as a
``WellPerturbation`` (a ``PerturbationAssignment`` plus the ``(plate, well)`` key),
written under the well group's ``dca`` block as the ``well_perturbation`` key —
i.e. ``dca.well_perturbation`` on the well group's ``zarr.json``, the well-level
counterpart to the image group's ``dca.perturbation``. The well level is derived from
the image group's perturbation information, so a consumer can enumerate a plate's
design without opening every field of view.

The image-group ``perturbation`` block is **authoritative**; the ``WellPerturbation``
row is **derived** from it. The two MUST agree — for every image group under a well,
its ``perturbation`` block is that well's assignment — and because the direction is
defined, a disagreement has one correct repair: re-derive the well row from the image
groups, never the reverse. Two reasons the image group holds the primary copy:

* It is the only placement **every** layout can carry. A plateless
  ``image_collection`` store has no well group to write to.
* It matches the grain of the fact for datasets where it can differ. Per-well is a
  property of arrayed screens specifically, not of perturbation in general.

Both placements are checked for their own internal validity, and the validator also
cross-checks one against the other: every field of view under a well must carry an
identical ``perturbation`` value, and — when the well also carries a
``well_perturbation`` row — that consensus must match the row (see :doc:`/v0.2/validator`).
Authoritativeness is what gives a mismatch one defined repair (re-derive the well row)
rather than an ambiguous merge.

Producers that hold the well assignment as their source data (an arrayed screen's
plate map, typically) should project it onto the image groups and derive the well row
from the same projection, so a single input produces both placements and they cannot
diverge.

See the ``Perturbation``, ``ChemicalPerturbation``, ``GeneticPerturbation``,
``PerturbationAssignment``, and ``WellPerturbation`` classes in the
:download:`LinkML schema <../../schema/v0.3/dca_experimental_metadata.yaml>`.

.. list-table:: Cross-source field mapping (minimal set)
   :header-rows: 1
   :widths: 34 33 33

   * - Concept
     - OME ``Reagent``
     - CellPainting Gallery
   * - External id
     - ``ReagentIdentifier``
     - ``Treatment_Broad_Sample``
   * - Modality
     - —
     - ``Treatment_Category``
   * - Control class
     - —
     - ``Treatment_Control_Class``
   * - Concentration
     - —
     - ``Treatment_Concentration`` (str)
   * - Compound ids
     - (``AnnotationRef``)
     - ``InChIKey`` / ``SMILES`` / ``PubChem_CID``
   * - **Gene target**
     - —
     - **absent**

The reconciled vocabularies (see the schema enums):

* ``modality`` (on the ``definition``) — see the table below.
* ``control_class`` — ``treatment`` | ``negcon`` | ``poscon`` | ``empty`` (CPG's
  redundant ``Control`` dropped; ``empty`` added for untreated wells).

.. _v03-experimental-perturbation-modality:

**Modality and CELLxGENE.** ``modality`` is aligned with the CELLxGENE schema 7.1
perturbation fields. CELLxGENE's ``perturbation_types`` and
``genetic_perturbation_strategy`` are **derived** from it, not curated separately, so
the two can never disagree:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - ``modality``
     - CxG ``perturbation_types``
     - CxG ``genetic_perturbation_strategy``
   * - ``compound``
     - ``chemical``
     - —
   * - ``CRISPR activation screen``
     - ``genetic``
     - ``CRISPR activation screen``
   * - ``CRISPR interference screen``
     - ``genetic``
     - ``CRISPR interference screen``
   * - ``CRISPR knockout mutant``
     - ``genetic``
     - ``CRISPR knockout mutant``
   * - ``CRISPR knockout screen``
     - ``genetic``
     - ``CRISPR knockout screen``
   * - ``orf`` / ``sirna`` / ``shrna`` / ``mirna``
     - ``genetic``
     - ``ORF`` / ``siRNA`` / ``shRNA`` / ``miRNA`` (DCA extensions)
   * - ``diet`` / ``protein`` / ``temperature``
     - ``diet`` / ``protein`` / ``temperature``
     - —
   * - ``physical`` / ``biological`` / ``not reported`` / ``other``
     - the same value (DCA extensions)
     - —
   * - the block's ``"no perturbations"``
     - ``no perturbations``
     - ``no perturbations``

* A control (``negcon`` / ``poscon``) maps to the strategy ``control``, and an
  ``empty`` well to ``no perturbations``.
* An assignment with a ``cotreatment`` takes the union of both definitions' types,
  joined as CELLxGENE does (``chemical || genetic``).
* ``temperature`` is a temperature *perturbation* (heat or cold shock), not the imaging
  temperature, which is ``acquisition.temperature_c``.
* ``not reported`` is for a perturbation of a kind the source does not state. There is
  no ``unknown``.
* CELLxGENE 7.1 itself lists only ``chemical`` / ``diet`` / ``genetic`` / ``protein`` /
  ``temperature`` as types and the four CRISPR strategies plus ``control``; the values
  marked as DCA extensions go beyond it.

**Migration from v0.2.** ``crispr`` and ``unknown`` have no automatic v0.3 value:
whether a CRISPR screen was activation, interference or knockout, and what an
``unknown`` modality was, come from the source, so those datasets are re-curated.
A v0.2 image group without a ``perturbation`` block gains
``{"modality": "no perturbations"}``.

Known gaps and contradictions:

* **Gene target gap** — CellPainting Gallery has no gene field; the genetic block adds one.
* **Concentration type** — CPG records free text; DCA uses
  ``{value, unit}``. The schema keeps structured ``value``/``unit`` with a
  free-text ``text`` fallback.
* **Gene identifier** — NCBI Gene ID is the canonical key; HGNC
  symbol is the label; Ensembl is optional cross-map.

.. _v03-experimental-deferred-pooled:

Deferred: pooled / optical pooled screens (OPS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PerturbationAssignment`` covers **arrayed** screens (one perturbation per image
group, and per well). In a **pooled / optical pooled screen (OPS)** a single well
contains a library of many sgRNAs and the perturbation is assigned **per cell** by
decoding an integrated barcode after imaging. That assignment is therefore **not**
expressible per image group, let alone per well. The
pooled-screen reagent fields (sgRNA protospacer, PAM, barcode, target locus) and the
per-cell assignment table are **deferred** to a future version, as is the broader move
from this spec's current **Parquet** tables to AnnData / SpatialData conventions for
single-cell data. The ``Perturbation`` *definition* above is unchanged for pooled
libraries; only the level at which a perturbation is assigned differs (per image
group for arrayed screens, per cell for pooled).

.. _v03-experimental-gap-fill:

Free-Text Gap-Fill
------------------

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
       "citation": {
         "source_url": "https://doi.org/10.1234/example",
         "quote": "hiPSCs were differentiated toward definitive endoderm over five days",
         "match_status": "exact"
       },
       "source_kind": "agent"
     }
   }

Each ``additional_metadata`` entry **MUST** carry a ``name`` and ``value`` and
**SHOULD** carry its :ref:`provenance fields <v03-experimental-evidence>`. An agent
**MAY** mint new ``additional_metadata`` fields; as with every agent-curated value,
each **MUST** carry its extracted value, its source citation, and the citation's
``match_status``.
As fields here stabilize and gain a community schema, they are promoted to
first-class structured slots in a future version. ``additional_metadata`` is a
**staging area**, not a permanent home.

.. _v03-experimental-promotion:

Promotion to the required floor
-------------------------------

The whole record is written to ``zarr.json`` and to the Parquet, but only the required
floor is *required*; every *optional* or *recommended* field rides along, validated
when present and never demanded. This keeps extracted, unstandardized, or experiment-specific fields
queryable as soon as they are curated, so consumers do not have to wait for a spec
revision before they are available. An optional or recommended field **MAY** be promoted
into the required floor in a future spec version, but only when **all** of the following
hold:

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
     for a defined class of experiments* (e.g. ``cell_line_*`` when ``tissue_type``
     is "cell line"). Such a field is promoted
     to ``zarr.json`` and made **conditionally required** on that class, rather than
     left optional — so consumers of that experiment type can rely on it. In
     v0.2 the ``perturbation`` block was the worked example; v0.3 makes it
     corpus-wide instead, with ``{"modality": "no perturbations"}`` where nothing
     was applied (see
     :ref:`Where it is written <v03-experimental-perturbation-placement>`).

Promotion is a **per-field specification decision made via a version bump**. For a
corpus-wide field it applies uniformly; for an experiment-class field it applies
to that class. When a field is promoted, the Parquet remains the **source of truth**
and the ``zarr.json`` copy is a derived, versioned snapshot.

.. _v03-experimental-evidence:

Provenance
----------

Because metadata can be extracted by an agent, each value carries its **provenance**
(``source_kind`` + grounding ``citation``). ``source_kind`` is
**always required** — it is the discriminator that tells a consumer whether a value was
``agent``-extracted, ``human``-entered, or taken from the source's own
``original_data``. The grounding ``citation`` is **required for ``agent`` values** and
**optional for ``human`` / ``original_data``** (left blank when there is nothing to
cite). This lets a consumer decide how much to trust a value without re-reading the
source.

Provenance is the ``ExperimentalMetadata.provenance`` **map**, keyed by each value's
dotted ``field_path`` (e.g. ``biosample.organism``). The values stay plain and
provenance travels as a parallel layer. Free-text ``additional_metadata`` values are
keyed the same way, by ``<block>.additional_metadata.<name>``, so all provenance lives
in the one map. The Parquet (``tables/obs``) carries the same record, one row per image
group, with the block held as an embedded JSON object rather than expanded into
per-field columns.

``match_status`` is **computed deterministically by the agent's grounding gate, not
self-reported by the model**: the gate checks whether the cited ``quote`` is an (exact
or fuzzy) substring of the fetched source page and records the result as ``exact`` /
``fuzzy`` / ``none``. It runs at curation time against the raw source pages, so
``match_status`` carries that result and the grounding stays **auditable downstream**,
where the raw corpus is no longer available.

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
   * - ``source_kind``
     - enum
     - ``agent`` (provenance required) | ``human`` (manually entered; provenance
       optional) | ``original_data`` (from the source's own structured metadata;
       provenance optional).

Validation
----------

The :download:`LinkML schema <../../schema/v0.3/dca_experimental_metadata.yaml>` is the
authoring layer and single source of truth for all three groups; from it we generate
**Pydantic models** (enforced by the :doc:`validator </v0.2/validator>`) and a committed
**JSON Schema** (interop artifact and test-suite oracle cross-check).

Authoring in LinkML (rather than hand-maintaining JSON Schema and Pydantic) also gives
machine-readable cross-spec alignment: each slot records its ``exact_mappings`` to
Bio-Formats/OME, CellPainting Gallery, CZI cross-modality, schema.org, the
Scientific Dataset Catalog, and external ontologies. The same ``Study`` / ``Biosample``
classes can therefore be validated against, or projected to, other specs that publish a
LinkML schema (e.g. OPS) without a second validator.

What is and is not machine-checked
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ontology-term slots are checked by CURIE **prefix pattern**, not by resolving the term.
What the pattern cannot express falls into two groups: comparisons between fields already
in hand, which the validator makes, and questions about the ontology itself, which it
cannot answer offline.

**Checked.** These are field comparisons, enforced in the v0.2 validator
rather than the schema, since a LinkML rule is a per-slot predicate.

* **The stage ontology should match ``organism``.** The slot pattern is a flat union, so
  ``HsapDv`` on a zebrafish biosample satisfies it; the validator pairs the two slots and
  warns on a mismatch. It covers the five organisms cross-modality names a stage ontology
  for, and always accepts ``UBERON`` and the sentinels. It is skipped for any other
  organism, which has no stage ontology to compare against, and for a biosample listing
  several organisms, since ``organism`` and ``development_stage`` are independent lists
  with no positional correspondence.
* **A sentinel id constrains its ``label``.** ``unknown`` and ``na`` repeat themselves in
  the label; ``unavailable`` carries the free-text value there instead. The validator
  warns on both. This matters because ``label`` is the field consumers display, so a
  real-sounding label beside an ``unknown`` id shows a value the record does not have.
* **``tissue`` should carry a cell line's Cellosaurus term, and ``cell_line_id`` should
  agree with it.** ``tissue`` is authoritative and ``cell_line_id`` mirrors it. The
  validator warns when ``tissue`` holds no ``CVCL:`` accession (usually a UBERON or CL
  term instead), and separately when the mirror disagrees.

**Not checked offline.** Whether a term exists, and where it sits in the hierarchy, needs
a live ontology service. Cross-modality writes its per-organism rules as subtree
restrictions ("the most accurate descendant of ``ZFS:0100000``"), and no regex or field
comparison can evaluate descent, so offline a well-formed CURIE naming no real term, or a
real term outside the intended subtree, passes. The three terms cross-modality excludes
**by name** (``ZFS:0000000``, ``FBdv:00007012``, ``ZFA:0001093``) are rejected offline,
since a named exclusion is expressible where a subtree is not; the ``ZFA:0009000`` "cell"
branch stays prose-only.

``dca-validate --check-ontology`` closes both gaps. It resolves every
``ontology_term_id`` against the registry its prefix belongs to (EBI OLS4 for the
OBO-library and EFO ontologies, the Cellosaurus API for ``CVCL``) and, for the two FBbi
branch slots ``imaging_method`` and ``preparation_method``, checks descent from the
required ancestor. An unresolvable id, or a label matching none of the registry's names,
is an error; an unreachable registry is a warning, so a flaky network degrades to
"unverified" rather than failing the store. It is off by default because it makes network
requests; see :doc:`/v0.2/validator` for the full behavior.
