.. _experimental-timelapse-example:

Experimental Metadata Example
=============================

A worked example accompanying the :doc:`Experimental Metadata Specification
<experimental-metadata>`.

``zarr.json`` ``dca`` block
---------------------------

The following shows the ``dca`` block in the ``zarr.json`` of one deconvolved
store from an example live 3D time-lapse dataset (cultured human iPSCs, live confocal
fluorescence). It carries the **required experimental floor** mirrored alongside
the channel and normalization-statistics metadata the block already holds. Because this
is a time-lapse dataset, the conditionally-required ``acquisition.time_increment_s`` is
included. The complete curated record, including the optional and recommended fields not
shown here, lives in the sibling Parquet and is :ref:`described below <example-parquet>`.

The experimental blocks sit **directly** in ``dca`` (no wrapper key), next to
``version`` / ``url`` / ``channels`` / ``normalization_statistics``. Per-value provenance
is the ``provenance`` map keyed by each value's dotted ``field_path`` (so the values stay
clean and provenance travels as a parallel layer). Every value records at least its
``source_kind``; here all ``human``, since the dataset was hand-curated, whereas an
``agent``-extracted value would additionally carry a grounding ``citation``. The
``channels`` and ``normalization_statistics`` shown are defined by the
:doc:`channel-metadata` and :doc:`normalization-statistics` specs and their values here
are illustrative.

.. code-block:: json

   {
     "attributes": {
       "dca": {
         "version": "0.2",
         "url": "s3://my-bucket/datasets/example/002_deconv.ome.zarr",
         "channels": [
           {"name": "H2B-mCherry", "index": 0, "description": "nuclei (H2B-mCherry reporter)"},
           {"name": "membrane-GFP", "index": 1, "description": "cell membrane (membrane-GFP reporter)"}
         ],
         "normalization_statistics": {
           "0": {"dataset_statistics": {"p1": 104, "p5": 112, "p95": 932, "p99": 1187}},
           "1": {"dataset_statistics": {"p1": 99, "p5": 108, "p95": 845, "p99": 1062}}
         },
         "study": {
           "source": "Example Imaging Lab",
           "study_type": ["FourDPlus", "TimeLapse", "FP"],
           "description": "Long-term 3D+t live fluorescence imaging of cultured human iPSCs expressing nuclear and membrane reporters, following cell growth and division.",
           "year_imaged": "2023",
           "dataset_name": "example",
           "release_date": "2023-05-01",
           "license": "CC-BY-4.0",
           "experimenter": [
             {"first_name": "Jane", "last_name": "Doe", "institution": "Example Imaging Lab"},
             {"first_name": "John", "last_name": "Roe", "institution": "Example Imaging Lab"}
           ]
         },
         "biosample": {
           "group_id": "example__002_deconv",
           "tissue_type": "cell culture",
           "organism": [{"ontology_term_id": "NCBITaxon:9606", "label": "Homo sapiens"}],
           "tissue": [{"ontology_term_id": "CL:0002620", "label": "induced pluripotent stem cell"}],
           "disease": [{"ontology_term_id": "PATO:0000461", "label": "normal"}],
           "development_stage": [{"ontology_term_id": "na", "label": "na"}]
         },
         "specimen": {
           "specimen_state": "live",
           "preparation_method": [{"ontology_term_id": "FBbi:00100027", "label": "adherent live cells"}]
         },
         "study_component": {
           "imaging_method": [{"ontology_term_id": "FBbi:00000251", "label": "confocal microscopy"}],
           "assay": [{"ontology_term_id": "EFO:0002909", "label": "microscopy assay"}]
         },
         "acquisition": {"time_increment_s": 600.0},
         "provenance": {
           "study.source": {"source_kind": "human"},
           "study.study_type": {"source_kind": "human"},
           "study.description": {"source_kind": "human"},
           "study.year_imaged": {"source_kind": "human"},
           "study.dataset_name": {"source_kind": "human"},
           "study.release_date": {"source_kind": "human"},
           "study.license": {"source_kind": "human"},
           "study.experimenter": {"source_kind": "human"},
           "biosample.group_id": {"source_kind": "human"},
           "biosample.tissue_type": {"source_kind": "human"},
           "biosample.organism": {"source_kind": "human"},
           "biosample.tissue": {"source_kind": "human"},
           "biosample.disease": {"source_kind": "human"},
           "biosample.development_stage": {"source_kind": "human"},
           "specimen.specimen_state": {"source_kind": "human"},
           "specimen.preparation_method": {"source_kind": "human"},
           "study_component.imaging_method": {"source_kind": "human"},
           "study_component.assay": {"source_kind": "human"},
           "acquisition.time_increment_s": {"source_kind": "human"}
         }
       }
     }
   }

.. _example-parquet:

Additional metadata in the Parquet
----------------------------------

The ``dca`` block above carries only the required floor. The sibling Parquet is the
source of truth: it holds the **complete** record, including the recommended and optional
fields below. Each also carries its provenance,
keyed the same way by ``field_path``; free-text ``additional_metadata`` values use
``<block>.additional_metadata.<name>``, so all provenance has one home.

**Recommended study fields.** ``accession_id`` (``"S-BIAD0000"``, an example BioImage
Archive data deposit) and the ``related_publication`` entry (Doe et al. 2023, ``doi``
``"10.0000/example.0000"``, the paper). The data deposit and the paper are distinct:
the deposit identifier goes in ``accession_id``, the paper DOI in ``related_publication``.

**Instrument and acquisition details (optional).** The OME instrument set beyond the
conditionally-required ``time_increment_s``: ``microscope`` (an inverted spinning-disk
confocal microscope), ``objective`` (63x, oil immersion), and ``imaging_environment``
(37 °C, 5% CO₂).

**Processing variant.** Imaging experiments often contain multiple data pre-processing
steps to optimize segmentation, tracking, or other downstream outputs. This dataset
contains raw, denoised, and deconvolved versions of the same Zarr stores; the store here
is the deconvolved one, so its Parquet records ``ImageData.processing_variant`` =
``"deconvolved"``. The full derivation relationship is tracked in the Scientific Dataset
Catalog as a Lineage Edge (``transformed_from``); ``processing_variant`` is the
zarr-local label only.

**Specimen growth conditions.** The culture medium and substrate go in the existing optional
``Specimen.growth_conditions`` field, not ``additional_metadata``: ``"mTeSR Plus medium on
Matrigel-coated plates"``.

**Free-text gap-fill (``additional_metadata``).** Cell-line details with no first-class
field go here, named to mirror the cross-modality cell-line guidance so they map cleanly
if promoted later:

* **Passage number** (``passage_number`` = ``"p32"``) — the culture passage at imaging
  time.
* **Reprogramming method** (``reprogramming_method`` = ``"episomal"``) — how the iPSC line
  was derived. Donor age, reprogramming method, and passage number have no first-class
  field for cultured lines and live here per the cell-line guidance above.
