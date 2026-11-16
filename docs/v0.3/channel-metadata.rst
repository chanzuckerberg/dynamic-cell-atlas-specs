Channel Metadata Specification
===============================

Overview
--------

This specification defines additional per-channel metadata for multichannel light microscopy datasets stored in OME-NGFF v0.5 format.

Scope
-----

This part of the specification covers channels of **image arrays**. Other types of arrays, such as integer label arrays and learned feature maps, are out of scope.

Motivation
----------

Most multi-channel imaging pipelines assume a fixed, ordered set of channels and match channels across samples by indexing. This breaks down for a heterogeneous data corpus.
By attaching a mix free-text metadata and optional structured fields for common channel metadata to each channel, models can condition on channel semantics via large language model (LLM) embeddings, while keeping the specification minimally prescriptive.
Where we have adapted and extended standards for the Dynamic Cell Atlas, we intend to contribute improvements back to the broader community.

Location in Zarr Store
----------------------

Channel metadata is stored in the ``dca.channels`` array within the :doc:`DCA metadata object <dca-metadata>`.

The ``channels`` array **SHOULD** include metadata for each channel in the image.

.. code-block:: json

   {
     "ome": {
       "multiscales": [ "..." ]
     },
     "dca": {
       "version": "...",
       "url": "...",
       "channels": [
         {
           "name": "Phase",
           "index": 0,
           "description": "Quantitative phase imaging, label-free"
         },
         {
           "name": "H2B-GFP",
           "index": 2,
           "description": "Histone-Green Fluorescent Protein fusion protein (H2B-GFP)"
         }
       ],
       "normalization_statistics": { "..." }
     }
   }

Channel Fields
--------------

See the :download:`JSON Schema <schema.json>` (``#/$defs/ChannelMetadata``) for the authoritative definition.

Required Fields
~~~~~~~~~~~~~~~

Each stored channel metadata object **MUST** contain the following fields.

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Field
     - Type
     - Description
   * - ``name``
     - string
     - **SHOULD** be a short name (e.g. ``"H2B-GFP"``, ``"Phase2D"``) capturing the acquisition and/or biological context. **SHOULD** match a potential OME-NGFF ``omero.channels`` label.
   * - ``index``
     - integer
     - Zero-based channel index in the C axis.
   * - ``description``
     - string
     - **SHOULD** be a rich natural language description suitable for text embedding. (See `Guidance on Channel Description`_.)


Optional Fields
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Field
     - Type
     - Description
   * - ``channel_type``
     - string (enum)
     - See `Guidance on Channel Type`_ below.
   * - ``biological_annotation``
     - object
     - Biological target details — **SHOULD** be provided for ``fluorescence`` and ``predicted`` channels.

Guidance on Channel Metadata Vocabulary
---------------------------------------

Guidance on Channel Description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``description`` field **SHOULD** contain a rich description of the channel, containing a subset of the metadata fields outlined below.
It **MAY** use terminology from these different ontologies:

* CellPainting Gallery harmonized ontology (see `harmonized_ontology.json <https://github.com/broadinstitute/cellpainting-gallery-metadata/blob/main/harmonized_ontology.json>`_, BSD-3-Clause)
* `HGNC gene symbols <https://www.genenames.org/>`_ (mapped from Ensembl `ENSG identifiers <https://www.ensembl.org/info/genome/stable_ids/index.html>`_)
* `UniProt protein names <https://www.uniprot.org/>`_
* `Gene Ontology Terms for function/activity <https://geneontology.org/>`_

Examples:

* **Fluorescent:** ``"{biological_target} visualized via {marker}"``
* **Label-free:** ``"{reconstruction_type} label-free {modality} imaging"``
* **Virtual stain:** ``"{biological_target} predicted by virtual staining"``


Guidance on Channel Type
~~~~~~~~~~~~~~~~~~~~~~~~

The ``channel_type`` field **SHOULD** be populated with one of the following values.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Meaning
   * - ``fluorescence``
     - Fluorescent label microscopy (e.g. confocal, widefield fluorescence)
   * - ``chromogenic``
     - Chromogenic staining-based imaging (e.g. H&E, IHC, histological stains)
   * - ``labelfree``
     - Label-free modality (brightfield, phase, DIC)
   * - ``predicted``
     - Computationally predicted from other channels in the same dataset (*virtual staining*).

Biological Annotation Entries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``biological_annotation`` group **SHOULD** contain categorical metadata describing the biological target of the channel. The following string type fields are **OPTIONAL**.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``biological_target``
     - Target organelle, structure, or protein class visualized (e.g. ``"chaperones"``, ``"actin filament"``, ``"nuclei"``)
   * - ``marker_type``
     - See `Guidance on Marker Type Vocabulary`_ below. **SHOULD** be provided for fluorescent channels.
   * - ``marker``
     - Specific reagent, protein, or dye used (e.g. ``"HSPA1B"``, ``"FastAct_SPY555 Live Cell Dye"``, ``"virtual stain"``)
   * - ``cpg_labeled_structure``
     - The cellular compartment or structure made visible. Biology-grounded alignment across assay types. **SHOULD** match the ``Label_Structure`` field from the CellPainting Gallery harmonized ontology (see `harmonized_ontology.json <https://github.com/broadinstitute/cellpainting-gallery-metadata/blob/main/harmonized_ontology.json>`_).
   * - ``cpg_labeled_molecule``
     - The specific molecule the reagent binds. Precise molecular alignment. **SHOULD** match the ``Label_Molecule`` field from the CellPainting Gallery harmonized ontology (see `harmonized_ontology.json <https://github.com/broadinstitute/cellpainting-gallery-metadata/blob/main/harmonized_ontology.json>`_).

Guidance on Marker Type Vocabulary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``marker_type`` fields **SHOULD** be populated with one of the following values.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``endogenous_tag``
     - Genetically encoded fluorescent protein (e.g. GFP, mCherry knock-in)
   * - ``live_cell_dye``
     - Cell-permeable dye applied to living cells (e.g. SPY555-FastAct, MitoTracker)
   * - ``fixed_dye``
     - Dye applied after fixation (e.g. phalloidin, DAPI)
   * - ``antibody``
     - Fluorescently conjugated antibody (immunofluorescence)
