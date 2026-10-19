🚧 Version 0.3 Specification - Draft
=====================================

Last updated: |last_updated_v03|

Overview
--------

The Dynamic Cell Atlas (DCA) data resource provides 5D fluorescence microscopy datasets in a standardized format that makes them interchangeable for further processing.

The DCA standard aims to be optimized for handling 5D datasets (3D + time + channels).
It also supports lower-dimensional data (2D + channels, 2D + time + channels, 3D + channels), but will not be optimized for it.

Use Cases
---------

The standard is designed to support multiple use cases:

* Writing full arrays (data ingest)
* Reading full arrays (data processing such as denoising, AI inference)
* Reading partial arrays (visualization)
* Reading with (pseudo-)random access (training AI models)
* Copying full arrays (data management, backup)

While each use case might benefit from further optimization (e.g. chunk size), Zarr v3 chunking + sharding allows us to aim for a single array stored on disk that covers all use cases reasonably well.

Conformance
-----------

Throughout this document, conformance requirements are expressed with descriptive assertions using :rfc:`2119` terminology:

* **MUST** / **REQUIRED** / **SHALL**: Absolute requirement
* **MUST NOT** / **SHALL NOT**: Absolute prohibition
* **SHOULD** / **RECOMMENDED**: Strong recommendation
* **SHOULD NOT** / **NOT RECOMMENDED**: Strong discouragement
* **MAY** / **OPTIONAL**: Truly optional

Changes from v0.2
-----------------

* **Projections** (new): layout and metadata for maximum intensity projections,
  including a ``dca.projection`` block for the projected range. Its fields keep the
  names of the top-level ``mip`` attribute that v0.2-era projection stores carry.
  See :doc:`projections`.
* ``dca.version`` is ``"0.3"``.
* The experimental-metadata schema is unchanged from v0.2.
* No v0.3 validator exists yet. Use the :doc:`v0.2 validator </v0.2/validator>` for the
  parts that are unchanged.

Contents
--------

.. toctree::
   :maxdepth: 2

   array-standard
   projections

.. toctree::
   :maxdepth: 3

   metadata-standard

.. ingestion

Previous Versions
-----------------

* :doc:`Version 0.2 </v0.2/index>`
* :doc:`Version 0.1 </v0.1/index>`
