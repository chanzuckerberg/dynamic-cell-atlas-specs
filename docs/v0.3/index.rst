Version 0.3 Specification
=========================

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

* **Projections** (new): layout and metadata for 2D projections of 3D images
  (maximum, mean or sum intensity, or a single plane), with a ``dca.projection`` block
  that records the method and the planes actually used, uniformly or per channel. The
  model is adapted from biahub's ``project`` command. Its uniform form keeps
  the field names of the top-level ``mip`` attribute that v0.2-era projection stores
  carry. See :doc:`projections`.
* ``dca.version`` is ``"0.3"``.
* ``image_data.processing_variant`` is an ordered list of processing steps, with a
  recommended step vocabulary, instead of a single string, and a new free-text
  ``image_data.processing_description`` holds the full processing text. These are the
  only changes to the experimental-metadata schema
  (``schema/v0.3/dca_experimental_metadata.yaml``).
* The validator supports v0.3 with ``dca-validate --spec-version 0.3``, including the
  cross-store projection checks and an opt-in ``--require-projection`` (see the
  :doc:`v0.2 validator page </v0.2/validator>` for everything unchanged). The default
  spec version stays 0.2 while v0.3 is developed further, so pass
  ``--spec-version 0.3`` to validate against v0.3.

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
