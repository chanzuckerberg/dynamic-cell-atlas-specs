Version 0.1 Specification
==========================

Last updated: |last_updated_v01|

Contents
--------

.. toctree::
   :maxdepth: 2

   array-standard
   metadata-standard
   validator

Overview
--------

The Dynamic Cell Atlas (DCA) data resource provides 5D fluorescence microscopy datasets in a standardized format that makes them interchangeable for further processing.

The DCA standard aims to be optimized for handling 5D datasets (3D+time+channels). It also supports 4D data (2D+time+channel, 3D+channel), but will not be optimized for it.

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
