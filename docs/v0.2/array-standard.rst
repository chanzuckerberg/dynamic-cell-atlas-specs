Array Standard
==============

There are a **multitude of use cases to cover**, for example:

* writing full arrays (data ingest)
* reading full arrays (data processing such as denoising, AI inference)
* reading partial arrays (visualization)
* reading with (pseudo-)random access (training AI models)
* copying full arrays (data management, backup)

While each of the above use cases might benefit from further optimization of e.g. chunk size, Zarr v3 chunking + sharding allows us to aim for a single array stored on disk that covers all of the above use cases reasonably well.

Format Requirements
-------------------

* **Format:** All datasets **MUST** conform to the `Zarr v3 <https://zarr.readthedocs.io/en/stable/user-guide/v3_migration.html>`_ and `OME-ngff 0.5 <https://ngff.openmicroscopy.org/0.5/>`_ standards.

Multiresolution
---------------

* An array **MUST** include a scale/level-of-detail (LOD) where each spatial dimension is smaller than or equal to 2048.
  If downsampling is performed, it **MUST** use the following downsampling methods (further defined in `acquire-zarr <https://github.com/acquire-project/acquire-zarr/blob/28dbd0b37aa833abfeafa6adc94f606c16437334/python/acquire_zarr/__init__.pyi#L261>`_):

  * Mean filter for raw image arrays
  * Maximum filter or decimation filter for label arrays

* Downsampling factors **SHOULD** be 2 for all spatial dimensions, and **MUST** be 1 for time and channel dimensions.

Additional Array Types
----------------------

Some datasets may include additional processed data for ingestion, which includes segmentation files, illumination correction files, etc., or may even fall under a particular high-content screen (HCS) format.

Labels
~~~~~~

Following `OME-ngff 0.5 image layout specification <https://ngff.openmicroscopy.org/0.5/#image-layout>`_, labels **SHOULD** be added within the group level of the Zarr store.

High Content Screening Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data **SHOULD** follow the OME-ngff 0.5 standards for `plate <https://ngff.openmicroscopy.org/0.5/#plate-md>`_ and `well <https://ngff.openmicroscopy.org/0.5/#well-md>`_.

Dimensions
----------

All datasets **MUST** be in 5 dimensions (with placeholder values) in the following order: ``time, channels, z, y, x``:

* ``z`` corresponds to the 3rd spatial dimension
* ``y``, ``x`` correspond to the imaging plane

.. note::

   Axis names MUST be all lowercase (``t``, ``c``, ``z``, ``y``, ``x``) or all uppercase (``T``, ``C``, ``Z``, ``Y``, ``X``). Mixed casing such as ``T``, ``C``, ``Z``, ``y``, ``x`` is rejected. The original casing is preserved.

Data Types
-----------

Raw Image Data
~~~~~~~~~~~~~~~

Raw image data **SHOULD** be quantized and stored as ``uint8`` or ``uint16`` types to make lossless compression effective.

Label Image Data
~~~~~~~~~~~~~~~~~

Label image data **SHOULD** be stored as ``uint32``.

Performance
-----------

We aim to follow the `Zarr's Guidelines for Optimizing Performance (ZGOP) <https://zarr.readthedocs.io/en/stable/user-guide/performance.html>`_.

Chunk Size
~~~~~~~~~~

*Context: Chunks are the basic read unit. They are lossless compressed on disk, and a whole chunk needs to be decompressed to access any of its values. Fluorescence microscopy data compression rates are limited due to noise, they typically reach factors of 2 or 3 only.*

Requirements:

* Considering ZGOP, **uncompressed chunk size** **SHOULD** be at least 1MB, and **MUST** be >= 512KB (based on a uniform ``uint8`` datatype). If the image is smaller than 512KB, the chunk size can be equal to the image size.
* The **time and the channel** dimension chunk size **SHOULD** be 1.
* The **spatial dimensions** (z, y, x) chunk size **SHOULD** be 128 x 128 x 128. For example, this results in 2MB uncompressed chunk size for ``uint8`` dtype.

Edge Cases
^^^^^^^^^^^

It might be that the array shape in z is significantly smaller than 128 due to high anisotropy. *If 512KB chunk size is not reached*:

1. Increase the chunk size in time (by multiples of 2) until it reaches 512KB.
2. If 1) is not possible, increase the imaging plane (y,x) chunk sizes by factors of 2 until it reaches 512KB.

The 512KB **MUST** and 1MB **SHOULD** chunk size requirements apply only to the base resolution level (index 0 in the ``multiscales.datasets`` array). Downsampled levels **MAY** use any chunk size.

Typical Example Chunk Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``1, 1, 128, 128, 128`` (3D+time light sheet data)
* ``1, 1, 32, 128, 128`` (3D+time confocal data, with anisotropy and limited depth in z)
* ``16, 1, 1, 128, 128`` (2D+time data)

Shard Size
~~~~~~~~~~

*Context: Shards group multiple compressed chunks into a single file, with an index to the chunk positions. Shards are the basic write unit. They potentially allow grouped chunk reads, and allow to store very large datasets while keeping the number of files on disk low, which is a requirement for many file systems.*

High Priority Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Considering the example in ZGOP, **uncompressed shard size**:

* **MUST** be
   * at least 1GB, if the array is equal to or larger than 1GB
* **MUST** be smaller than 5TB
* **SHOULD** be smaller than 5GB

Low Priority Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^

* The **spatial dimensions** shard size **SHOULD** be equal to the shape of the entire array, but **SHOULD** also be <= 2048.
* The **time dimension** shard size **SHOULD** be >= 16.

Typical Example Shard Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``16, 1, 128, 512, 2048`` (3D+time data)
* ``16, 1, 1, 512, 512`` (2D+time data)
* ``1, 1, 42, 1248, 1824`` (3D+time data)
* ``16, 2, 1, 3798, 3798`` (2D+time data)

Compression
-----------

* All data **MUST** be compressed using a `zstd`, `LZ4` or `Blosc` codec.
* Data **MAY** use the `numcodecs' Shuffle <https://numcodecs.readthedocs.io/en/v0.15.1/filter/shuffle.html#numcodecs.shuffle.Shuffle>`_ codec in addition to the compression codec.
* All data **SHOULD** be compressed using:

   * For **Integers**:
      * ZSTD codec (no BLOSC)
      * Compression level 3 or lower (to keep decompression overhead low)
   * For **Floats** (Edge case):
      * First, byte shuffle
      * Then, use ZSTD codec (no BLOSC), compression level 3 or lower (to keep decompression overhead low)

.. note::

   This is based on benchmarking results documented `here <https://github.com/MaadhavKothari/zarr-benchmarks-cryoet/blob/main/CHUNKING_SHARDING_REPORT.md>`_.
