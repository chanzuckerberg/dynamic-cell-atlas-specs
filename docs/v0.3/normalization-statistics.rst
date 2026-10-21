Normalization Statistics Specification
======================================

Overview
--------

This specification defines lightweight per-channel intensity normalization statistics for multichannel light microscopy datasets stored in OME-NGFF v0.5 format.

Motivation
----------

Microscopy datasets often contain multiple channels with different intensity distributions, which requires intensity normalization for downstream analysis and visualization.
For AI model training on such datasets, samples are commonly smaller random crops from a larger image.
However, the intensity distribution of a random crop often does not match the overall intensity distribution of the full image.
By pre-computing and storing summary statistics at both dataset and per-timepoint granularity, consumers can apply different types of data normalization efficiently without reading full images.

Location in Zarr Store
----------------------

Normalization statistics are stored as part of the ``dca`` object directly within the ``zarr.json`` of the multiscales image group (i.e. the same ``zarr.json`` that contains ``ome.multiscales``).
The ``normalization_statistics`` field contains channel normalization objects with zero-based channel indices (as a string), and **OPTIONAL** metadata.


Example
-------

.. code-block:: json

   {
     "ome": {
       "multiscales": [ "..." ]
     },
     "dca": {
       "version": "...",
       "url": "...",
       "channels": [ "..." ],
       "normalization_statistics": {
         "0": {
           "dataset_statistics": {
             "p1": 42.0,
             "p5": 62.5,
             "p95": 192.0,
             "p99": 218.5,
             "p95_p5": 129.5,
             "p99_p1": 176.5,
             "mean": 126.45,
             "std": 38.12,
             "median": 124.0,
             "iqr": 51.3
           },
           "timepoint_statistics": {
             "0": {
               "p1": 43.0,
               "p5": 63.1,
               "p95": 193.2,
               "p99": 219.0
             },
             "1": {
               "p1": 41.5,
               "p5": 61.8,
               "p95": 190.7,
               "p99": 216.3
             }
           }
         },
         "1": {
           "dataset_statistics": {
             "p1": 10.2,
             "p5": 18.7,
             "p95": 245.1,
             "p99": 251.8
           }
         },
         "metadata": {
           "grid": 16
         }
       }
     }
   }

Channel Normalization Fields
----------------------------

See the :download:`JSON Schema <schema.json>` (``#/$defs/ChannelNormalization``) for the authoritative definition.

Each channel normalization object **MUST** contain:

* ``dataset_statistics`` — intensity statistics computed over all spatial dimensions and timepoints for that channel.

Each channel normalization object **MAY** contain:

* ``timepoint_statistics`` — an object mapping zero-based timepoint indices (as strings) to per-timepoint intensity statistics.
  If present, it **MUST** contain entries for all timepoints in the dataset.

Intensity Statistics Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Required Fields
^^^^^^^^^^^^^^^

Both ``dataset_statistics`` and each value in ``timepoint_statistics`` **MUST** contain the following fields.

.. list-table::
   :header-rows: 1
   :widths: 15 10 75

   * - Field
     - Type
     - Description
   * - ``p1``
     - number
     - 1st percentile of pixel intensities.
   * - ``p5``
     - number
     - 5th percentile of pixel intensities.
   * - ``p95``
     - number
     - 95th percentile of pixel intensities.
   * - ``p99``
     - number
     - 99th percentile of pixel intensities.

Optional Fields
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 10 75

   * - Field
     - Type
     - Description
   * - ``p95_p5``
     - number
     - Robust range: 95th percentile minus 5th percentile.
   * - ``p99_p1``
     - number
     - Wide robust range: 99th percentile minus 1st percentile.
   * - ``mean``
     - number
     - Arithmetic mean of pixel intensities.
   * - ``std``
     - number
     - Standard deviation of pixel intensities.
   * - ``median``
     - number
     - Median (50th percentile) of pixel intensities.
   * - ``iqr``
     - number
     - Interquartile range (75th percentile minus 25th percentile).

Normalization Metadata
----------------------

The ``normalization_statistics`` object itself **MAY** contain:

* ``metadata`` — an optional object for recording parameters used when computing normalization statistics.
  Useful for capturing implementation details that may affect reproducibility, for example downsampling factor before computing statistics.
