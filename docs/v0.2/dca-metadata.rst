DCA Metadata
============

DCA metadata lives in the ``zarr.json`` of the **multiscales image group** (i.e. the same level as ``ome.multiscales``), under the ``dca`` key,
enabling varying channel composition at arbitrary levels of the OME-Zarr hierarchy.
In an OME-Zarr HCS layout this is the field-of-view level (e.g. ``plate.ome.zarr/A/3/0/zarr.json``).

The ``dca`` object **MUST** contain:

* ``version`` — a string specifying the version of this specification (currently ``"0.2"``)
* ``url`` — a URL pointing to the specification in the GitHub repository
* ``channels`` — an array of channel metadata objects (see :doc:`channel-metadata`)
* ``normalization_statistics`` — per-channel normalization statistics keyed by channel index (see :doc:`normalization-statistics`)

.. code-block:: json

   {
     "ome": {
       "multiscales": [ "..." ]
     },
     "dca": {
       "version": "...",
       "url": "...",
       "channels": [ "..." ],
       "normalization_statistics": { "..." }
     }
   }

See the :download:`JSON Schema <schema.json>` for the authoritative definition.

Experimental metadata (acquisition, study/sample, perturbation) is **not** stored
in this ``dca`` block — it is agent-curated into a sibling Parquet (see
:doc:`experimental-metadata`).

.. toctree::
   :maxdepth: 2

   channel-metadata
   normalization-statistics
   experimental-metadata
   experimental-metadata-example
