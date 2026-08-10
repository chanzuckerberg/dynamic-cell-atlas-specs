Validator
=========

The DCA validator checks whether a Zarr dataset conforms to the DCA v0.1 specification.
Given a local path or S3 URL, it validates each store against OME-NGFF v0.5 structure
and then applies DCA-specific MUST and SHOULD requirements.

Install
-------

Clone the repository and install from source:

.. code-block:: bash

   git clone https://github.com/chanzuckerberg/dynamic-cell-atlas-specs
   cd dynamic-cell-atlas-specs
   uv pip install -e .

CLI
---

.. code-block:: text

   dca-validate PATH [OPTIONS]

   Arguments:
     PATH  Local filesystem path or s3://bucket/prefix URL.

   Options:
     --spec-version TEXT     DCA spec version to validate against [default: 0.1]
     --output [text|json]    Output format [default: text]
     --strict                Exit 1 if any SHOULD warnings are present
     --help                  Show this message and exit.

**Examples**

.. code-block:: bash

   # Validate a single store
   dca-validate path/to/image.zarr

   # Validate all stores under a directory
   dca-validate path/to/experiment/

   # Validate an S3 store and emit JSON
   dca-validate s3://my-bucket/data/plate.ome.zarr --output json

   # Fail on SHOULD warnings as well as MUST errors
   dca-validate path/to/image.zarr --strict

**Exit codes**

* ``0`` — all stores pass (SHOULD warnings permitted unless ``--strict``)
* ``1`` — one or more stores fail, or ``--strict`` and warnings are present

Python API
----------

.. code-block:: python

   from dca_helpers.validation import validate
   from dca_helpers.validation.result import ValidationSummary

   results = validate("path/to/experiment/")

   summary = ValidationSummary.from_results(results)
   print(f"{summary.stores_passed}/{summary.stores_validated} passed")

   for r in results:
       status = "PASS" if r.passed else "FAIL"
       print(f"{status} {r.node_path}")
       for issue in r.issues:
           print(f"  [{issue.severity.name}] {issue.message}")

``validate`` accepts any path in the hierarchy — a single store, a subdirectory, or an
ancestor containing multiple stores. If the path contains a ``zarr.json`` at its top
level it is validated directly; otherwise all Zarr stores found under the path are
validated.

The spec version defaults to ``"0.1"`` and is caller-supplied, not read from the store:

.. code-block:: python

   results = validate("path/to/experiment/", spec_version="0.1")

Interpreting Results
--------------------

Each result in the returned list corresponds to one validated Zarr node:

* ``r.passed`` — ``True`` when there are no ``ERROR``-severity issues.
* ``r.issues`` — list of :class:`Issue` objects, each with a ``severity``
  (``ERROR`` or ``WARNING``) and a ``message``.

**Errors** correspond to MUST violations — the store fails validation.
**Warnings** correspond to SHOULD violations — informational only, the store still passes.

.. code-block:: python

   from dca_helpers.validation.result import Severity

   for r in results:
       errors   = [i for i in r.issues if i.severity == Severity.ERROR]
       warnings = [i for i in r.issues if i.severity == Severity.WARNING]
       print(f"{r.node_path}: {len(errors)} errors, {len(warnings)} warnings")

S3 Access
---------

Set AWS credentials before calling ``validate``:

.. code-block:: bash

   AWS_PROFILE=my-profile dca-validate s3://my-bucket/data/image.zarr

Or in Python:

.. code-block:: python

   import os
   os.environ["AWS_PROFILE"] = "my-profile"

   from dca_helpers.validation import validate
   results = validate("s3://my-bucket/data/image.zarr")

.. note::

   For large HCS plates with thousands of wells, point ``validate`` at the plate root
   or an individual field image directly rather than a parent directory. Recursive S3
   discovery over thousands of ``zarr.json`` files can exhaust session credentials
   before completing. Validating the plate root also runs structural checks (e.g.
   verifying all declared wells exist as valid Zarr groups).
