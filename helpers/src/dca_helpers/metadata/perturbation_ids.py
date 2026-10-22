"""Perturbation identifiers the DCA schema asks for, from what an ingestor has.

Genetic perturbations carry an Ensembl gene id; compounds carry a PubChem CID and
the pinned Morgan fingerprint of that CID's PubChem record. The lookups query
Ensembl REST and PubChem PUG REST through the shared retrying client, so they raise
the same network errors the ontology lookups do. ``morgan_fingerprint`` is offline
and needs the optional ``chem`` extra (RDKit); ``rdkit_version`` reports the RDKit
the fingerprint's provenance should name.
"""

import re
from urllib.parse import quote

from dca_helpers.validation.ontology_lookup import (
    ENSEMBL_BASE_URL,
    PUBCHEM_BASE_URL,
    _get_with_retry,
)

_ENSEMBL_JSON = {"content-type": "application/json"}
_ENSEMBL_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LRG_PREFIX = "LRG_"
_NCBI_GENE_ID = re.compile(r"^(?:NCBIGene:)?(\d+)$")
_PUBCHEM_CID = re.compile(r"^(?:PubChem:)?(\d+)$")
_RDKIT_MISSING = "needs RDKit: install dca_helpers[chem]"
_HTTP_NOT_FOUND = 404

_MORGAN_RADIUS = 2
_MORGAN_FP_SIZE = 2048


def _ensembl_gene_ids(url: str, params: dict) -> list[str]:
    resp = _get_with_retry(url, params)
    resp.raise_for_status()
    ids: dict[str, None] = {}
    for record in resp.json():
        gene_id = record.get("id", "")
        if (
            record.get("type") == "gene"
            and _ENSEMBL_STABLE_ID.match(gene_id)
            and not gene_id.startswith(_LRG_PREFIX)
        ):
            ids.setdefault(f"ensembl:{gene_id}")
    return list(ids)


def search_ensembl(symbol: str, *, species: str = "homo_sapiens") -> list[str]:
    """Return ``ensembl:<id>`` CURIEs for the genes Ensembl links to ``symbol``.

    Calls ``/xrefs/symbol/<species>/<symbol>`` with ``object_type=gene``. Every
    gene record's stable id is kept, whatever the species' id scheme (``ENSG…``,
    ``FBgn…``, ``WBGene…``, yeast ORF names); Locus Reference Genomic (``LRG_…``)
    records are dropped. Ids are de-duplicated, in the order Ensembl returns them. A
    symbol with genes on alternate haplotypes returns several ids. An unknown symbol
    returns an empty list (Ensembl answers 200 with ``[]``). ``species`` and
    ``symbol`` are percent-encoded as path segments; Ensembl still answers 404 to an
    encoded ``/``, which raises ``requests.HTTPError``.
    """
    return _ensembl_gene_ids(
        f"{ENSEMBL_BASE_URL}/xrefs/symbol/{quote(species, safe='')}/"
        f"{quote(symbol, safe='')}",
        {"object_type": "gene", **_ENSEMBL_JSON},
    )


def ensembl_from_ncbi_gene(
    ncbi_gene_id: str, *, species: str = "homo_sapiens"
) -> list[str]:
    """Return ``ensembl:<id>`` CURIEs for an NCBI Gene id (``NCBIGene:7157`` or ``7157``).

    Calls ``/xrefs/symbol/<species>/<id>`` with ``external_db=EntrezGene``, which
    matches the NCBI Gene id among Ensembl's EntrezGene cross-references. Results
    are filtered and ordered as in :func:`search_ensembl`. Raises ValueError when
    the id is not numeric.
    """
    match = _NCBI_GENE_ID.match(ncbi_gene_id.strip())
    if not match:
        raise ValueError(f"not an NCBI Gene id: {ncbi_gene_id!r}")
    return _ensembl_gene_ids(
        f"{ENSEMBL_BASE_URL}/xrefs/symbol/{quote(species, safe='')}/{match.group(1)}",
        {"object_type": "gene", "external_db": "EntrezGene", **_ENSEMBL_JSON},
    )


def search_pubchem(
    *,
    name: str | None = None,
    smiles: str | None = None,
    inchikey: str | None = None,
) -> list[str]:
    """Return ``PubChem:<cid>`` CURIEs for one compound name, SMILES or InChIKey.

    Pass exactly one non-empty argument; anything else raises ValueError. Calls
    ``/compound/<namespace>/cids/JSON`` with the value as a query parameter, so
    SMILES characters such as ``/``, ``#`` and ``=`` are percent-encoded and never
    read as part of the path. CIDs come back in PubChem's order, de-duplicated. A
    404 (no match) returns an empty list.
    """
    given = {
        namespace: value
        for namespace, value in (
            ("name", name),
            ("smiles", smiles),
            ("inchikey", inchikey),
        )
        if value
    }
    if len(given) != 1:
        raise ValueError(
            "search_pubchem needs exactly one non-empty name, smiles or inchikey"
        )
    [(namespace, value)] = given.items()
    resp = _get_with_retry(
        f"{PUBCHEM_BASE_URL}/compound/{namespace}/cids/JSON", {namespace: value}
    )
    if resp.status_code == _HTTP_NOT_FOUND:
        return []
    resp.raise_for_status()
    cids = resp.json().get("IdentifierList", {}).get("CID", [])
    return list(dict.fromkeys(f"PubChem:{cid}" for cid in cids))


def pubchem_smiles(cid: str) -> str:
    """Return the ``SMILES`` of the PubChem record for ``cid`` (``PubChem:2244`` or ``2244``).

    Calls ``/compound/cid/<cid>/property/SMILES/JSON``. This record SMILES is the
    structure the schema's ``morgan_fingerprint`` is computed from, salt or parent as
    the record holds it. Raises ValueError naming the CID when it is not numeric,
    when PubChem answers 404, or when the record carries no SMILES (PubChem answers
    200 with only the CID for an id it has no compound for).
    """
    match = _PUBCHEM_CID.match(cid.strip())
    if not match:
        raise ValueError(f"not a PubChem CID: {cid!r}")
    resp = _get_with_retry(
        f"{PUBCHEM_BASE_URL}/compound/cid/{match.group(1)}/property/SMILES/JSON", {}
    )
    if resp.status_code == _HTTP_NOT_FOUND:
        raise ValueError(f"PubChem has no record for {cid}")
    resp.raise_for_status()
    properties = resp.json().get("PropertyTable", {}).get("Properties", [])
    smiles = properties[0].get("SMILES") if properties else None
    if not smiles:
        raise ValueError(f"PubChem record {cid} has no SMILES")
    return smiles


def rdkit_version() -> str:
    """Return the version of the RDKit :func:`morgan_fingerprint` runs on.

    The schema asks for it in the ``morgan_fingerprint`` provenance, since the
    fingerprint is only comparable across the RDKit versions that agree on it. Raises
    ImportError when RDKit is not installed, as :func:`morgan_fingerprint` does.
    """
    try:
        import rdkit
    except ImportError as error:
        raise ImportError(f"rdkit_version {_RDKIT_MISSING}") from error
    return rdkit.__version__


def morgan_fingerprint(smiles: str) -> str:
    """Return the schema's pinned Morgan fingerprint of ``smiles`` as FPS hex.

    RDKit Morgan generator, radius 2, 2048 bits, chirality not included, encoded
    with ``DataStructs.BitVectToFPSText`` (512 lowercase hex characters). Offline
    and deterministic. Raises ValueError when RDKit cannot parse ``smiles``, and
    ImportError when RDKit is not installed.

    RDKit is imported here rather than at module top because it is the optional
    ``chem`` extra: importing ``dca_helpers.metadata`` must work without it, and
    the missing-extra error belongs to the one call that needs RDKit.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as error:
        raise ImportError(f"morgan_fingerprint {_RDKIT_MISSING}") from error
    mol = Chem.MolFromSmiles(smiles) if smiles.strip() else None
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES {smiles!r}")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=_MORGAN_RADIUS, fpSize=_MORGAN_FP_SIZE, includeChirality=False
    )
    return DataStructs.BitVectToFPSText(generator.GetFingerprint(mol))
