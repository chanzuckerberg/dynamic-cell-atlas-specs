# Recorded registry responses

Each `*.json` file holds one real Cellosaurus, OLS, Ensembl or PubChem response plus the request that produced it, replayed by `tests/metadata/_registry.py` at the `_get_with_retry` boundary. A trimmed file keeps the response envelope and every key as returned, keeps whole records unedited and in their original order, and drops the rest. Trimmed files keep every record that matches the query exactly plus at least two near-miss records, so the exact-match filter still runs.

| File | Request | Recorded | Contents |
|---|---|---|---|
| `cellosaurus_id_A549.json` | `GET https://api.cellosaurus.org/search/cell-line?q=id:"A549"&format=json&fld=ac&fld=id&fld=sy&rows=100` | 2026-10-23 | 3 of 100 records kept (no exact match in the response) |
| `cellosaurus_sy_A549.json` | same, `q=sy:"A549"` | 2026-10-23 | 3 of 100 records kept (CVCL_0023 plus the records before and after it) |
| `cellosaurus_id_HEK293.json` | same, `q=id:"HEK293"` | 2026-10-23 | 3 of 100 records kept (CVCL_0045 plus the next two) |
| `cellosaurus_sy_HEK293.json` | same, `q=sy:"HEK293"` | 2026-10-23 | 3 of 100 records kept (no exact match in the response) |
| `cellosaurus_id_HeLa_Cas9.json` | same, `q=id:"HeLa Cas9"` | committed 2026-10-21 | full response, 13 records |
| `cellosaurus_sy_HeLa_Cas9.json` | same, `q=sy:"HeLa Cas9"` | committed 2026-10-21 | full response, 2 records |
| `ols_search_ncbitaxon_Zika_virus.json` | `GET https://www.ebi.ac.uk/ols4/api/search?q=Zika virus&ontology=ncbitaxon&exact=true&rows=5` | committed 2026-10-21 | full response |
| `ols_terms_UBERON_0002048.json` | `GET https://www.ebi.ac.uk/ols4/api/terms?iri=http://purl.obolibrary.org/obo/UBERON_0002048` | 2026-10-23 | 3 of 49 terms kept (`uberon`, the defining ontology, plus imported copies from `doid` and `wbphenotype`); `page.totalElements` still reads 49 |

| `ensembl_xrefs_entrezgene_7157.json` | `GET https://rest.ensembl.org/xrefs/symbol/homo_sapiens/7157?external_db=EntrezGene&object_type=gene` | committed 2026-10-21 | full response (HTTP 200) |
| `ensembl_xrefs_symbol_ACT1_yeast.json` | `GET https://rest.ensembl.org/xrefs/symbol/saccharomyces_cerevisiae/ACT1?object_type=gene` | committed 2026-10-21 | full response (HTTP 200) |
| `ensembl_xrefs_symbol_CYP2D6.json` | `GET https://rest.ensembl.org/xrefs/symbol/homo_sapiens/CYP2D6?object_type=gene` | committed 2026-10-21 | full response (HTTP 200) |
| `ensembl_xrefs_symbol_TP53.json` | `GET https://rest.ensembl.org/xrefs/symbol/homo_sapiens/TP53?object_type=gene` | committed 2026-10-21 | full response (HTTP 200) |
| `ensembl_xrefs_symbol_no_match.json` | `GET https://rest.ensembl.org/xrefs/symbol/homo_sapiens/NOTAREALGENEXYZ?object_type=gene` | committed 2026-10-21 | full response (HTTP 200) |
| `ensembl_xrefs_symbol_unc-22_worm.json` | `GET https://rest.ensembl.org/xrefs/symbol/caenorhabditis_elegans/unc-22?object_type=gene` | committed 2026-10-21 | full response (HTTP 200) |
| `ensembl_xrefs_symbol_w_fly.json` | `GET https://rest.ensembl.org/xrefs/symbol/drosophila_melanogaster/w?object_type=gene` | committed 2026-10-21 | full response (HTTP 200) |
| `pubchem_cid_smiles_aspirin.json` | `GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/SMILES/JSON` | committed 2026-10-21 | full response (HTTP 200) |
| `pubchem_cid_smiles_no_record.json` | `GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/999999999/property/SMILES/JSON` | committed 2026-10-21 | full response (HTTP 200) |
| `pubchem_inchikey_aspirin.json` | `GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/cids/JSON?inchikey=BSYNRYMUTXBXSQ-UHFFFAOYSA-N` | committed 2026-10-21 | full response (HTTP 200) |
| `pubchem_name_aspirin.json` | `GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/cids/JSON?name=aspirin` | committed 2026-10-21 | full response (HTTP 200) |
| `pubchem_name_no_match.json` | `GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/cids/JSON?name=not+a+real+compound+name` | committed 2026-10-21 | full response (HTTP 404) |
| `pubchem_smiles_aspirin.json` | `GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/JSON?smiles=CC%28%3DO%29Oc1ccccc1C%28%3DO%29O` | committed 2026-10-21 | full response (HTTP 200) |

The Cellosaurus search returns every line whose identifier or synonym contains the query, up to `rows`, even with `fld=ac,id,sy`, so those responses are trimmed instead of narrowed. The OLS `/terms?iri=` request is shared with the `--check-ontology` validator (`lookup_ols`), and OLS4 has no smaller endpoint that returns the same defining-ontology label in the same order (`/terms/findByIdAndIsDefiningOntology` returns HTTP 500, and `/search` orders by relevance), so it is trimmed too.

To re-record, repeat the request with `curl -G --data-urlencode ...`, then drop records as described above.
