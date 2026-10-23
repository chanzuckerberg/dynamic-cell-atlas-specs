# Recorded registry responses

Each `*.json` file holds one real Cellosaurus or OLS response plus the request that produced it, replayed by `tests/metadata/_registry.py` at the `_get_with_retry` boundary. A trimmed file keeps the response envelope and every key as returned, keeps whole records unedited and in their original order, and drops the rest. Trimmed files keep every record that matches the query exactly plus at least two near-miss records, so the exact-match filter still runs.

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

The Cellosaurus search returns every line whose identifier or synonym contains the query, up to `rows`, even with `fld=ac,id,sy`, so those responses are trimmed instead of narrowed. The OLS `/terms?iri=` request is shared with the `--check-ontology` validator (`lookup_ols`), and OLS4 has no smaller endpoint that returns the same defining-ontology label in the same order (`/terms/findByIdAndIsDefiningOntology` returns HTTP 500, and `/search` orders by relevance), so it is trimmed too.

To re-record, repeat the request with `curl -G --data-urlencode ...`, then drop records as described above.
