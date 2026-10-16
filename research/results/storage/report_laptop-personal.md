# Index cost of `laptop-personal`

Scanned: 12,394 files, 68.9 GB. Rows: 212,361 chunks (95% interval 86,138 to 443,184, from the document sample) and 6,170 single-row files.

| Embedding model | chunk index | of which vectors | SQLite catalogue | total | without compaction |
|---|---:|---:|---:|---:|---:|
| e5-small, 384-d (default) | 519.4 MB | 320.1 MB | 15.3 MB | 534.7 MB | 730.1 MB |
| 1024-d model (bge-m3 class) | 1.0 GB | 853.6 MB | 15.3 MB | 1.0 GB | 1.4 GB |

Scaled to 500 GB with the same composition (x7.3):

| Embedding model | total index | share of the disk | embedding time, laptop CPU | LLM calls |
|---|---:|---:|---:|---:|
| 384-d | 3.8 GB | 0.76% | 40 to 50 h | 102,501 |
| 1024-d | 7.6 GB | 1.51% | not measured (slower) | 102,501 |

LLM descriptions (102,501 calls: one per document, text file, media and unknown file, two per image) dominate the time if they are on: at an assumed 2 to 4 s per call on a GPU, 57 to 114 h. This rate is an assumption, not a measurement.
