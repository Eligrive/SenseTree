# Index cost of `laptop-projects`

Scanned: 16,453 files, 17.3 GB. Rows: 805,670 chunks (95% interval 794,559 to 821,476, from the document sample) and 10,880 single-row files.

| Embedding model | chunk index | of which vectors | SQLite catalogue | total | without compaction |
|---|---:|---:|---:|---:|---:|
| e5-small, 384-d (default) | 1.9 GB | 1.2 GB | 20.3 MB | 1.9 GB | 2.6 GB |
| 1024-d model (bge-m3 class) | 3.8 GB | 3.1 GB | 20.3 MB | 3.9 GB | 5.3 GB |

Scaled to 500 GB with the same composition (x28.8):

| Embedding model | total index | share of the disk | embedding time, laptop CPU | LLM calls |
|---|---:|---:|---:|---:|
| 384-d | 55.2 GB | 11.05% | 600 to 743 h | 605,583 |
| 1024-d | 111.4 GB | 22.27% | not measured (slower) | 605,583 |

LLM descriptions (605,583 calls: one per document, text file, media and unknown file, two per image) dominate the time if they are on: at an assumed 2 to 4 s per call on a GPU, 336 to 673 h. This rate is an assumption, not a measurement.
