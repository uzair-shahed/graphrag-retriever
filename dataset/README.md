# Dataset Setup

The full MuSiQue dataset is deliberately not committed to this repository.
The root project `.gitignore` excludes downloaded dataset files; this document
is tracked so a fresh clone can explain how to acquire them.

Download the official MuSiQue v1.0 archive from the link in the upstream
[dataset README](https://github.com/StonyBrookNLP/musique#data), then extract
its `data/` directory into `dataset/musique/`. The application expects the
JSONL files under `dataset/musique/data/data/`, for example:

```text
dataset/musique/data/data/musique_full_v1.0_dev.jsonl
```

Alternatively, set `GRAPH_RAG_DATASET_PATHS` in the local `.env` to one or more
JSONL file paths. The application can be installed and its unit tests run
without downloading this dataset; the small project fixtures are tracked under
`tests/fixtures/musique/`.

MuSiQue is released under CC BY 4.0. Preserve the attribution and license
requirements described in the upstream README and the local
[dataset description](../agent-coordination/musique-dataset-description.md).