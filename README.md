# SpeakToMe

This package implements beam search controllers and utilities for generating text.

## Sentence Transformer Model

Many features rely on a [SentenceTransformer](https://www.sbert.net/) model. The model
is loaded lazily on first use. By default, the package downloads the
`paraphrase-MiniLM-L6-v2` model if it is not already available locally.

To avoid network downloads, set the environment variable
`SENTENCE_TRANSFORMER_MODEL_PATH` to the directory of a pre-downloaded model or a
custom model path before running the program.

