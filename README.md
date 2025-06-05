# SpeakToMe

This package implements beam search controllers and utilities for generating text.

## Sentence Transformer Model

Many features rely on a [SentenceTransformer](https://www.sbert.net/) model. The model
is loaded lazily on first use. By default, the package downloads the
`paraphrase-MiniLM-L6-v2` model if it is not already available locally.

To avoid network downloads, set the environment variable
`SENTENCE_TRANSFORMER_MODEL_PATH` to the directory of a pre-downloaded model or a
custom model path before running the program.

## Command Line Usage

The `speaktome` entry point exposes several options. Long flags have short aliases
for convenience. For example:

```
python speaktome.py -s "Hello" -m 10 -c -a 5
```

The above runs the search with seed text "Hello", a maximum depth of 10, enables
human control, and automatically performs five `expand_any` rounds before
interactive control resumes. Use `-g N` to let the PyGeoMind model control for `N`
rounds before handing control back.

