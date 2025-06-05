# SpeakToMe

This package implements beam search controllers and utilities for generating text.

## Sentence Transformer Model

Many features rely on a [SentenceTransformer](https://www.sbert.net/) model. The model
is loaded lazily on first use. By default, the package downloads the
`paraphrase-MiniLM-L6-v2` model if it is not already available locally.

To avoid network downloads, set the environment variable
`SENTENCE_TRANSFORMER_MODEL_PATH` to the directory of a pre-downloaded model or a
custom model path before running the program.

## Environment Setup

Use the included script to create a virtual environment and install
dependencies:

```bash
bash setup_env.sh
```

On Windows use the PowerShell script:

```powershell
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

Activate the environment with:

```bash
source .venv/bin/activate
```

On Windows activate with:

```powershell
.venv\Scripts\Activate.ps1
```

### Offline Setup

After creating the virtual environment, run the following script to download the
required models:

```bash
bash fetch_models.sh
```

On Windows run:

```powershell
powershell -ExecutionPolicy Bypass -File fetch_models.ps1
```

The script stores GPT-2 and the SentenceTransformer model under the `models/`
directory. Set the following environment variables to use the local copies:

```bash
export GPT2_MODEL_PATH=models/gpt2
export SENTENCE_TRANSFORMER_MODEL_PATH=models/paraphrase-MiniLM-L6-v2
```

Once the models are downloaded you can run the program without further network
access.

## Command Line Usage

The `speaktome` entry point exposes several options. Long flags have short aliases
for convenience. By default the program expands once automatically, disables
retirement, and uses maximum lookahead. For example:

```
python speaktome.py -s "Hello" -m 10 -c -a 5 --final_viz
```

The above runs the search with seed text "Hello", a maximum depth of 10, enables
human control, automatically performs five `expand_any` rounds before
interactive control resumes, and shows the final tree. Use `-g N` to let the
PyGeoMind model control for `N` rounds before handing control back.

Pass `-x` or `--safe_mode` to force CPU execution when you want to avoid GPU
usage entirely.

