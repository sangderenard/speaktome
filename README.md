# SpeakToMe

This package implements beam search controllers and utilities for generating text.

## Sentence Transformer Model

Many features rely on a [SentenceTransformer](https://www.sbert.net/) model. The model
is loaded lazily on first use. By default, the package downloads the
`paraphrase-MiniLM-L6-v2` model if it is not already available locally.

If the environment variable `SENTENCE_TRANSFORMER_MODEL_PATH` is set its value is
used. Otherwise the package looks for a local copy under `models/paraphrase-MiniLM-L6-v2`
before attempting any network download.

## Environment Setup

Use the included script to create a virtual environment and install only the
core dependencies. Optional packages are installed lazily on first use. Pass
`--extras` to install them up front, and `--prefetch` if you want to download
models during setup:

```bash
bash setup_env.sh --prefetch           # minimal install
bash setup_env.sh --extras --prefetch  # install optional packages too
```

On Windows use the PowerShell script:

```powershell
powershell -ExecutionPolicy Bypass -File setup_env.ps1 --prefetch  # optional
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
directory. The program automatically checks this location, but you can also set
the following environment variables to specify custom paths:

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

Use `--preload_models` if you prefer to load all models up front rather than on-demand.

### Optional GNN Features

The project includes an experimental controller called **PyGeoMind** built on top of
PyTorch Geometric. The `torch_geometric` package will be installed automatically
the first time you enable the GNN with `--with_gnn`. Use `--extras` during setup
if you prefer installing it ahead of time.

### Resetting the Environment

If you ever need to recreate the virtual environment from scratch, run
`reinstall_env.sh`. This script removes the existing `.venv` directory and then
invokes `setup_env.sh` with any arguments you pass. It prompts for confirmation
and defaults to **N** if you simply press Enter.

```bash
bash reinstall_env.sh --prefetch  # minimal reinstall
bash reinstall_env.sh --extras --prefetch  # reinstall with optional packages
```

On Windows use the accompanying PowerShell script:

```powershell
powershell -ExecutionPolicy Bypass -File reinstall_env.ps1 --prefetch
```

