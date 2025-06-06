# SpeakToMe

This package implements beam search controllers and utilities for generating text.

## Project Layout

Source code now lives under `speaktome/` with only helper scripts and documentation in the repository root.  A `testing/` directory contains small inspection scripts used to evaluate components in isolation without any external framework.  For example run `python testing/lookahead_demo.py` to exercise the lookahead controller.

## Sentence Transformer Model

Many features rely on a [SentenceTransformer](https://www.sbert.net/) model. The model
is loaded lazily on first use. By default, the package downloads the
`paraphrase-MiniLM-L6-v2` model if it is not already available locally.

If the environment variable `SENTENCE_TRANSFORMER_MODEL_PATH` is set its value is
used. Otherwise the package looks for a local copy under `models/paraphrase-MiniLM-L6-v2`
before attempting any network download.

## Environment Setup

Use the included script to create a virtual environment and install only the
core dependencies (`numpy` and other light utilities). Heavy libraries such as
PyTorch and Transformers are now *optional* and live in
`optional_requirements.txt`. Pass `--extras` to install them up front, and
`--prefetch` if you want to download models during setup:

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
required models **only if you installed the optional dependencies**
(PyTorch and Transformers):

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
access. If you skip this step, the application falls back to the lightweight
CPU demo mode.

## Running SpeakToMe

Use the provided wrapper scripts to ensure the `.venv` interpreter is used
consistently across platforms. They call Python from the virtual environment
directly so you don't need to activate it first.

```bash
bash run.sh -s "Hello" -m 10 -c -a 5 --final_viz
```

On Windows run either script:

```powershell
.\run.ps1 -s "Hello" -m 10 -c -a 5 --final_viz
```

From `cmd.exe`:
```cmd
run.cmd -s "Hello" -m 10 -c -a 5 --final_viz
```

These scripts are equivalent to running:

```bash
.venv/bin/python -m speaktome.speaktome [args]
```

## Command Line Usage

The `speaktome` entry point exposes several options. Long flags have short aliases
for convenience. By default the program expands once automatically, disables
retirement, and uses maximum lookahead. For example:

```
bash run.sh -s "Hello" -m 10 -c -a 5 --final_viz
```

On Windows run either script:

```powershell
.\run.ps1 -s "Hello" -m 10 -c -a 5 --final_viz
```

From `cmd.exe`:
```cmd
run.cmd -s "Hello" -m 10 -c -a 5 --final_viz
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
by default, but you can supply `-y` (or `--yes`) to skip the prompt for fully
automated workflows.

```bash
bash reinstall_env.sh -y --prefetch             # minimal reinstall
bash reinstall_env.sh -y --extras --prefetch    # reinstall with optional packages
```

On Windows use the accompanying PowerShell script:

```powershell
powershell -ExecutionPolicy Bypass -File reinstall_env.ps1 -Yes --prefetch
```

### Windows Without PowerShell
If you prefer to avoid PowerShell, create the environment and run the program from `cmd.exe`:

1. `py -3 -m venv .venv`
2. `.venv\Scripts\activate.bat`
3. `pip install --upgrade pip`
4. `pip install -r requirements.txt`  # installs only NumPy
   Add `optional_requirements.txt` to get PyTorch and Transformers
5. Run the program with `run.cmd`:
   `run.cmd -s "Hello" -m 10 -c -a 5 --final_viz`

   Or manually invoke:
   `.venv\Scripts\python.exe -m speaktome.speaktome [args]`

To download models without PowerShell, open Python from the virtual environment and execute the commands shown in `fetch_models.sh`.

### Automated Demo

Use the helper scripts `auto_demo.sh` (Bash) and `auto_demo.ps1` (PowerShell) to reinstall the environment non-interactively and run a few example searches. Pass `--interactive` (or `-interactive` on Windows) to end with a manual test run.

```bash
bash auto_demo.sh --interactive
```

```powershell
./auto_demo.ps1 -interactive
```

## Testing

Install the development requirements and run `pytest`:

```bash
pip install -r requirements-dev.txt
pytest -v
```

The test suite exercises the command line interface and other helper modules.

## License

SpeakToMe is released under the terms of the [MIT License](LICENSE).

## Contributing

Contributions are welcome! New explorers should read `AGENTS.md` and add an
experience report under `todo/experience_reports/`. Include any prompts that
informed your work so future agents can trace the discussion. Use pull requests
for code changes and feel free to open issues for questions or feature
requests.
