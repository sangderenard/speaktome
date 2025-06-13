# SpeakToMe

This package implements beam search controllers and utilities for generating text.

## Project Layout

Source code resides under `speaktome/` with helper scripts and documentation in the repository root. The `tests/` directory contains the automated pytest suite. The nearby `testing/` folder holds ad-hoc inspection scripts for quick manual experiments—for example run `python testing/lookahead_demo.py` to exercise the lookahead controller.

This repository is a cooperative development environment for biological and digital agents. A **codebase** refers to any project directory registered in `AGENTS/CODEBASE_REGISTRY.md`, such as `speaktome`. When adding parallel numeric functionality, always rely on the `AbstractTensor` interface so implementations remain backend agnostic.

### Faculty Levels
The project operates at three resource tiers:
1. **NumPy** – lightweight demo mode.
2. **Torch** – full production features with PyTorch.
3. **PyGeo** – advanced graph search via PyTorch Geometric.
The current level is detected automatically and printed when running the CLI.

## Sentence Transformer Model

Many features rely on a [SentenceTransformer](https://www.sbert.net/) model. The model
is loaded lazily on first use. By default, the package downloads the
`paraphrase-MiniLM-L6-v2` model if it is not already available locally.

If the environment variable `SENTENCE_TRANSFORMER_MODEL_PATH` is set its value is
used. Otherwise the package looks for a local copy under `models/paraphrase-MiniLM-L6-v2`
before attempting any network download.

## Environment Setup

All environment setup is documented in `../ENV_SETUP_OPTIONS.md`.


### Running Python Modules

Always run modules from the repository root so editable imports resolve
correctly. Ensure the virtual environment described in
`../ENV_SETUP_OPTIONS.md` is active before invoking any module.


## Running SpeakToMe

Use the provided wrapper scripts to ensure the `.venv` interpreter is used
consistently across platforms. They call Python from the virtual environment
directly so you don't need to activate it first.

```bash
bash run.sh -s "Hello" -m 10 -c -a 5 --final_viz
```

If PyTorch or Transformers are not installed the program automatically runs the
CPU demo. Any unsupported options are ignored so you can use the same command
line regardless of which dependencies are available. The demo drives the
LookaheadController using NumPy-only pseudotensors and returns the top ``k``
results after ``d`` steps. Any extra tokens on the command line become the seed
text, so `bash run.sh hello` will search starting from ``hello``. The default
depth is ``5`` and width is ``3``. Use `-d`/`--depth` to change the number of
lookahead steps and `-w`/`--width` for the beam width.

On Windows run either script:

```powershell
.\run.ps1 -s "Hello" -m 10 -c -a 5 --final_viz
```

From `cmd.exe`:
```cmd
run.cmd -s "Hello" -m 10 -c -a 5 --final_viz
```

## Job Selection

The `AGENTS` directory contains repeatable jobs for agents. To fetch a task at
random run:

```bash
python -m AGENTS.tools.dispense_job
```

For a guided menu that also executes any obvious setup commands, run:

```bash
python -m AGENTS.tools.select_and_run_job
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
PyTorch Geometric. The `torch_geometric` package is installed automatically the
first time you enable the GNN with `--with_gnn`. You may also install the
`ml` extras group manually if desired.


### Resetting the Environment




## Testing

Refer to `../ENV_SETUP_OPTIONS.md` for instructions on running the test suite.

Each test run writes a log to `testing/logs/pytest_<TIMESTAMP>.log` so results
are preserved across sessions. Older logs are automatically pruned to keep only
the ten most recent files. `.gitignore` excludes these log files from version
control.

The test suite exercises the command line interface and other helper modules.

## Changelog

Changes are summarized in `CHANGELOG.md`. We generate entries automatically
using [standard-version](https://github.com/conventional-changelog/standard-version).
Run `npm run release` after merging notable commits to bump the version and
append to the log. Additional YAML snapshots can be stored under `.changes/`
for future automation.

## License

SpeakToMe is released under the terms of the [MIT License](LICENSE).

## Contributing

Contributions are welcome! New explorers should read `AGENTS.md` and add an
experience report under `AGENTS/experience_reports/`. For a high level view of
the project see `AGENTS/PROJECT_OVERVIEW.md`, and consult
`AGENTS/CONTRIBUTING.md` for detailed contribution guidelines. Include any
prompts that informed your work so future agents can trace the discussion. Use
pull requests for code changes and feel free to open issues for questions or
feature requests.
You can also review the consolidated digest under `AGENTS/messages/outbox/archive/` for a short summary of prior insights.
