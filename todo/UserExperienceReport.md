# SpeakToMe User Experience Simulation

This report documents a new user's attempt to set up and explore the repository using the provided scripts. It summarizes observations from several perspectives.

## 1. Initial Setup Attempt (Linux)

1. Ran `bash setup_env.sh`.
2. Script created `.venv/` and started installing dependencies.
3. Package downloads were large; network access may be required.
4. Setup was interrupted to avoid long waits.

Excerpt from the setup output:

```
Requirement already satisfied: pip in ./.venv/lib/python3.11/site-packages (24.0)
Collecting pip
  Using cached pip-25.1.1-py3-none-any.whl.metadata (3.6 kB)
Using cached pip-25.1.1-py3-none-any.whl (1.8 MB)
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 24.0
    Uninstalling pip-24.0:
      Successfully uninstalled pip-24.0
Successfully installed pip-25.1.1
Collecting torch>=1.13 (from -r requirements.txt (line 2))
  Downloading torch-2.7.1-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (29 kB)
Collecting torch_geometric>=2.4 (from -r requirements.txt (line 3))
  Downloading torch_geometric-2.6.1-py3-none-any.whl.metadata (63 kB)
Collecting transformers>=4.30 (from -r requirements.txt (line 6))
  Downloading transformers-4.52.4-py3-none-any.whl.metadata (38 kB)
Collecting sentence-transformers>=2.2 (from -r requirements.txt (line 7))
  Downloading sentence_transformers-4.1.0-py3-none-any.whl.metadata (13 kB)
Collecting matplotlib>=3.7 (from -r requirements.txt (line 10))
  Downloading matplotlib-3.10.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
Collecting networkx>=3.1 (from -r requirements.txt (line 11))
  Downloading networkx-3.5-py3-none-any.whl.metadata (6.3 kB)
Collecting scikit-learn>=1.2 (from -r requirements.txt (line 12))
```

## 2. Setup Expectations (Windows)

* Windows users run `setup_env.ps1`.
* Activation uses `.venv\Scripts\Activate.ps1`.
* Pass `--prefetch` to `setup_env.ps1` to download models during setup.
* Offline environments can also run `fetch_models.ps1` separately.

## 3. Exploring the Code

* Core modules include `beam_search.py`, `pyg_graph_controller.py`, and `pygeo_mind.py`.
* Configuration is handled in `config.py` with lazy loading of the SentenceTransformer model.
* The `README.md` details command line usage and environment variables for model paths.

## 4. Next Steps

* Complete installation by allowing `setup_env.sh` to finish downloading dependencies.
* Optionally run `bash fetch_models.sh` (or `setup_env.sh --prefetch`) to cache models for offline use.
* Start exploring with `python speaktome.py -h` to view command line options.
  Use `--preload_models` for a one-time load of all models if you want to avoid lazy loading delays.

---

This document serves as a preliminary record of a user's onboarding journey and can be expanded with more detailed notes as development continues.
