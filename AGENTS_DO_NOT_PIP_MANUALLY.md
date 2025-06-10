# **Stop Installing with `pip` Like a Caveman**

The `setup_env.sh` and `setup_env_dev.sh` scripts already know how to install everything for you. Manually invoking `pip` defeats the purpose of the curated environment. Study these scripts to understand how dependencies are configured, which optional groups exist, and how editable installs are performed.

Running `setup_env.sh` creates a virtual environment, installs CPU Torch by default, and launches the `AGENTS/tools/dev_group_menu.py` helper so you can pick which codebases and optional dependency groups should be installed. The developer variant, `setup_env_dev.sh`, activates the environment, installs `requirements-dev.txt`, and opens an interactive menu for documentation and stub tracking.

To run either script non-interactively, invoke the menu tool yourself with explicit selections:

```bash
# create the environment without prompts
bash setup_env.sh --from-dev --extras --prefetch
# choose codebases and groups headlessly
python AGENTS/tools/dev_group_menu.py --install \
    --codebases speaktome \
    --groups speaktome:dev
```

`--codebases` specifies which project folders to install and `--groups` lists the optional dependency groups. Passing these arguments bypasses every menu prompt so the environment can be prepared automatically.

If something is missing, re-run `setup_env_dev.sh` and select all relevant groups instead of hammering `pip install` by hand. This keeps the environment reproducible for everyone and avoids half-installed states.

**In short:** read the setup scripts, don't bypass them.
