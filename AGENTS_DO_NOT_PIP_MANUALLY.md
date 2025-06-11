# **Stop Installing with `pip` Like a Caveman**

The `setup_env.sh` and `setup_env_dev.sh` scripts already know how to install everything for you. Manually invoking `pip` defeats the purpose of the curated environment. Study these scripts to understand how dependencies are configured, which optional groups exist, and how editable installs are performed.

Running `setup_env.sh` creates a virtual environment, installs CPU Torch by default, and launches the `AGENTS/tools/dev_group_menu.py` helper so you can pick which codebases and optional dependency groups should be installed. The developer variant, `setup_env_dev.sh`, activates the environment, installs `requirements-dev.txt`, and opens an interactive menu for documentation and stub tracking.

To run either script non-interactively, invoke the menu tool yourself with explicit selections:

```bash
# create the environment without prompts using the developer script
bash setup_env_dev.sh --prefetch
# choose codebases and groups headlessly
python AGENTS/tools/dev_group_menu.py --install \
    --codebases speaktome \
    --groups speaktome:dev
```

Refer to `AGENTS/CODEBASE_REGISTRY.md` for a complete list of available codebase
s.

`--codebases` specifies which project folders to install and `--groups` lists the optional dependency groups. Passing these arguments bypasses every menu prompt so the environment can be prepared automatically.

If something is missing, re-run `setup_env_dev.sh` and select all relevant groups instead of hammering `pip install` by hand. This keeps the environment reproducible for everyone and avoids half-installed states.

### Headless Testing

When running tests in a non-interactive environment, first invoke the
menu tool as shown above so every optional group you need is installed.
After the environment is prepared, execute `python testing/test_hub.py` to
run the suite. Never call `pip` directly to add extras for test runs.

**In short:** read the setup scripts and use the headless menu variant
instead of manual `pip` commands.
