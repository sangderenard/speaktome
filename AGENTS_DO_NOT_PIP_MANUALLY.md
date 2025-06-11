# **Stop Installing with `pip` Like a Caveman**

The `setup_env.sh` and `setup_env_dev.sh` scripts already know how to install everything for you. Manually invoking `pip` defeats the purpose of the curated environment. Study these scripts to understand how dependencies are configured, which optional groups exist, and how editable installs are performed.

Running `setup_env.sh` creates a virtual environment and installs CPU Torch by default. Previous revisions advised calling `AGENTS/tools/dev_group_menu.py` with flags such as `--prefetch`. Those instructions are obsolete. Simply run `setup_env_dev.sh` or `setup_env.sh` with no extra options.

Refer to `AGENTS/OBSOLETE_SETUP_GUIDE.md` for details.

If something is missing, re-run `setup_env_dev.sh` and select all relevant groups instead of hammering `pip install` by hand. This keeps the environment reproducible for everyone and avoids half-installed states.

### Headless Testing

For non-interactive environments run `setup_env_dev.sh` normally and then
execute `python testing/test_hub.py`. Do not call `pip` directly to install
extras.

**In short:** read the setup scripts and use the headless menu variant
instead of manual `pip` commands.
