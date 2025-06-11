# **Stop Installing with `pip` Like a Caveman**

The `setup_env.sh` and `setup_env_dev.sh` scripts already know how to install everything for you. Manually invoking `pip` defeats the purpose of the curated environment. Study these scripts to understand how dependencies are configured, which optional groups exist, and how editable installs are performed.

Running `setup_env.sh` creates a virtual environment and installs CPU Torch by default.

Refer to `AGENTS/OBSOLETE_SETUP_GUIDE.md` for details.

If something is missing, re-run `setup_env_dev.sh` and select all relevant groups instead of hammering `pip install` by hand. This keeps the environment reproducible for everyone and avoids half-installed states.

### Headless Testing

For automated environments see `ENV_SETUP_OPTIONS.md`.
