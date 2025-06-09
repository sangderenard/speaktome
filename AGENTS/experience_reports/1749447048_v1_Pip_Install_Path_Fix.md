Describe the purpose of this run or scenario.

## Prompts
The user observed the following error when running `setup_env.sh`:
"ERROR: file:///workspace/speaktome does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found."

## Steps Taken
1. Reviewed the environment setup scripts.
2. Noticed `pip install -e .` was executed from the repository root which lacks `pyproject.toml`.
3. Modified `setup_env.sh` and `setup_env.ps1` to install from the `speaktome` subdirectory by default.
4. Updated the default `CODEBASES` value in both scripts.

## Observed Behaviour
The scripts now install the package without error when no root `pyproject.toml` is present.

## Lessons Learned
Ensuring the correct working directory for editable installs avoids confusion when the Python package lives in a subfolder.

## Next Steps
Run the test suite to confirm no regressions.
