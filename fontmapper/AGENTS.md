# FontMapper Utilities

## Quick Setup

Consult `../ENV_SETUP_OPTIONS.md` for environment setup.

This directory contains experimental scripts for converting images to ASCII art and training related models. The subfolder `FM16` holds the current iteration including configuration files and pretrained weights.

These tools rely on packages such as Torch, PIL, and Flask. Prepare the environment with the standard setup scripts.

`FMS6.py` and `FM38.py` expose command line options for text-only output or a small Flask server defined by `server.yaml`. Models are experimental and may evolve quickly. Keep additions documented and follow `AGENTS/CODING_STANDARDS.md`.

### Modular Utilities

Character set generation functions and model evaluation helpers now live in
`FM16/modules/charset_ops.py` and `FM16/modules/model_ops.py`. Import these
utilities instead of relying on the large scripts when integrating with other
projects.
