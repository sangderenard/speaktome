# FontMapper Utilities

This directory contains experimental scripts for converting images to ASCII art and training related models. The subfolder `FM16` holds the current iteration including configuration files and pretrained weights.

These tools rely on packages such as Torch, PIL, and Flask. Prepare the environment with the standard menu helper:

```bash
python AGENTS/tools/dev_group_menu.py --install \
    --codebases fontmapper \
    --groups speaktome:dev
```

`FMS6.py` and `FM38.py` expose command line options for text-only output or a small Flask server defined by `server.yaml`. Models are experimental and may evolve quickly. Keep additions documented and follow `AGENTS/CODING_STANDARDS.md`.
