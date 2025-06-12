# Headless Environment Setup

This document explains how to install codebases without interactive prompts.

1. **Update the codebase map**

   ```bash
   python AGENTS/tools/update_codebase_map.py > AGENTS/codebase_map.json
   ```

   The JSON file lists every project directory and its optional dependency groups.

2. **Run the developer setup in headless mode**

Provide the codebases you wish to install via the `CODEBASES` environment
variable or the `--codebases` flag. When no codebases are specified,
`setup_env_dev` installs every entry from the map.

```bash
CODEBASES="speaktome laplace time_sync" \
  bash setup_env_dev.sh --groups=gui --groups=ml --headless
```

Windows PowerShell equivalent:

```powershell
$env:CODEBASES = 'speaktome laplace time_sync'
powershell -ExecutionPolicy Bypass -File setup_env_dev.ps1 -groups gui -groups ml -headless
```

The headless option installs each selected codebase in editable mode based on the JSON map. Avoid manual `pip install` commands and rerun the setup script if dependencies change.
