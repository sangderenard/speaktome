# Headless Environment Setup (Deprecated)

Earlier versions of the project supported a `--headless` flag for non-interactive installation. That mode has been removed. The instructions below remain for historical reference only.

1. **Update the codebase map**

   ```bash
   python AGENTS/tools/update_codebase_map.py > AGENTS/codebase_map.json
   ```

   The JSON file lists every project directory and its optional dependency groups.

2. **Run the developer setup**

Specify the codebases you wish to install via the `--codebases` flag. When no codebases are specified,
`setup_env_dev` installs every entry from the map.

```bash
CODEBASES="speaktome laplace time_sync" \
  bash setup_env_dev.sh --groups=gui --groups=ml
```

Windows PowerShell equivalent:

```powershell
$env:CODEBASES = 'speaktome laplace time_sync'
powershell -ExecutionPolicy Bypass -File setup_env_dev.ps1 -groups gui -groups ml
```

The previous headless option installed each selected codebase in editable mode. Avoid manual `pip install` commands and rerun the setup script if dependencies change.
