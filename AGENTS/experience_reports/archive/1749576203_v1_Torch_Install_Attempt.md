# Environment Setup with Torch Attempt

## Overview
Followed previous feedback to rerun environment setup script and capture logs of Torch installation attempt.

## Prompt History
- "log in and follow guidance to set up environment"
- "no attempt to install torch is in the logs please run the script again so I can see exactly what torch was attempted where and what the result was. Do you hide logs or did you not do it?"

## Steps Taken
1. Executed `bash setup_env_dev.sh --extras --prefetch` and saved output to `setup_log.txt`.
2. Allowed default selections in the interactive menu (auto-skipped after timeout).
3. Examined `setup_log.txt` for Torch installation lines.

## Observed Behaviour
- Script attempted to install Torch from `https://download.pytorch.org/whl/torch_stable.html` but failed due to proxy restrictions.
- Multiple retries were logged followed by an error stating no matching distribution was found.
- After failure, the script proceeded to install `requirements-dev.txt` packages successfully.

Excerpt from log:
```
Successfully installed pip-25.1.1
Installing latest stable CPU-only torch (default)
Looking in links: https://download.pytorch.org/whl/torch_stable.html
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 403 Forbidden'))': /whl/torch_stable.html
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 403 Forbidden'))': /whl/torch_stable.html
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 403 Forbidden'))': /whl/torch_stable.html
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 403 Forbidden'))': /whl/torch_stable.html
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 403 Forbidden'))': /whl/torch_stable.html
ERROR: Could not find a version that satisfies the requirement torch==2.3.1+cpu (from versions: 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0, 2.7.0, 2.7.1)
ERROR: No matching distribution found for torch==2.3.1+cpu
Warning: command './.venv/bin/pip install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html' failed with status 1
```

## Lessons Learned
Running the setup script again confirms Torch installation is attempted automatically but fails in this environment due to proxy blocks. Logs show warnings and the final error line.

## Next Steps
Consider offline installation or providing a prebuilt Torch wheel. Tests still fail without Torch and due to the `time_sync` module.
