# Log Report

**Date:** 1749895937
**Title:** Pytest environment setup failure log

## Command
`python testing/test_hub.py`

## Log
```text
[INFO] poetry-core missing; installing to enable editable builds
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
[INFO] Skipping torch groups
Creating virtualenv speaktome-hub in /workspace/speaktome/.venv
The `--sync` option is deprecated and slated for removal in the next minor release after June 2025, use the `poetry sync` command instead.

The dependency name for testenv does not match the actual package's name: speaktome-testenv
Warning: command 'env POETRY_VIRTUALENVS_IN_PROJECT=true poetry install --sync --no-interaction --without cpu-torch --without gpu-torch' failed with status 1
[DEBUG] Codebases: 
[DEBUG] Groups: 
The `--sync` option is deprecated and slated for removal in the next minor release after June 2025, use the `poetry sync` command instead.

The dependency name for testenv does not match the actual package's name: speaktome-testenv
Warning: No codebases recorded; pytest will remain disabled.
Updating dependencies
Resolving dependencies...
Launching codebase/group selection tool for editable installs...
Updating dependencies
Resolving dependencies...

Select codebases and optional groups (press letter to toggle, 'c' to continue, or 'q' to quit):
  (a) [ ] Codebase: fontmapper
       (b) [ ] Group: amqp
       (c) [ ] Group: dev
       (d) [ ] Group: gui
       (e) [ ] Group: server
       (f) [ ] Group: ssim
       (g) [ ] Group: torch
  (h) [ ] Codebase: laplace
       (i) [ ] Group: dev
  (j) [ ] Codebase: speaktome
       (k) [ ] Group: dev
       (l) [ ] Group: ml
       (m) [ ] Group: plot
  (n) [ ] Codebase: tensorprinting
       (o) [ ] Group: dev
  (p) [ ] Codebase: tensors
       (q) [ ] Group: ctensor
       (r) [ ] Group: dev
       (s) [ ] Group: jax
       (t) [ ] Group: numpy
       (u) [ ] Group: opengl
       (v) [ ] Group: torch
  (w) [ ] Codebase: timesync
       (x) [ ] Group: dev
       (y) [ ] Group: gui
  (z) [ ] Codebase: tools
No input, continuing...
Selected codebases: []
Selected packages: {}
Environment setup complete.
[OK] Environment ready. Activate with 'source .venv/bin/activate'.
   * Torch = missing
Selections recorded to /tmp/speaktome_active.json
Install without virtualenv? [y/N] (auto-N in 3s): 
Traceback (most recent call last):
  File "/workspace/speaktome/testing/test_hub.py", line 59, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/workspace/speaktome/testing/test_hub.py", line 45, in main
    ret = pytest.main(pytest_args, plugins=[collector])
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/config/__init__.py", line 156, in main
    config = _prepareconfig(args, plugins)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/config/__init__.py", line 341, in _prepareconfig
    config = pluginmanager.hook.pytest_cmdline_parse(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/pluggy/_callers.py", line 167, in _multicall
    raise exception
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/pluggy/_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/helpconfig.py", line 105, in pytest_cmdline_parse
    config = yield
             ^^^^^
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/config/__init__.py", line 1140, in pytest_cmdline_parse
    self.parse(args)
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/config/__init__.py", line 1494, in parse
    self._preparse(args, addopts=addopts)
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/config/__init__.py", line 1398, in _preparse
    self.hook.pytest_load_initial_conftests(
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/pluggy/_hooks.py", line 512, in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/pluggy/_manager.py", line 120, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/pluggy/_callers.py", line 167, in _multicall
    raise exception
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/pluggy/_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/warnings.py", line 151, in pytest_load_initial_conftests
    return (yield)
            ^^^^^
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/pluggy/_callers.py", line 139, in _multicall
    teardown.throw(exception)
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/capture.py", line 172, in pytest_load_initial_conftests
    yield
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/pluggy/_callers.py", line 121, in _multicall
    res = hook_impl.function(*args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/config/__init__.py", line 1222, in pytest_load_initial_conftests
    self.pluginmanager._set_initial_conftests(
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/config/__init__.py", line 581, in _set_initial_conftests
    self._try_load_conftest(
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/config/__init__.py", line 619, in _try_load_conftest
    self._loadconftestmodules(
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/config/__init__.py", line 659, in _loadconftestmodules
    mod = self._importconftest(
          ^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/config/__init__.py", line 710, in _importconftest
    mod = import_path(
          ^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/pathlib.py", line 587, in import_path
    importlib.import_module(module_name)
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/assertion/rewrite.py", line 185, in exec_module
    exec(co, module.__dict__)
  File "/workspace/speaktome/tests/conftest.py", line 109, in <module>
    pytest.skip(
  File "/root/.pyenv/versions/3.12.10/lib/python3.12/site-packages/_pytest/outcomes.py", line 160, in skip
    raise Skipped(msg=reason, allow_module_level=allow_module_level)
Skipped: Environment not initialized. See ENV_SETUP_OPTIONS.md
Automated setup failed. Skipping all tests.
```

## Prompt History
```
who in the scripts is responsible for poetry core and can you dump the failure to a markdown log experience report and create a new experience report type and guidance for logs, which should try pretty logger but if that can't be loaded, the log experience reports should just be raw log files made markup so github tracks them, their template should just explain there are to be piped output from repeatable errors or requested analysis
```
