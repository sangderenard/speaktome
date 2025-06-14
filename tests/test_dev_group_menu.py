from __future__ import annotations

try:
    import os
    import json
    import subprocess
    import sys
    from pathlib import Path

    ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]

except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---
def run_menu(args, env=None):
    return subprocess.run(
        [sys.executable, 'AGENTS/tools/dev_group_menu.py', *args],
        capture_output=True, text=True, env=env
    )


def test_list_option() -> None:
    result = run_menu(['--list'])
    assert 'speaktome' in result.stdout


def test_noninteractive_json(tmp_path) -> None:
    active = tmp_path / 'active.json'
    result = run_menu([
        '--codebases', 'speaktome',
        '--groups', 'speaktome:dev',
        '--json', '--record', str(active)
    ])
    data = json.loads(result.stdout)
    assert data['codebases'] == ['speaktome']
    assert 'dev' in data['packages']['speaktome']
    assert active.exists()


def test_show_active(tmp_path) -> None:
    env = {**os.environ, 'SPEAKTOME_ACTIVE_FILE': str(tmp_path / 'active.json')}
    result = run_menu(['--show-active'], env=env)
    assert str(tmp_path / 'active.json') in result.stdout


def test_single_dash_flags(tmp_path) -> None:
    active = tmp_path / 'active.json'
    result = run_menu([
        '-codebases', 'speaktome',
        '-groups', 'speaktome:dev',
        '-json', '-record', str(active)
    ])
    data = json.loads(result.stdout)
    assert 'speaktome' in data['codebases']
    assert active.exists()
