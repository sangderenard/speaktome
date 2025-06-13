# Poetry-Core Auto Install Update

## Overview
Addressed feedback regarding missing `poetry.core.masonry.api` by refining the automatic installation in `setup_env.sh` and implementing a previously stubbed environment detection helper.

## Prompts
- "investigate the reason powtry.core.masonry.api is not available, determine if we need a pre-poetry install of poetry"
- "attempt to follow instructions to set up the environment and document the results"
- "please make the necessary adjustments"

## Steps Taken
1. Replaced direct `pip install` call with `python -m pip` for reliability.
2. Implemented the `detect_agent_environment` stub in `todo/agent_environment_detection_proto.py`.
3. Ran the guestbook validator and test hub.

## Observed Behaviour
- `python AGENTS/validate_guestbook.py` succeeded with no issues.
- `python testing/test_hub.py` failed due to missing environment setup.

## Lessons Learned
- Using `python -m pip` ensures the installation uses the same interpreter.
- The environment detection helper now checks for `.venv/pytest_enabled` and a populated active file.

## Next Steps
- Re-run `setup_env.sh` to confirm the new poetry-core logic works as expected.
