# Agent Tools

Utility scripts and helper functions for the SpeakToMe project.

* `report_abstracttensor_methods.py` – compare `AbstractTensor` methods
  against `tensors/abstraction_functions.md` to highlight documentation gaps.
* `sort_abstracttensor_methods.py` – reorder `AbstractTensor` methods based on
  `tensors/abstraction_functions.md` and write the result to a new file.
* `run_header_checks.py` – orchestrate `auto_fix_headers.py`,
  `validate_headers.py` and `test_all_headers.py` in one step.

> **Note**
> The legacy helper `header_utils.py` exists only for backwards
> compatibility. Do **not** import it in new code.
