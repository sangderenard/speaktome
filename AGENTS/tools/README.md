# Agent Tools

Utility scripts and helper functions for the SpeakToMe project.

* `report_abstracttensor_methods.py` – compare `AbstractTensor` methods
  against `tensors/abstraction_functions.md` to highlight documentation gaps.
* `sort_abstracttensor_methods.py` – reorder `AbstractTensor` methods based on
  `tensors/abstraction_functions.md` and write the result to a new file.
* `headers/run_header_checks.py` – orchestrate `headers/auto_fix_headers.py`,
  `headers/validate_headers.py` and `headers/test_all_headers.py` in one step.
* `stats_counter.py` – print friendly line counts across the repository.

> **Note**
> The legacy helper `headers/header_utils.py` exists only for backwards
> compatibility. Do **not** import it in new code.
