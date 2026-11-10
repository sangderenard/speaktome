# Spectral wave continuation energy

Date: 2026-11-09

## Scope and result

Audited the existing native wave in/out coupling in
`C:/dev/Powershell/spectral-analyzer` without altering production renderer code
or existing exposures. The ray/field adapters already preserve discrete power
locally, but separated wave arenas reapply source power and reset polarization.
Compatible linked field arenas preserve the full S/P state.

Native reproduction: source weight 0.25 produces relative field powers
`[0.25, 0.0625]` across two separated arenas; weight 0.5 produces `[0.5, 0.25]`.
A unitary S/P swap produces P fractions `[1, 0]` after an ordinary-ray gap.

The exact code references, implementation constraints, and test commands are
in `spectral-analyzer/WAVE_CONTINUATION_ENERGY_AUDIT_2026-11-09.md`.
Added `spectral-analyzer/tests/test_wave_t4_continuation.py` with two passing
linked controls and three strict expected-failure continuation gates.
Focused CPU-only suite: 60 passed, 4 deselected, 3 xfailed. With `--runxfail`,
the new file gives exactly 2 passed and 3 failed on the documented assertions.

No production coupling fix is claimed. The existing compact branch-handle and
generation contract must be routed through ordinary transport before an
automatic consumer can safely retain mutable per-branch Jones state. Scalar
energy compensation alone would hide the polarization failure.

## Prompt History

User: "If you need to finish the energy transfer in and out for complex wave sim coupling i would really appreciate if you did that but it did work without the aperture, it has great lens distortion but the same trouble with color"
