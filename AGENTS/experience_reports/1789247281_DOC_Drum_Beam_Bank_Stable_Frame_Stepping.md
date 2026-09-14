# Drum beam-bank frame-step overflow

Date: 2026-09-12

## Finding

The reported `C:\dev\Powershell\engine\_toy\drum\_reference.py` is not
present in this workspace. The matching source is
`C:\dev\Powershell\engine_toy\drum_reference.py`; it builds the same
232-member bank with source digest `9e146110cce72cc1` and reproduces the
overflow. No applicable local AGENTS.md was found under engine_toy.

At 35 fps, eight substeps give 3.571 ms per integration step. The fastest
driven mode is 565.426 Hz, with a damped semi-implicit Euler stability
boundary of 0.560149 ms. With the reported 14.4 kN distributed over the
32 driven members, the old update overflows at zero-based frame 17.

## Changes

- `structure_native.py`: shared Python/native scheduling derives stiffness
  k and damping c directly from the symbolic velocity update. It chooses
  h <= 1/(sqrt(k)+c), so k*h*h + 2*c*h <= 2, inside the stability boundary
  of 4. The full outer duration is divided into equal substeps; caller
  n_sub is a minimum. Parameters are evaluated on every outer call.
- The Python bank runs its unchanged generated source with a NumPy member
  index vector and elementwise max/min. This retains the scalar law and
  source digest while making the necessary extra substeps affordable.
- `drum_reference.py`: delegates subdivision to the bank.
- `beam_theory.py`: corrects the claim about semi-implicit stability.
- `tests/test_structure_native.py`: persistent regression coverage for
  the drum load/release/reversal, scalar/vector/native parity, changed
  temperature/geometry/added mass/damping, elapsed time, zero dt, empty
  banks, and invalid dt.

The relevant source files were already untracked under the root repository.
No files were staged, committed, or force-added, and ignore rules were not
changed. The underlying structural model and generated beam law are unchanged.

## Validation

From engine_toy:

```text
python -m unittest discover -s tests -p test_structure_native.py -v
Ran 6 tests in 27.886s
OK
```

The drum regression advances 600 frames with 1/60, 1/35, and 0.05 second
durations and verifies finite stress/state, bounded displacement relative
to static equilibrium, and decay after release.

Also ran the real `drum_reference.main(['--live'])` rendering path for
360 frames in a hidden OpenGL window, injecting lock-up, both clutch
controls, pause/resume, reset, and quit events. NumPy overflow, invalid,
and divide warnings were errors; each bank result and velocity was checked
for finiteness. Passed, with up to 180 substeps on a 50 ms frame and a
maximum transient tip reading of 5.697437 mm including lock-up loading.
Sample running frame rates were 32.3 to 39.5 fps, plus a paused sample at
47.8 fps. Stress and displacement readings stayed finite.

## Prompt History

The user supplied the failing console transcript, including:

```text
C:\dev\Powershell\engine\_toy>python drum\_reference.py --live
C:\dev\Powershell\engine\_toy\drum\_reference.py:370: RuntimeWarning: overflow encountered in multiply
<beam-bank>:29: RuntimeWarning: overflow encountered in scalar multiply
<beam-bank>:29: RuntimeWarning: invalid value encountered in scalar subtract
```

The workspace instructions required:

> Before starting nontrivial work anywhere in this tree, check
> [`speaktome/AGENTS/experience_reports/`](speaktome/AGENTS/experience_reports/)
> for prior agents' reports.

The relevant engine_toy/CONTINUATION.md guidance read during the task:

> **one outer dt is handed to every subsystem, and each subdivides that dt
> by its own stiffest term.** Not independent clocks. The subdivision is
> the subsystem's business; the dt is not.
