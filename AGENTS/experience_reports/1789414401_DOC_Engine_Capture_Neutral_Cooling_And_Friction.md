# Engine capture neutral, cooling and friction

## Findings and changes

The default union's Camry profile (seed 3747, 256 named samples per fuel,
104 randomized rows) initialized a neutral output shaft at
78,539,816,339.74483 rad/s through `omega / max(ratio, 1e-9)`. The regression
failed at that initialization before repair. Both initialization sites now
preserve load speed when disconnected. Automatic transmission oxidation
compares log damage against remaining life, avoiding exponential overflow
without capping temperature or changing the damage law.

Following the user's cooling clarification, the demo loader now selects the
real solver's `EngineTestStand` subclass. External reservoir/pump/heat
rejection equipment connects to an existing coolant or oil circuit when its
own pump or active exchanger is absent. Air-cooled machines without liquid
circuits are left as built. The finite external inventory is included in
training snapshots and deep copies; the engine graph/spec remains untouched.
The existing station's PlatformCoolantRuntime supplies the thermal exchange.

Following the friction clarification, drivetrain bearings, clutches and
rolling contacts use impulses capped by the connected inertias, preventing
friction-only slip reversal and numerical energy creation. Exact dissipated
energy is recorded by contact and delivered to an unambiguous thermal circuit
or retained in an unrejected energy ledger. Crank reaction is averaged over
all substeps instead of publishing only the final reaction.

The graph has some bearing edges with radius but no drag coefficients.
Those coefficients were not invented. The contact heat ledger is not a
complete temperature/wear model, and aggregate engine FMEP remains in place.
No claim of complete per-part friction calibration or 30-epoch live training
is made. Existing unrelated working changes were preserved; no commit made.

## Validation

- Original neutral regression: failed with the shaft speed above (195.14 s).
- Overflow correction and neutral initialization: full capture file passed
  10 tests in 254.25 s.
- External cooling/station tests: 8 passed in 9.97 s.
- Capture/live tests after adding stand support: 9 passed, one expensive
  neutral regression deselected, in 58.67 s.
- Final friction, stand and oxidation unit group: 16 passed in 11.15 s.
- Final full capture suite after all changes: 11 passed in 250.73 s,
  including the exact Camry allocation/seed, live shadow, external inventory,
  and heterogeneous union checks. All launched test processes are terminal.
- Guestbook validator reports all filenames conform. Targeted diff checks
  report no whitespace errors. No full all-profile 30-epoch run was made.

## Prompt History

The initial user supplied the command:

> python -m src.common.tensors.abstract_nn.demo_engine_dendrite_live --epochs 30 --samples-per-fuel 256 --batch-size 16 --branches 2 --learning-rate 0.003

and a traceback ending at automatic_transmission.py's oxidation power with:

> OverflowError: (34, 'Result too large')

Subsequent instructions, verbatim:

> if you have an engine doesn't have a radiator or coolant pump, etc. the test should provide an apropriate reservoir and pump to supply it I think... I don't want to force every machine to have a coolant pump and radiator just because they need cooling

> can we fix that by taking friction per part more seriously?
