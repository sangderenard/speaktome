# Meowdoku studio size and visual capture receipt

**Date:** 2026-11-07
**Title:** All fountain sizes in Studio with bounded visual capture

## Scope

Worked in the independent Turing repository, isolated checkout `C:\dev\Powershell\turing-studio-particles`, branch `codex/meowdoku-studio-particles`, commit `4c3e3cad`. Main was not edited.

## Findings

- `fountain.py` defines eight sizes from `small` (256) through `10M` (10,000,000), including `256k` (262,144). `fountain_build.py` normally builds each; the deployed `fountain.manifest.json` is the authority for actually available products.
- Studio had exposed only `small` and `4k`, despite the larger simulation products.
- The compute runtime uses `recording.js` with a 192 MiB sliding snapshot window on desktop or 64 MiB on touch devices. A long large-particle run cannot keep its complete physics replay in that window.
- The saved MLA1 loading animation is position only, with a one MiB raw stream limit. Studio allows at most 1,024 saved grains. A 262,144-id simulation can therefore yield a visual sample, not a 262,144-grain trace.

## Work

- Studio lists all eight sizes and disables sizes missing from the deployed manifest. Record refuses a missing product.
- When the requested duration's full snapshots exceed the runtime window, the recorder copies a bounded subset of the same packed compute snapshots at the chosen cadence, up to 96 MiB. It does not retain physics replay for that recording. The existing `filterTape` and MLA1 encoder still produce the saved animation.
- The Studio text and completion statistics distinguish simulated ids from saved grains and identify visual-only capture. Defaults restores automatic size fitting.

## Verification and limits

- Node syntax checks passed for changed browser modules. Node parameter handling and visual capture checks passed; the sample encoded to 722 bytes. `git diff --check` passed.
- The focused Python loading-library pytest could not import Turing through the shared Speaktome venv because Torch failed loading `torch_python.dll` (`WinError 126`). No native compiler build, browser GPU run, page export, live restart, or deployment was performed.
- A browser run on an actual 256k deployed product is still required. WebGPU structural limits and allocation can reject a size on a particular device. The visual tape and one MiB MLA1 limits remain enforced.

## Next Steps

Review and integrate commit `4c3e3cad`; run the focused test under a working Python environment; export the page and test Studio with 256k in a WebGPU browser before live deployment.

## Prompt History

> "on studio, we need options for more intense particle sims that defer to visual record only if they become too many particles to afford tracing, but I bet using just the system already in place we could comfortable record and deploy a 200k sim for the hourglass"

> "go and let the studio page pick all the sizes of sim instead of just small and 4k"

Delegation: "Implement studio options for more intense particle simulations, aiming to support recorded/deployed 200k hourglass simulation using existing Meowdoku mechanisms, with visual-record-only fallback when tracing unaffordable. Use isolated Turing git worktree/branch from current main, do not alter main."
