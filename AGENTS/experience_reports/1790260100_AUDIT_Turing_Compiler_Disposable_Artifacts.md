# Turing compiler disposable-artifact audit

**Date:** 2026-09-24
**Title:** Turing compiler disposable build, image, and log inventory

## Scope

Read-only audit of `C:\dev\Powershell\turing`, focused on compiler build
products, generated images, logs, and caches. No files were deleted or moved.
Scripts and preserved source material found under `build/` were explicitly
excluded from automatic-cleanup recommendations.

## Methodology

- Read the workspace, Turing, compiler, and guestbook agent guidance plus the
  prior reports that established `build/`, `.turing-cache/`, and `artifacts/`
  conventions.
- Measured file counts and allocated byte totals by top-level scope, extension,
  and repeated build family.
- Compared log and media paths against references in tracked Markdown and the
  shared experience reports.
- Inspected tracked media separately from ignored output.
- Checked for active Python processes and files modified after the oldest
  active process began at 2026-09-24 08:15:35. Those files are protected from
  cleanup decisions in this audit.

## Detailed Observations

### Main stores

| Store | Files | Size | Observed role |
|---|---:|---:|---|
| `build/` | 62,688 | 32,933.62 MiB | Mixed generated artifacts, diagnostics, captured compiler checkpoints, emitted source, and misplaced scripts |
| `.turing-cache/` | 2,544 | 4,842.75 MiB | Rebuildable compiler/runtime cache; dominated by 2,569.59 MiB of `.pkl` and 1,807.69 MiB of `.pickle` |
| `artifacts/` | 1,813 at first measurement | 387.86 MiB | Identity logs plus live LLVM pieces; the identity-log count increased during the audit because a compiler process was active |
| ignored root log/media output | 523 | 587.21 MiB | Mostly `.codex-*`, `.probe_*`, `native-trace.log`, AVI, and generated PNG/JPG output |

`build/` is dominated by 22,649.61 MiB of `.pkl`, 2,406.20 MiB of
`.log`, 1,791.32 MiB of `.pdb`, 1,362.61 MiB of extensionless output,
883.14 MiB of `.bin`, 655.29 MiB of generated `.c`, 437.99 MiB of
`.tflite`, 411.95 MiB of `.dll`, and 229.97 MiB of `.exe`.

The largest repeated compiler families are:

| Family | Files | Size |
|---|---:|---:|
| `patch_sequence_*` | 923 | 12,973.28 MiB |
| `validator_*` | 105 | 4,113.21 MiB |
| compiler bootstrap catalogues | 38,172 | 2,993.36 MiB |
| `managed_dt_*` | 287 | 2,812.73 MiB |
| native voxel-fluid builds | 157 | 1,434.71 MiB |
| `vehicle_validator_*` | 330 | 1,276.00 MiB |
| Chrome profiles/servers inside `build/` | 7,406 | 1,222.17 MiB |

### Logs and media

Tracked Markdown and experience reports name 77 build/log/media paths. Of
those, 73 still exist and occupy 345.59 MiB. The bulk is the documented
`artifacts/compiler_evidence/patch_sequence_replay_v145-trace-o0/native-trace.log` at about
344 MiB. Documentation also names two identity logs and several compiler
frontier logs. Removing those files without updating the documents would
break their evidence links.

Measured unreferenced output:

| Scope | Kind | Unreferenced files | Size |
|---|---|---:|---:|
| `build/` | logs | 4,458 | 2,061.09 MiB |
| `build/` | media | 1,508 | 23.06 MiB |
| `artifacts/` | identity logs | 1,669 at measurement | 241.69 MiB |
| repository root | logs | 445 | 580.92 MiB |
| repository root | media | 78 | 6.29 MiB |
| `.turing-cache/` | media | 12 | 4.40 MiB |

This is approximately 2,917 MiB of unreferenced logs and media. The count is
dynamic because the active compiler run continues to emit identity logs.

Twenty-eight media files totaling 4.035 MiB are tracked by Git. They include
12 PNGs under `docs/generated/`, five chamber PNGs under `examples/`, six
root-level AVI files, one compiler-convergence evidence image, and three
intentional `src/rendering/ascii_diff` reference images. The ascii-diff images
are explicitly protected by `.gitignore` commentary. The other tracked media
has little or no textual reference, but is not classified as automatically
disposable because tracking indicates a deliberate historical decision.

### Protected material under build

- There are 165 scripts directly under `build/`, totaling 5.671 MiB. Their
  names include compiler probes, trace tools, patch drivers, and replay tools.
  They are source-like material accidentally living in an ignored directory
  and must not be removed by a broad `build/` cleanup.
- Thousands of additional `.py` files under build products are emitted source
  snapshots or copied runtimes. Directory location alone does not distinguish
  those from the 165 hand-authored scripts.
- `artifacts/worktree_preservation/20260919/` contains six patch/source files
  totaling about 0.06 MiB. A prior report explicitly identifies these as
  preserved worktree state. They are not disposable.
- `build/native-voxel-fluid-full-physics-probe/` contains expensive compiler
  checkpoints that a prior handoff explicitly requires for resuming the real
  fluid compiler investigation. Do not remove it as generic build output.
- `artifacts/llvm_pieces*` occupies about 146 MiB. `.gitignore` states that
  these generated pieces are live inputs to `examples/llvm_dt_system.py` and
  should be kept locally until deliberately rebuilt.
- Twelve identity logs (0.69 MiB) and one compiler catalogue file were written
  after an active Python compiler process began. All current-run output is
  protected.

### Peripheral disposable state

Twenty-one ignored `.tmp-chrome-*` profiles occupy roughly 1.37 GiB. They are
not primarily compiler artifacts, but `.gitignore` explicitly describes them
as temporary browser profiles containing credentials and trust tokens. They
are cleanup candidates only after confirming no browser automation session is
using them.

## Analysis

The repository has a naming problem rather than a single disposable directory.
`build/` combines generated binaries, multi-gigabyte compiler checkpoints,
documented diagnostic evidence, and hand-authored debugging programs. A broad
directory deletion would violate both the user's instruction and the recorded
handoffs.

The safest immediate yield is the approximately 2.85 GiB of unreferenced
ignored logs and media outside active-run files. Larger savings require
lineage-aware retention. In particular, `patch_sequence_*` accounts for nearly
13 GiB and contains many sequential variants, but at least the documented
v145 trace and any source/probe drivers must be retained. `.turing-cache/` is
explicitly rebuildable but saves expensive compiler work, so deleting it is a
space-for-time decision rather than routine hygiene.

## Recommendations

1. First-pass cleanup: remove only unreferenced ignored logs and media older
   than the current run, with an allowlist for the 73 documented evidence
   files. Expected recovery is about 2.85 GiB.
2. Relocate the 165 direct `build/*.py` scripts into a tracked compiler-tools
   or documented probe location before any directory-level cleanup.
3. For repeated `patch_sequence_*`, validator, and managed-dt families, retain
   the final documented artifact, the minimal replay inputs, and one failure
   predecessor; remove redundant native binaries, symbols, and intermediate
   pickles only after verifying the retained replay path.
4. Keep `worktree_preservation_20260919`, the native voxel-fluid checkpoint,
   active-run output, and LLVM pieces unless their owning handoffs are closed.
5. Review the 28 tracked media files separately. Keep the three ascii-diff
   fixtures and the compiler-convergence evidence image. Untrack other media
   only after confirming it is not a published example or intentional visual
   receipt.
6. Add a retention helper only after the scripts are relocated. It should use
   explicit allowlists and age/reference checks, never `Remove-Item build -Recurse`.

## Prompt History

> do an audit of disposable build artifacts (but not scripts erroneously placed in build) images and logs primarily focused on the turing compiler

Applicable workspace instruction:

> Before starting nontrivial work anywhere in this tree, check `speaktome/AGENTS/experience_reports/` for prior agents' reports.

Applicable Turing instruction:

> When something seems wrong, the system is right until proven otherwise. Read more before editing.

## 2026-09-24 storage-boundary follow-up

The user clarified that `build/` is not persistent storage. The protected
material identified above was moved without deleting it:

- 165 direct build scripts moved to `tools/compiler_probes/`, with repository
  root and sibling-script references adjusted for the new location;
- 71 documentation-linked logs and images (345.432 MiB) moved to
  `artifacts/compiler_evidence/`, and their tracked references were updated;
- the resumable voxel-fluid checkpoint moved to
  `artifacts/compiler_checkpoints/native-voxel-fluid-full-physics-probe/`;
- preserved worktree state moved to `artifacts/worktree_preservation/20260919/`.

Tracked `README.md` files now define the transient `build/` boundary and the
retention roles of `artifacts/` and `tools/compiler_probes/`. Active compiler
outputs and ordinary rebuildable artifacts remain in `build/`.

Follow-up prompt:

> let's move things that are being saved out of build and leave a note as to why they were moved, build is not meant for persistent storage

## 2026-09-24 disposable-material removal

After the retention move, the user requested removal of all remaining
disposable material. Approximately 37 GiB was removed, including:

- all stale `build/` outputs except a currently loaded bootstrap DLL;
- the complete 4.84 GiB `.turing-cache/`;
- 1,657 old uncited identity logs;
- ignored root logs, images, videos, PID files, generated modules, and an old
  compiler-state zip;
- stale native-build directories, browser profiles, Python caches, law caches,
  repository caches, precision benchmark output, generated site output, and
  nested Wooly build directories;
- 24 tracked generated media files and one accidentally tracked cache record.

The four intentionally tracked visual fixtures remain: the compiler-convergence
evidence image and the three `src/rendering/ascii_diff` reference images.
All 73 promoted evidence references resolve, and all 165 relocated Python
probe scripts still parse.

Active processes were not interrupted. Their live residue remains temporarily:
one loaded bootstrap DLL under `build/`, one loaded CFFI cache binary, the live
`woodshop-current-6.log`, current-run identity logs, and a small `repo-cache/`
string table recreated during cleanup. These are disposable after their owning
processes exit.

Cleanup prompt:

> when you've completed that remove all the extra disposable material
