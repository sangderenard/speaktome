# LLVM Wooly Willy OpenGL demo

## Scope

Added `C:\dev\Powershell\engine_toy\wooly_willy.py`, a small interactive
Pygame/OpenGL Wooly Willy driven by equations selected from Engine Toy's
honorary equation catalogue.

The selected laws are:

- `eq_F16_9`: radial component of the magnetic dipole field.
- `eq_F16_10`: polar component of the magnetic dipole field.
- `eq_F16_15`: aligned dipole force.

## Compiler route

The demo uses the established `src.compiler.native_law_kernels.law_kernel`
route. The SymPy laws are lowered to SSA, materialized as AbstractTensor
source with the filing batch shape, lowered through the sanctioned compiler
entry, and then compiled to LLVM. There is no parallel numerical
implementation of the magnetic laws in the demo.

For a batch of 1,400 filings the generated entry is
`wooly_willy_magnetic_field_batched__tick`, with arguments
`m, m_1, m_2, mu_0, r, theta` and outputs `B_r, B_theta, F_dd`.
The LLVM artifact is complete, reports no lowering shortfalls, and the
compiler concordance detector reports zero findings.

## Verification

The following checks passed on 2026-09-23:

```text
python wooly_willy.py --probe --filings 64
python wooly_willy.py --filings 64 --frames 45 --hidden --screenshot build/wooly_willy_smoke.png
python wooly_willy.py --filings 1400 --frames 180 --hidden --screenshot build/wooly_willy.png
```

The hidden OpenGL runs exited successfully and produced visually inspected
screenshots. The interactive controls are left mouse to attract/comb, right
mouse to repel/erase, `R` to reset, `S` to shake, and Escape to quit.

## Prompt history

The implementation followed these user directions:

> the way these compile is by going to abstract tensor first to get their shape parameter and then to llvm

> please do so, I was actually thinking if we could make a wooly willy that would be awesome

