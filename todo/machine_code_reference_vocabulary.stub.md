# Extend the machine-code reference vocabulary

Continue the direct decoder and program graph recorded in
`AGENTS/experience_reports/1785723650_DOC_Direct_X86_Vocabulary_To_SSA.md`.
The owning path is PE bytes to machine vocabulary tokens to the token-ID
MultiDiGraph in the precompiler meta package. Repository SSA is an optional
derived projection, not the owning program IR. Preserve exact widths,
fail-closed reporting, and machine-byte provenance; do not restore objdump or
introduce a new SSA-to-C backend.

The one-function region raiser now has a whole-region diagnostic vocabulary
audit. Keep its distinction sharp: only the initial strict decode is proven;
post-gap bytewise matches are candidates used to inventory vocabulary holes.
Any diagnostic gap must continue to suppress SSA. The current 212-byte
`cmd.exe` target census reports 118 candidate-covered bytes and 94 missing
bytes across 9 gaps.

Resolved by the earlier convergence pass: the same region now has 49
strictly decoded instructions, 212/212 proven bytes, zero gaps, and a complete
five-block SSA CFG. Remaining work in this stub concerns generalization beyond
this bounded function and installing the tensor read head, not `cmd.exe` RVA
`0x19584` coverage.

The configurable AbstractTensor read head now exists but is not yet installed
as the scalar decoder's execution engine. First add emitted-instruction
materialization/provenance and a differential verification corpus, then switch
the region raiser to it. Extend field accumulation beyond signed 32-bit
displacements carefully: 64-bit immediates are retained as bit patterns until
the tensor dtype contract includes an explicit unsigned representation.

Current whole-program census for `C:\Windows\System32\cmd.exe`: all 751
`.pdata` runtime-function entries have zero reachable vocabulary failures;
46,834 reachable instructions and 189,493 bytes are proven. The vocabulary has
219 handwritten encoding forms, including distinct XMM and legacy-high-byte
operand identities. `.text` contains
200,704 raw bytes, of which `.pdata` describes 190,979. The remaining 9,725
bytes are 603 explicit unclassified executable graph ranges. Continue cycling
the vocabulary only when new reachable failures appear. There are also 596
unreached ranges totaling 1,486 bytes inside runtime-function extents after
global direct-target fixed-point propagation. Now
classify/discover those ranges from indirect control targets,
exports, entrypoint, imports/thunks, unwind metadata, and padding rules.

After the complete machine graph is authoritative, project its control,
state, data, and call relationships into the existing `ProcessGraph` and
`ControlProgram`/`RegionCode` abstractions. Feed that projection to the existing
`render_c_shell`/`compile_cffi_shell` whole-program AOT assembler. Do not use
the scalar numeric C emitter as if it were the program compiler.

The byte-complete structure graph and fail-closed execution orchestrator now
exist. Next enrich PE inspection with data-directory/import/export/relocation
regions, unwind-handler entrypoints, padding classifications, and indirect
jump-table targets. Emulation should grow by registering semantic-token effect
handlers over explicit register, flags, address-space, and versioned-memory
state; do not equate successful decoding with implemented execution semantics.
