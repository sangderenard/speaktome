# C GPT Skeleton

## Overview
Initial stub for a C-based GPT model with vendored dependencies.

## Prompts
- "create code stubs and skeleton for a c version of gpt in speaktome/tensors/models..."

## Steps Taken
1. Added `speaktome/tensors/models/c_gpt` with C and Python stubs.
2. Created `third_party` directory with placeholder dependencies.
3. Wrote a basic Zig build file to fetch BLIS if missing.

## Observed Behaviour
Compilation not yet executed; stubs compile in cffi during tests.

## Lessons Learned
Vendoring ensures offline builds; Zig fallback is convenient.

## Next Steps
Implement actual attention layers and integrate real libraries.
