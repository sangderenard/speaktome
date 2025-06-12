# Backend Hooks Added

## Overview
Implemented or stubbed missing tensor backend methods based on the `AbstractTensor` interface. Added `_apply_operator__` aliases and `repeat_` stubs across non-Torch backends and included save/load/to_dtype placeholders for the OpenGL backend.

## Prompt History
- "Check the function list in abstract tensor and then go through all the back ends and make them all conformant with the planned functions for each back end, stub out anything for which nothing has been defined"
- "Pure python is not the authority undo your changes and only use abstraction as the standard do not under any circumstances trust any backend as a design authority please"

