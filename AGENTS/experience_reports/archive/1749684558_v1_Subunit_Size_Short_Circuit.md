# Subunit Size Short Circuit

## Prompt History
- User complained the previous fix failed and requested a short-circuit when subunit size does not match.

## Overview
Implemented an early exit in `get_changed_subunits` that marks any partial subunit at the frame edge as changed without comparing to the old frame. Updated docs and added a regression test.
