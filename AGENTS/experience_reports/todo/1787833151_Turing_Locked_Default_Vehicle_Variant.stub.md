# Turing locked-default vehicle variant

Add optional locks to vehicle controls/parameters. When every relevant value is
locked at its default, compile and cache a constant-specialized engine/vehicle
artifact so ordinary compiler optimization can fold away parameter plumbing.
Retain the current parametric artifact and switch to it at a fixed-tick boundary
when a parameter is unlocked or changed. Preserve the single worker/state owner
and transfer the complete resident state across the artifact switch.

Source: `1787833151_DOC_Turing_Open_Offroad_Playground.md`.
