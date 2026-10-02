# Documentation Report

**Date:** 2026-08-26
**Title:** Living Data Map mega portal mode

## Overview

Added a second, vehicle-scale mode to the placement/portal tool while retaining
the existing person-scale mode. Mega and standard portals form independent
probabilistic edge classes so a large entrance cannot select a small exit.

## Changes

- Added persisted `standard` and `mega` modes to the placement tool. The shared
  mode control and `M` key cycle them.
- Published mode profiles in the portal contract. Mega scales aperture radius,
  tube throat, and cubic control handles by four and records `vehicle` as its
  aperture class.
- Stored tool mode, aperture class/scale, throat radius, and handle scale on
  every newly placed portal splat. Legacy splats default to standard/person.
- Restricted each IN node's Gaussian OUT distribution to matching aperture
  classes before normalization.
- Made the trumpet-radius function interpolate size-aware throat radii while
  retaining the endpoint invariant: the first and final tube rings equal their
  corresponding portal-circle radii.
- Regenerated the page and synchronized the static demo.

## Verification

- Python source compilation succeeded.
- The direct mega placement contract check passed.
- Generated inline JavaScript parsed in Node.
- Numerical Node checks confirmed standard and mega trumpet endpoints equal
  their portal outlines and the mega midpoint throat equals the authored 4x
  throat radius.
- Generator and static-site artifacts have identical SHA-256 hashes.

## Concurrency Note

Another compile published the same generated artifact while the first static
copy/hash check was running. The newer combined generator output contained the
mega markers and was copied to the static demo once; no compiler process was
interrupted or modified.

## Prompt History

> can you put a mode 2 in for the portal gun that's vehicle sized, just, mega sized, everything in larger proportion,t he tube, and can you make sure you gave us the trumpet flares on the ends to meet the portal outline

> Your are compiling at the same time as someone
