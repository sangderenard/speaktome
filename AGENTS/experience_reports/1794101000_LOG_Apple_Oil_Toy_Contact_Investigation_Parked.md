# Apple oil-toy contact investigation — parked

**Date:** 2026-11-07

The public iPhone ticket `1794100682-1467606889` and server log are preserved. The failure is specific to the oil toy: the same iPhone runs the other enclosure without escapes, while the toy rapidly accumulates escaped grains and dropped contacts. A desktop RTX 3060 run of the toy did not reproduce it. The reported visual symptom is a force that urgently expels grains from the toy's middle toward its perimeter, overpowering the other interactions. It is **not established to be gravity**.

Concrete leads for the next investigation: Safari/Apple GPU evaluation of the oil toy's composed SDF may invert or misselect a contact normal near an internal surface or seam; a non-finite distance/normal/contact impulse may enter that path; the toy's internal ramps or inlet may have a platform-sensitive contact-side error. The source SDF sampled on both sides of a ramp returned positive distance and opposite normals, which does not rule out a compiled GPU or seam issue. Compare the live iPhone ticket's first divergent step, SDF distance and normal, contact impulse and finite checks against the desktop run. Keep the ticket open and make no physics change from this hypothesis alone.

The separate Safari `ALIASED_POINT_SIZE_RANGE` null query was guarded in Turing commit `8d334394`; that page exception did not explain the force.

## Prompt History

> “also put the apple thing into a log somewhere and park it for right now while you finish my camera requests and then look into the marble mesh, saved marble mesh, not sure if it's working because it just shows color disks like it's not finding anything, are the saves from the site of marbles not persistent? does it not know what to do with them yet?”

> “I would 100% believe apple or safari is inverting just that sdf somehow or there's a non finite it's like urgently just wanting to not be in the middle”
