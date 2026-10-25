# Station first-shot joint coupling follow-up

Repair the first post-shot frame at the actual cause recorded in
`1789332010_AUDIT_Station_First_Shot_Frame_Freeze.md`: the active
`turret.equilibrator` is declared as a linear directional spring-damper but is
silently loaded as a generic quadratic gas-over-oil strut, after which explicit
joint/beam coupling diverges and requests 4,559,679 internal iterations.

Acceptance requires the real SPACE event path to present the following frame,
with finite beam/joint state and a bounded honest substep count.  Do not alter
or replace the projectile carrier to make this test pass.
