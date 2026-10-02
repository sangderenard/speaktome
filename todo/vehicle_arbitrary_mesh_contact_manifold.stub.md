# Vehicle arbitrary-mesh contact manifold

Compile a triangle/BVH contact-candidate stage behind
`abstract-ui-contact-surfaces-v1`. It must provide the same normal, separation,
identity, and stable-generation records for platformers, projectiles, vehicle
patches, and rigid bodies. Extend the chassis layer from its current circle/AABB
approximation to an oriented multi-contact manifold without introducing a
terrain- or vehicle-name branch. See
`../AGENTS/experience_reports/1787715187_DOC_General_Contact_Surfaces_And_Vehicle_Diagnostics.md`.
