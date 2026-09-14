# Turing open off-road playground

**Date:** 2026-08-27
**Title:** Localized multi-zone truck terrain and outer-yard spawn

## Overview

Removed sampled height variation from the document courtyard and replaced the
recent second extruded courtyard/hoop with an open outer driving yard. A bounded
81 x 81 play surface now contains a hill climb, rock crawl, whoops, dry creek
bed, and two sampled ramps. Two analytic practice ramps occupy the surrounding
flat apron, and Springtail's authored first spawn is on that apron beside the
play area.

## Steps Taken

- Replaced `sampled_hoop_height_field` with a multi-zone off-road generator.
- Removed the inner courtyard's `sampled_mud_oval_height_field` instance.
- Removed the 90 m extruded outer courtyard box while retaining an open driving
  area as world-floor metadata.
- Reduced the outer sampled terrain from 129 x 129 over nearly the whole yard to
  81 x 81 over a 56 m bounded play patch.
- Made authored-world-pose initialization honor the vehicle pose rather than
  silently substituting the player's position.
- Added focused terrain/model regressions and checked Python and JavaScript
  syntax.

## Observed Behaviour

- The focused off-road generator test passed.
- The generated page-model test passed all new terrain, no-inner-height-field,
  no-second-courtyard, ramp, and spawn assertions. It later encountered an
  unrelated stale assertion expecting the old `dispatchVehicleContacts` worker
  name; the checked-in worker has already moved to the unified GPU graph.
- Direct Node syntax validation of `DIV_MAP_JAVASCRIPT` passed.
- `git diff --check` reported only the repository's expected LF/CRLF warnings.

## Lessons Learned

The open yard and the sampled play surface are separate concepts. Representing
the yard as another courtyard caused a giant visible boundary box, while making
the terrain fill that yard needlessly multiplied mesh and contact data. A small
depth-map island plus ordinary world floor provides both performance and a
legible staging area.

## Next Steps

- Add a locked-default vehicle control/program variant whose constants can be
  aggressively folded, retaining the parametric artifact and switching to it
  only when a user changes an unlocked parameter.
- Browser-drive the course and tune obstacle spacing/amplitude based on actual
  Springtail wheelbase, clearance, and crawl gearing.

## Prompt History

> an agent started refusing work after making several deliberate errors. I need you to help clean them. We need to remove any height variation texture from the inner courtyard. the outer courtyard has a box, it needs to be eliminated. the height textured area needs to have WAY more variety for playing with the truck, which should now spawn out in the larger courtyard next to the play area. terrain resolution or local-to-entities only floor tracking may fix performance issues with large depth textures

> at the very least I want a hill climb, rock crawl, woopdiedoos, dry creek bed, and some ramps on the flat space outside the play area and in

> just a heads up, if we put optional locks on controls and for instance, locked default everything, we could bake that engine without parameters, it will optimize LIKE CRAZY and then we can keep both the default and the parametric in the system and use the default until a parameter changes
