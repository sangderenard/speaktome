# Account-scoped Studio photo assets

## Prompt History

> make sure when users make assets from rest/mcp the file is placed and lives only in their user area as an asset tracked to them, so if they go into studio, for instance, the still image will be there, in a section like the one for fountain, maybe the loading screen becomes a collapsible option to deal with because it could be any image, animation, or slideshow, or program

## Findings and work

The REST/MCP `show` picture route already commits through `studio_pictures` and `PhotoJobs.start`, with a trusted `account:<id>` owner and an account asset claim. The Studio job endpoint already filters by that identity. The gap was physical storage: new photos still went to the shared `content/marble_photos/<digest>` directory, and Studio did not preview the saved result.

New admitted account-owned photos now store their entire job directory in `content/users/account/<id>/assets/marble_photos/<digest>`. Browser-owned jobs have the parallel `content/users/browser/<token-hash>/assets/marble_photos/<digest>` path. The existing job-owner index selects the path; request bodies and digests do not select an identity. Old job directories remain readable in their original location. Live control, field, retouch, animation expiry and job listing use the same resolved directory. State migration includes `content/users`.

Studio's collapsible asset/job section previews the selected still and links to its orbit animation, while keeping the Loading Studio link for existing loading animations. The job list now carries the image variant and animation fields required by that view.

## Verification

Focused Turing tests: 56 passed, 1 skipped, 2 deselected. The two deselected timed-light tests require `yaml`, absent from the Turing environment, and were not changed for this work. `node --check` on `marble_jobs.js` and `git diff --check` passed.

## Limit

Existing photo directories were not moved while the server may still be using them. Account-scoped placement applies to jobs created after the server loads this source. Loading screen asset types beyond its existing recordings were not invented here.
