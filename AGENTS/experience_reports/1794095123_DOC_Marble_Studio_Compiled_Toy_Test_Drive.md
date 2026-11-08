# Marble Studio compiled toy test drive

## Work

Marble Studio's independent `/api/marble_preview_spec` now uses the same `MarbleMadness` 4k SDF toy product as the game. Its real chamber mesh supplies 2,162 triangles, including the toy's ramps and pegs. Distinct catalogue finishes make contact obstacles visible; they do not claim separate physical wall friction. The visual preview remains selectable.

A held on-stage D-pad maps screen directions through camera yaw into the law's X/Y gravity components while retaining the backplate-normal Z component. Input uses `ComputeRuntime.setGravity` and the product's existing runtime inputs. Pressing a direction turns off automatic case motion for direct driving. The authored marble mesh now takes the runtime's physical angular quaternion, including in close follow camera view.

## Verification

`node --check` on Studio, the focused marble preview Node check, and `git diff --check` passed. Direct construction of `MarbleMadness().state()["compute"]` returned the 4k toy manifest and a 2,162 triangle chamber mesh. Browser visual and interactive drive checks remain for the parent integration after static export and Python server restart. No Wasm build ran.

## Prompt History

> "in marble studio lets test drive marble madness engine a little by giving the marble camera oriented dpad center of gravity nudge controls and put a few gameified surfaces in the demo cube"

## Next Steps

Export, restart the server, then hold each D-pad direction in the real browser to check camera-relative motion and the obstacle finishes.

## Concurrent marbles follow-up

Studio now offers 1–16 identical copies of the current authored marble. The server uses the law's existing age start column and a 0.16 s emission interval to bound eligible IDs. The unchanged 4k compiled product still computes all IDs, but only the selected IDs may emit. GPU and Wasm snapshot routes carry each selected ID's physical angular velocity; `rotationFor(id)` integrates its recorded quaternion. Random saved designs are explicitly disabled because arbitrary saved mechanical properties cannot be assigned faithfully through the current four physical species. The existing per-ID emoji mesh selection handles visuals only.

A direct symbolic SDF toy law step with 32 IDs, the first three at age zero and the rest at age −1e9, showed only ID 0 initially. Every ID at or above 3 stayed hidden and at exact zero position. At float32 precision −1e9 cannot advance by the 1/360 s time step, so those IDs remain parked. The existing 8768 process returned its old endpoint before restart; browser validation must use the refreshed server. No Wasm build ran.

> "custom N concurrent other marbles, either identical to current design or random from saved designs"

After restart, check counts 1 and 16, each mesh's physical rolling pose, and that IDs above the selected count never appear. Random saved designs need a verified mapping of each saved radius, density, friction and material geometry into the compiled law.
