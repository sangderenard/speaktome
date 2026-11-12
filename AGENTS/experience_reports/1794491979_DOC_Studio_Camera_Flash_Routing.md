# Studio camera flash routing and manual Phong trigger

## Scope and findings

The camera's existing `RoutingGraph` and `RoutingGridView` now drive named Studio and Shop receivers through the browser Studio page. The page relays pointer events to the original Pygame mixer and receives its PNG and graph serialization. The photo request carries that graph. The Spectral worker resolves a bounded Flash pulse plus nonnegative delays on a uniform 0.1 ms control grid and splits exposure frames only where routed light values change. Physical shutter times and sensor weights remain in the original exposure frame/barrier.

The page offers a manual Flash action. It routes a pulse through the same graph to the Phong fixture values and displays receiver gains and model milliseconds. A selectable slowdown makes the 10 ms pulse visible. An empty mixer leaves the existing photo Flash behavior intact.

Focused verification: Spectral camera trigger and worker suites, 18 passed; standalone actual Pygame grid check passed; Turing JavaScript syntax and Shop light/camera checks passed. Existing older scene-packer VM harnesses require unrelated globals and stale emitter counts; they were not used as evidence for this integration. No GPU exposure was run for this report.

## Prompt History

> okay that is proper behavior, we're going to use the spectral analyzer signal graph resolver for our flash bus signals so they'll all have configurable delay and amplitude

> the goal is to lift the n^2 signal mixer gui for the camera bus and all light busses

> Marble Studio browser (Recommended)

> we need to give the flash a manual trigger too so people can watch their timing coding int he phong light values

## Next Steps

Publish the Studio web export to Stable, reload the stack through its operator script when no recent exposure runs, verify the public DNS page and authenticated routing endpoint, and inspect the Phong pulse in a browser.
