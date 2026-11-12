# Studio camera Flash modulators

## Findings and implementation

The prior Studio bus exposes the original Spectral N×N `RoutingGraph` and Pygame mixer. This pass adds three named `mod:1`–`mod:3` source nodes. Camera Flash rising edges trigger those sources; each can restart its waveform and ADSR envelope multiple times at a configured spacing. They use Spectral's existing `ConstantPhasePath`, sine, warped or harmonic waveform manifold, linear or piecewise ramp field, and `ADSREnvelope`. The generated samples enter the existing `GraphSolver`, so bus gain, phase, delay and fan-out remain authoritative. Wave frequency, phase, tension, amplitude, attack, decay, sustain level, hold time, release, count and spacing are bounded in the Studio photo contract.

The original routing matrix accepts Shift from browser forwarded events, including horizontal scrolling and fine Delay drag. Saved earlier graphs gain the three new nodes without discarding their existing routes. The manual Phong flash and timed worker both use the same source settings.

## Verification

Focused Spectral tests cover phase, tension, repeats, delayed routing, exposure interval gains and the actual Pygame grid. Focused Turing tests cover saved photo request admission. No GPU exposure was run in this report.

## Prompt History

> provide a few signal source generators too so users can use waves with different waveshape and tension, phase, frequency, a few ramp generators with basic configuration, make sure multiple triggers are supported by those signal generators in between camera on and the lighting, you can make the generators modulators on the camera signal, and we need to add sustain identical to the delay panel in use, has an ASDR envelope

## Next Steps

Publish the Studio page to Stable through the existing exporter and restart with the official operator script after its exposure guard passes. Verify the DNS page and route.
