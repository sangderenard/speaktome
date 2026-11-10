# Marble RGB film response contract

## Findings and changes

Spectral marble photos now accept validated independent R/G/B Gaussian centers
in nm and sigmas in log2-frequency octaves, preserving em_rgb_fixed3's current
650/550/450 nm and 0.06-octave defaults. A per-job deep copy drives both forward
transport and the camera response. The normalized response is recorded in job
status, effective_config.json, and finished HDR metadata. Monochrome behavior
is unchanged. This is a custom sensor response, not a calibrated color profile
or display gamma control.

The complete contract, actual uniform paths, and verification limits are in
C:\dev\Powershell\spectral-analyzer\MARBLE_FILM_RESPONSE_HANDOFF_2026-11-09.md.
Forty-six focused tests passed in 3.40 seconds. No live GPU exposure, server
restart, deployment, compiler build, or full suite was run. Existing dirty
worker changes were backed up and preserved.

## Prompt History

> if there are color profile or correction conventions we should follow them, but we do desperately need to adjust the film layer responsiveness

## Fixed transport integration follow-up

The copied film now opts into the fixed 650/550/450 nm transport basis through
transport_wavelengths_nm. Both saved marble and fixture mesh imports opt into
spectral_emission_rgb for RGB transport. The separately implemented color-path
correction keeps material optics independent of custom film controls and
applies the Gaussian once at final sensor accumulation. Default sensitivity
parameters remain exact; corrected material transport can change output pixels.
Forty-seven film/status tests passed in 3.63 seconds after this integration.
