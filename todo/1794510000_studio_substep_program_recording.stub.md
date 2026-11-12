# Studio light program state recording

Use `recordStudioExposureFrames` and its existing exposure clock to capture
user-authored routed light programs. Save a bounded, account-owned compressed
state sequence with optional changed geometry per slice, PBR emissive power,
and corresponding Phong light state. Play it through Studio's Phong renderer.
REST/MCP and the browser mixer must submit the same authored contract. Show an
actual Phong recording; no hardcoded program or WebM-only save path.
