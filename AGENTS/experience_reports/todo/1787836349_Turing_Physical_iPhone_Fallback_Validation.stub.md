# Turing physical iPhone fallback validation

Load the generated Living Data Map on a physical iPhone/Safari. Verify that the
HUD reports `resident-wasm-fallback` when worker WebGPU is absent, throttle
produces visible wheel rotation and chassis motion, terrain contact remains
stable, and Safari reports no worker/Wasm exception. Repeat on an iOS 26+ device
with WebGPU available to verify seamless upgrade to `resident-webgpu-graph`.

Source: `1787836349_DOC_Turing_Safari_Vehicle_Wasm_Fallback.md`.
