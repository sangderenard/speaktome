# Camera kernel pipeline continuation

See `../1794546802_DOC_Nodus_Camera_Kernel_Pipeline_Adoption.md` and
`C:/dev/Powershell/nodus/docs/KERNEL_PIPELINE_ADOPTION_2026-11-13.md`.
The one-ray sensor raygen→T1→T2 mapped-address path now passes. Extend it with
T3 and preserve original spectral, sensor, and scene buffers. Implement the graphics render-target tensor bridge
before claiming the Phong raster pass is a state-library pipeline component.
