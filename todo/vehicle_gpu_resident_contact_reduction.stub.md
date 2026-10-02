# Vehicle GPU-resident contact reduction

The Living Data Map vehicle kernel currently reads four wheel records back to
JavaScript for chassis force/torque reduction. Add a second compiled WebGPU stage
that keeps the contact outputs GPU-resident, reduces them to chassis inputs, and
joins the existing 120 Hz worker tick barrier before the Wasm/world snapshot is
published. Preserve the page-WebGPU bridge and Wasm fallback for environments
without worker WebGPU. See
`../AGENTS/experience_reports/1787713592_DOC_Living_Data_Map_Vehicle_WebGPU_Physics.md`.
