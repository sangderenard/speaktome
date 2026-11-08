# Marble course, capture shelf, and case lights

## Work performed

- In canonical `C:/dev/Powershell/turing`, committed `a7c4d1b6` for Marble Madness backplate-normal gravity, visible X/Y tilt controls, close simulated-marble camera, a physical case-motion view, plate clearance, a backlit plate toggle, follow spotlight toggle, emissive exterior controls, and arcade lights mounted to the case.
- Added compact visual MLA1 capture from Fountain and Fountain3D to a three-day temporary server shelf; Studio separates temporary entries and lets the owner graduate them before activation. Fountain3D's current MLA1 capture projects X/Y and says so in the interface.
- Exported static pages to canonical live 8766 and scratch dev 8768, restarted both Python servers from canonical Turing with WebGPU products ready and Wasm disabled. Preserved other dirty compiler and chess files.

## Verification and limits

- `pytest` on the library and Marble Madness suites: 20 passed. Node compact-capture and gravity checks passed. JavaScript syntax and Git diff checks passed. Both servers report `all_settled=true`, and the served pages contain the new controls.
- Browser visual QA of the close follow view, marble case tilt, light mounting, and a real candidate promotion flow remains. The in-app browser control tool returned `browser.user is unavailable` when claiming the dev tab, so only HTTP and code-path checks were possible here.
- The 3D law already computes angular velocity, slip/friction torque, rolling resistance, and spherical moment of inertia. Its 20-byte display record omits spin. A separate active task is joining that law state to the marble mesh, with no new law or Wasm build planned.

## Prompt History

> "last detail, uh, marble madness needs the follow camera from the studio which I can tell you, didn't follow close enough it's got to be marble perspective scale"

> "also those tilt controls shouldn't be invisible we need the follot the case rotation option from the fountain game controls and we also need the arcade lights to be tied to plate location so on games that move the plate the lights stay mounted right"

> "make the follow spot able to be toggled and also let the emissive properties of the marble base material be configurable in the marble madness game so users can choose to glow instead of be lit or do both"

> "the particle sim needs rotation state for marble madness to work, it has friction qualities, it even has roughness I think which we can use to scuff marbles from forces, very cool, but, we need that frictional rolling that I think the 3d sim actually has we just need to link to the mesh representation of the marble so it's not just slip sliding, right?"

> "also maybe we need materials to use their roughness maybe for friction and have sane baselines? maybe the marble is just very slippery in the test room you know? we have to figure it out"

## Next steps

- Connect packed angular state to modeled marble orientation and measure contact-driven roll.
- Browser-check the actual scene and candidate shelf. Distinguish Phong surface roughness from the law's physical rolling resistance before mapping material presets.
