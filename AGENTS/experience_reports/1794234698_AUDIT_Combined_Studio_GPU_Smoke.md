# Combined Studio GPU smoke

The existing production exposure completed before isolated GPU work began. The exact saved payload was copied unchanged. Production sceneForPhoto and its pose, room, table and backdrop builders created a 2,064-triangle scene with the staged material catalogue. It included all ten lights, both tabletop snoots, gridded underlight, transparent table and six concentric mats. The saved 10 mm radius preserves tabletop contact and the current Studio camera uses the actual 360/7-degree FOV.

The canonical 128x128 worker completed 16 epochs at four bounces in 0.578 seconds of exposure. All ten source identities participated in the 4,096 baked rays. Raw sensor HDR, profiled linear sRGB, metadata and PNG were finite and matched exact recomputation. The short image remains noisy; this is not a quality or convergence approval.

A separate instrumented one-epoch probe retained the actual 200-bounce shaders while limiting work to 128 initial plus 32 pumped forward rays and one 16x16 camera strip (256 paths). Both GPU uMaxBounces values read back 200. Pump and sensor dispatch took about 2.5 ms and 6.25 ms. It completed with finite HDR and matching profile/display. Full-resolution 200-bounce performance and general watchdog safety remain unmeasured.

Important bounded-probe detail: setting rows_per_frame=1 in row mode does not limit the shader to one row because its workgroups are 16 rows. The successful probe used a 16x16 sensor, rows_per_frame=16, and one strip. The worker's nominal status counters are not instrumented; true counts are in dispatch_receipt.json.

Artifacts: C:/dev/Powershell/turing/build/dot-combined-studio-20261109/README.md, input_receipt.json, result_receipt.json and high200/dispatch_receipt.json. No implementation edits, live API actions, exports, saved-object mutations or restart. Fresh process enumeration after QA found no photo/probe worker.

## Prompt History

Parent assignment: "Verify combined current Studio scene end-to-end with bounded isolated GPU smoke ONLY when production photo worker exits."

Parent constraint: "Own new QA scripts/tests/artifacts only, no implementation source."

Parent safety clarification: "Do not simply reuse128px default8192 jobs at200 if unsure. If safely reducing unsupported, report CPU forwarding proof plus hardware timing unmeasured rather than risking watchdog."
