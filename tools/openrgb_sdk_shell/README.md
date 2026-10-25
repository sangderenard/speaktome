# Direct ASRock HID light-field studio and OpenRGB diagnostic shell

## Native ARGB3 strobe experiment

### Typed Python/C/compiler ABI

The native-effect boundary uses one semantic message rather than a positional
collection of output pointers.  `asrock_hid_abi.py` declares the live ctypes
layout and `native_asrock_hid/asrock_polychrome_hid_abi.h` declares its exact C
counterpart.  Both sides require ABI version 1 and a 16-byte structure; C++
also statically checks every field offset.  The three controller-native color
bytes are a nested structure, so they remain one typed value to Python and to
the compiler.

```python
from asrock_hid_abi import ASRockHIDExternalZoneEffect

effect = ASRockHIDExternalZoneEffect.strobe(speed=0x20)
controller.write_external_zone_effect(effect)
captured = controller.read_external_zone_effect()
print(captured.mode, captured.speed, captured.color_bytes)
```

The structure deliberately omits mutable zone selection and `allzone`: its C
entry points always address Audio/ARGB3 (`0x07`) with `allzone=false`.  The C
library alone translates the semantic object to the physical 65-byte HID
report.  Existing scalar C exports remain as compatibility shims, while the
Python transport uses the structured exports.

This is also the compiler-facing type.  The repository's existing ctypes
interception consumes the live class through `python_bindings` and records
ctypes' actual size, alignment, nested type, and member offsets in its struct
tables.  No separately maintained compiler layout or packet parser is needed.

`openrgb_native_strobe_probe.py` is separate from the host-driven framebuffer
loop.  It configures physical zone `0x07` (`Addressable Header 3/Audio`) once
and leaves the motherboard MCU to run native mode `0x03`.  It never selects
Direct mode, sends framebuffer chunks, changes header/RG-swap configuration,
addresses another zone, or sends commit command `0x12`.
OpenRGB nevertheless annotates this controller as saving on every update, so
the probe treats restoration as mandatory rather than assuming a zone write is
volatile.

The implementation follows OpenRGB's Polychrome USB protocol exactly.  A
zone write is one zero-filled 65-byte HID report followed by one 64-byte
reply:

| Byte | Value |
|---:|---|
| 1 | `0x10` (`SET_ZONE`) |
| 3 | `0x07` (Audio/ARGB3 physical zone) |
| 4 | `0x03` (native Strobe) |
| 5..7 | raw zone color bytes; the probe uses `FF FF FF` |
| 8 | effect speed |
| 9 | `0xFF`, as used by OpenRGB |
| 16 | `0x00` (`allzone=false`) |

The corresponding read uses command `0x11`, with zone `0x07` in byte 3; its
reply supplies zone, mode, color, and speed at bytes 3 through 8.  OpenRGB
defines speed `0xFF` as slowest, `0xE0` as the default, and `0x00` as fastest.
No period, phase, pulse-width, duty-cycle, or direction field is exposed by
this zone-effect packet.  Consequently, optical frequency and ON duration
must be measured rather than inferred from the byte value.

Close OpenRGB, ASRock Polychrome, and the field studio before opening this
exclusive HID path.  Inspecting is read-only:

```powershell
cd C:\dev\Powershell\speaktome\tools\openrgb_sdk_shell
py -3.11 .\openrgb_native_strobe_probe.py
```

Run one allowlisted speed, or step through the complete conservative sequence:

```powershell
py -3.11 .\openrgb_native_strobe_probe.py --speed E0 --apply
py -3.11 .\openrgb_native_strobe_probe.py --sweep --apply
```

The allowlist is `E0 C0 A0 80 60 40 20 10 08 04 02 01 00`.  Before changing
the zone, the probe captures its raw mode/speed/color and writes a recovery
file beside the configuration.  Normal completion, Ctrl+C, and ordinary
errors restore that captured state and remove the recovery file.  If the
process is forcibly killed or the machine loses power, restore the saved state
before another experiment:

```powershell
py -3.11 .\openrgb_native_strobe_probe.py --restore --apply
```

The source establishes that the host performs one configuration transaction;
it does not establish the MCU's optical frequency or duty cycle.  Use a
photodiode/oscilloscope or adequately high-speed camera for the speed table.
For a 2000 RPM fan, compare measured repetition frequency around 33.33 Hz and
measure pulse width independently—the existing direct-lighting completion
rate is not a limit on an effect generated after this one-time transaction.

## Direct hardware transport

The interactive studio no longer uses the OpenRGB server in its frame loop.
`run_field_studio.ps1` builds `native_asrock_hid/asrock_polychrome_hid.cpp`,
closes a competing OpenRGB process, and opens the motherboard controller at
HID `26CE:01A2` directly. The established 245-slot logical layout and the
firmware-facing 313-color padded image are configured in
`openrgb_field_studio_config.json`.

The native transport owns persistent 65-byte reports. Python supplies one
contiguous RGB buffer per update; native code applies the established per-zone
R/G convention, patches changed framebuffer bytes, and performs the configured
HID report program. Frame submission uses a non-blocking, one-frame
latest-wins mailbox: a newly submitted frame replaces an older frame that has
not begun transmission, so obsolete animation states cannot accumulate.
Ordered frame reports are written as one burst and their replies are drained
afterward rather than placing every output behind a reply round trip. Direct
mode and the controller R/G table are set once at startup. Exit still queues a
complete black image and waits for the native worker before closing HID.

The studio status line reports requested simulation rate separately from the
rolling HID completion rate. `sent`, `done`, `drop`, and `fail` are respectively
submitted frames, completed configured hardware transactions, mailbox
replacements, and transport failures. A completed transaction means all HID
replies arrived; it cannot by itself prove the LEDs optically latched.

On the installed ASRock B650M Pro RS WiFi controller, direct-HID observation
showed short peaks near 15 complete frames/second and a sustained rate of
10.0 frames/second within approximately +/-0.1 Hz. Producer rates above that
remain useful for smooth simulation and GUI rendering, but the latest-frame
mailbox samples that evolving field at the controller's achieved cadence.

The lower-left packet bench controls vary three properties independently while
the field continues running:

- `msg stream` sends an `E3` initial report followed by `E4` continuation
  reports. `msg restart` sends only `E3`; `msg continue` sends only `E4`.
- `reports -/+` selects a transmitted prefix from 1 through 18 reports. The
  former 17-report padded stream is only the saved baseline.
- `I/O serial`, `windowed`, and `burst` drain replies after every report,
  after the displayed window, or after the entire selected prefix.

Changing a packet profile resets the rolling-rate observation window, keeping
the displayed throughput comparison local to the selected transaction shape.

Live bench result for this controller: `msg stream`, `I/O burst`, and 14
reports is the minimum useful full-field transaction and the production
default. Thirteen or fewer reports produce no visible update even though 13
reports have enough nominal payload capacity for the 245 logical RGB values;
this demonstrates a controller-side latch threshold rather than a simple byte
coverage requirement. Reports beyond 14 only reduce throughput. Windowed reply
handling matches burst once its acknowledgement window is greater than four
reports; small windows and serial acknowledgement are slower.

Run it from PowerShell without starting OpenRGB:

```powershell
cd C:\dev\Powershell\speaktome\tools\openrgb_sdk_shell
.\run_field_studio.ps1
```

The older SDK shell below remains available as a diagnostic/reference path,
but it is no longer the studio's configured transport.

This directory provides a deliberately small Python interface to an already-running OpenRGB SDK server. It does not start OpenRGB, install a driver, rescan hardware, create a service, change a firewall rule, or write anything on import.

The wrapper exposes the underlying `client` and `device` objects, but its supported write path is intentionally narrow: select a mode explicitly and send one complete controller frame. This is important for the ASRock Polychrome USB driver, whose zone and single-LED SDK methods collapse to one whole-zone color. Its only per-position packet path is the full-device `UpdateLEDs()` operation used by `device.set_colors(...)`.

## Current hardware evidence

- ASRock B650M Pro RS WiFi is reached through USB HID `26CE:01A2`.
- One clean all-zone `Static` `400000` test produced red on the cooler and motherboard.
- One clean all-zone `Static` `202020` test produced white on both. The apparently pale-red motherboard was the colored backing plate seen through white illumination, not an RGB channel error.
- Direct full-frame writes are verified on `Addressable Header 3/Audio`.
- That zone advertises 80 logical positions, but the measured cooler/fan chain uses positions 0 through 7. In the installed, upside-down view the radial map is `0=bottom-left`, `1=left`, `2=top-left`, `3=top`, `4=top-right`, `5=right`, `6=bottom-right`, and `7=bottom`. Increasing positions walk clockwise.
- Logical red appears physically green and logical green appears physically red on this chain, establishing an R/G channel swap. White is unaffected.
- The motherboard `PCB` zone has four independently addressed backside LEDs: position 0 appears above the board, while positions 1 through 3 appear along its right side.
- No GPU is present in the current OpenRGB SDK device inventory; the 245 mapped addresses all belong to the motherboard controller.
- The 80-entry header listing is configured logical capacity, not a physical LED census. Parallel splitter outputs still mirror and cannot be made independent in software.

## Environment

Use the repository's documented environment setup rather than installing into
the system interpreter. `requirements.txt` bounds the maintained SDK client to
`openrgb-python>=0.3.7,<0.4`; it does not install an RGB driver or a background
service. The managed-dt effects additionally use the adjacent `turing/`
checkout. Their eight-band display basis is imported from the adjacent
`spectral-analyzer/` ray tracer, so the light field and tracer use the same
700-to-400 nm band ordering and band-to-linear-RGB response.

## Start exactly one loopback-only OpenRGB server

The supplied `run_field_studio.ps1` and `run_temperature_three_body.ps1`
launchers now perform this automatically. They first probe
`127.0.0.1:6742`; an available SDK server is reused without touching its
process. If the socket is unavailable and no OpenRGB process exists, they
start the reviewed executable below, hidden, with the same isolated
configuration and wait for the SDK socket. An existing OpenRGB process with
no reachable SDK server produces an error instead of creating a competing
second controller owner.

First exit ASRock Polychrome completely and verify that `AsrPolychromeRGB.exe` and old `OpenRGB.exe` instances are absent. Then, in a separate terminal, start the reviewed portable OpenRGB build with the existing isolated configuration:

```powershell
& 'C:\Users\alber\AppData\Local\Temp\OpenRGB-1.0-81bbe18\OpenRGB Windows 64-bit\OpenRGB.exe' `
  --config 'C:\Users\alber\AppData\Local\Temp\OpenRGB-1.0-total-coordination-20261003' `
  --noautoconnect `
  --server `
  --server-host 127.0.0.1 `
  --server-port 6742
```

Pinned OpenRGB 1.0 source confirms that `--server-host` is passed to the server and that its documented default is also `127.0.0.1`. Keep the explicit loopback argument. This wrapper refuses a non-loopback host unless `--allow-remote` is supplied. No firewall rule is required or recommended.

Starting OpenRGB initializes the USB lighting controller once. Leave that one process running while programming; repeated CLI launches repeat initialization and global writes.

## Inventory and dry runs

Inventory sends no lighting update and lists every SDK device, each zone, its
dense law-domain offset, and the controller-provided addresses in that region:

```powershell
.\.venv\Scripts\python.exe .\openrgb_shell.py --device-name 'ASRock B650M Pro RS WiFi' inventory
```

All mutating CLI operations are dry runs unless `--apply` is present:

```powershell
.\.venv\Scripts\python.exe .\openrgb_shell.py --device-name 'ASRock B650M Pro RS WiFi' `
  solid --mode Static --rgb 32 32 32
```

## Minimal one-position mapping test

Use the fresh inventory name/index rather than assuming that Polychrome's `Addressable RGB LED3` equals an earlier OpenRGB zone number. The following constructs a full black controller frame with one dim-red logical position. Without `--apply`, it only validates and prints the plan:

```powershell
.\.venv\Scripts\python.exe .\openrgb_shell.py --device-name 'ASRock B650M Pro RS WiFi' `
  pixel --mode Direct --zone 'Addressable Header 3/Audio' --position 0 --rgb 64 0 0
```

After checking the inventory and accepting the Direct-mode risk, append `--apply` once. Stop if the result is a pattern, mirrored devices, no isolated point, or an unexpected motherboard change. If position zero isolates correctly, move one position at a time and record the physical result.

Direct mode is a capability test here, not established support. Entering it and sending a frame are hardware writes; the ASRock driver updates the complete controller even when the logical target is one header.

## Python API

```python
from openrgb_shell import connect

rgb = connect(device_name="ASRock B650M Pro RS WiFi")

for zone in rgb.zones:
    print(zone.index, zone.name, zone.start, zone.count, zone.addresses)

for address in rgb.address_domain:
    print(address)

rgb.select_mode("Direct")
frame = rgb.pixel_frame("Addressable Header 3/Audio", 0, (64, 0, 0))
rgb.send_frame(frame)

# Escape hatches, if needed:
raw_client = rgb.client
raw_device = rgb.device
```

`address_domain` is the explicit boundary between arbitrary controller identity
and numerical law identity. Every entry preserves the system address, region
name/index, position within that region, and full-frame slot while assigning a
dense `domain_index` from `0` through `N - 1`.

## Managed-dt LLVM linear walk

`openrgb_dt_walk.py` implements the first animation law:

1. Project the controller's arbitrary addresses and regions onto the dense
   `domain_index` described above.
2. Author the cursor, wrap, and RGB state as simultaneous SymPy equations.
3. Compile that law through the repository's generated AbstractTensor stage
   and LLVM piece route.
4. Instantiate one persistent Turing dt system. Each display frame is exactly
   one `advance_round` window; the law publishes both `max_vel` and `dt_limit`.
5. Convert the resulting RGB columns back through `frame_index` and send one
   complete controller frame.

The default is one full dry-run pass at five addresses per second. Because the
law speed is `fps` addresses/second and each managed window is `1/fps` seconds,
the white point advances by exactly one dense address per frame:

```powershell
py .\openrgb_dt_walk.py --device-name 'ASRock B650M Pro RS WiFi' --fps 5
```

The dry run prints the dense index and its preserved region/system identities.
It compiles and advances the real LLVM/dt path, but sends no hardware update.
After the one-position Direct-mode mapping test has succeeded, an explicit
finite hardware run is:

```powershell
py .\openrgb_dt_walk.py --device-name 'ASRock B650M Pro RS WiFi' `
  --mode Direct --fps 5 --frames 20 --white-level 255 --apply
```

For the measured eight-slice cooler/fan chain on Header 3, project only those
active positions into the dense compiler domain. Nine frames demonstrate one
complete pass and the wrap back to position zero:

```powershell
py .\openrgb_dt_walk.py --device-name 'ASRock B650M Pro RS WiFi' `
  --zone 'Addressable Header 3/Audio' --position-count 8 `
  --mode Direct --fps 2 --frames 9 --white-level 128 --apply
```

Use `--start-position 1` to phase this installed radial map at the left-side
midpoint, producing the observed `left -> top-left -> top -> top-right -> right
-> bottom-right -> bottom -> bottom-left` pass.

The symbolic law remains independent of this wiring calibration. `--zone` and
`--position-count` choose the measured system addresses before assigning dense
law indices; generated colors are still returned in a black 245-slot full
controller frame.

`--apply` selects the requested mode once and sends complete frames through the
existing `FullFrameController`. There is no retry, daemon, or background
service. Stop immediately if the controller mirrors, patterns, or affects an
unexpected physical site.

Do not use `rgb.device.zones[n].set_colors(...)` or `rgb.device.zones[n].leds[m].set_color(...)` for this ASRock controller. Those SDK operations reach driver methods that transmit only a zone's first color rather than its per-LED array.

## Offline verification

Tests use fake devices and make no socket or hardware call:

```powershell
py -m unittest -v test_openrgb_shell.py test_openrgb_dt_walk.py
```

## Continuous radial spirograph locator

`openrgb_dt_spirograph.py` treats the calibrated eight-light ring as a
continuous float field. For point `p` and unit light position `u_i`, it
evaluates the compact squared-distance lobe
`max(0, 1 - ||p-u_i||^2 / 2)^2`. At the perimeter this gives the aligned
light `1.0`, both direct 45-degree neighbors `0.5`, and
perpendicular/opposite lights `0.0`. At the exact center every light receives
the same dim `0.25` contribution. The configurable center correction fades
quadratically to zero at the perimeter rather than adding a broad uniform
background.

The spatial row is now the first matrix in a spectral pipeline. The white
source is a nonnegative eight-band spectrum selected to project to equal
linear RGB through the spectral ray tracer's response. Spectral power is
accumulated before display conversion; RGB is never used as the transported
field representation. Total deposited energy now varies with geometric
distance instead of being normalized to a constant, producing the requested
sharper localization.

Evaluating that analytic generator at arbitrary sites produces an `N x 8`
contribution matrix; the sites are not the model's spatial discretization. An
optional LUT bake adaptively subdivides one complete hypotrochoid repeat until
adjacent rows differ by no more than one selected color-code step. The default
`R=5, r=3` path repeats after `6*pi`, not `2*pi`. Site and wall-time budgets
cut off long bakes dynamically and report the achieved epsilon without
claiming saturation.

Repeat discovery evaluates the actual normalized point law at several phase
offsets over successive whole base turns. This prevents a path
self-intersection from being mistaken for full closure and lets degenerate
geometry expose its shorter real period. The discovered value is registered as
the managed `repeat_period` column consumed by the compiled SymPy law; phase
therefore passes through `2*pi` on the default path, wraps at `6*pi`, and wraps
symmetrically when angular speed is negative.

The verified 60 Hz live command is:

```powershell
py .\openrgb_dt_spirograph.py --device-name 'ASRock B650M Pro RS WiFi' `
  --zone 'Addressable Header 3/Audio' --position-count 8 `
  --fps 60 --frames 300 --angular-speed 1 --center-contribution 0.25 `
  --white-level 96 --exposure 3 --color-bits 8 --apply
```

The 2026-10-03 five-second live run achieved 59.996 Hz with zero deadline
misses. Total work was 3.184 ms at p95 and 10.387 ms maximum; SDK send time was
0.467 ms at p95. The Python process used 14.69% of one core, OpenRGB used
0.14% of one core, system CPU averaged 7.92% and peaked at 11.79%, and OpenRGB
working-set growth was 0 MiB. The 8-bit LUT saturated at 2,916 sites with a
maximum adjacent float step of 0.00391839 and repeat error of `1.665e-16`.

## Chaotic CMY three-body field

`openrgb_dt_three_body.py` integrates three freely gravitating masses on a
standard torus under the managed dt system.  Each body has two angular
coordinates `(u, v)`.  Gravity is the zero-mean Green operator of the torus
Laplace--Beltrami operator.  The repository `ParametricDomain` derives the
exact embedded-torus metric
`ds^2=(R+r cos(v))^2 du^2+r^2 dv^2`.  A finite real Fourier Galerkin space
omits the constant mode, `AbstractTensor` assembles and solves its weak
`-Delta_g` matrix once, and the resulting smooth periodic SymPy basis supplies
the potential and both coordinate derivatives.  This is intrinsic 2D gravity
projected into a continuous spectral space, not inverse distance on a chosen
parameter path, a periodic planar box, or a chord through the torus.
Christoffel terms keep free motion on the curved surface.

Particle coordinates remain unbounded.  Periodicity lives in the sine/cosine
basis itself, so neither a wrap counter nor a normalized wrap ratio enters the
state, energy, or force.  Excluding the constant mode enforces the compact
surface compatibility equation `-Delta_g G = delta - 1/Area`; the finite mode
order also regularizes the point source without a separately chosen softening
length.

The repository `RK4Integrator` manifests four causal symbolic stages, followed
by one endpoint law.  The endpoint emits relative mechanical-energy and
cyclic-`u` momentum errors as declared dt-system channels.  The ordinary
rollback/retry controller therefore divides a frame step when either measured
error exceeds its target; no animation-local snapshot or timestep loop is
used.  The bodies start from asymmetric `3, 4, 5` masses at rest.  There are no
prescribed tracks or phase-driven orbits.

Cyan, magenta, and yellow are nonnegative spectra on the ray tracer's eight
wavelength bands. Cyan and yellow are single spectral lines; magenta is an
explicit red-plus-violet composite. Each integrated planet position generates
its own continuous eight-light radial row. At each light the symbolic law
evaluates the matrix product
`(1x3 source weights) @ (3x8 source spectra) @ (8x3 display response)`.
The first multiplication is the field: pure and composite spectra add there
before any RGB projection. The final linear HDR RGB is passed through a smooth
exponential exposure curve rather than a per-channel hard clip, so overlaps do
not collapse immediately into an indistinguishable white plateau.

The final display boundary uses the compiler colorimetry module's IEC
61966-2-1 sRGB perceptual profile. HSL controls are decoded from sRGB to linear
light before the nonnegative eight-band spectral fit. Spectra and overlapping
sources therefore continue to accumulate linearly; after exposure, the shared
sRGB transfer function encodes the LED drive. Uniform and Sites already pack
their HSL colors in sRGB form and are not encoded a second time. This is a
standard perceptual transfer profile, not a motherboard- or LED-specific
chromatic calibration.

A face-on orthographic projection maps the embedded torus to an annulus whose
outer edge is at `0.82` of the calibrated light-ring radius.  The hole and the
far/near tube motion remain visible as radial motion inside the ring instead
of being erased by per-frame radius normalization.  The measured Header 3
red/green swap is corrected at the final full-frame boundary.

```powershell
py .\openrgb_dt_three_body.py --device-name 'ASRock B650M Pro RS WiFi' `
  --zone 'Addressable Header 3/Audio' --position-count 8 `
  --fps 60 --frames 600 --brightness 96 --center-contribution 0.125 `
  --exposure 3 --apply
```

For an observable run controlled from its own Pygame window, use `--frames 0`.
The window draws the face-on torus annulus, CMY body positions and trails, all
eight logical LED colors, coordinates, dt continuation, error maxima, and
accepted/rejected attempt counts.  Space pauses and Escape or closing the
window stops it. Terminal coordinate logging remains optional:

```powershell
py .\openrgb_dt_three_body.py --device-name 'ASRock B650M Pro RS WiFi' `
  --zone 'Addressable Header 3/Audio' --position-count 8 `
  --fps 60 --frames 0 --backend eager --window --apply
```

The Pygame ring labels every logical light as `index R/G/B`, using the exact
integer drive values calculated for that frame. Terminal coordinate logging is
off by default. Add `--print-every N` only when sampled coordinate output is
wanted.

`--cpu-temperature` starts one persistent LibreHardwareMonitor sensor worker
and copies the AMD processor's `Core (Tctl/Tdie)` value into the managed
`system_temperature_c` column before every frame. The Pygame status shows that
live system/processor value. Temperature is no longer a uniform velocity
scaler. Before each conservative managed frame, an exact
Ornstein--Uhlenbeck heat-bath step acts on each body's orthonormal torus-tangent
velocity components. Independent Gaussian impulses change direction as well
as energy, while damping and noise approach a temperature-dependent velocity
distribution. The simulation uses `temperature_C / 100` as its dimensionless
thermal energy and exposes the bath coupling rate directly. The status reports
the tangent-plane impulse RMS and signed kinetic-energy exchange. Gravity
remains the Laplace--Beltrami Green interaction, and its conservation channels
govern the subsequent managed frame step after that explicit environmental
impulse.

In the field studio, processor coupling remains opt-in. Without
`-CpuTemperature`, every independent field gets an ordinary persisted
`temperature C` slider and requires no elevation. With `-CpuTemperature`, the
real sensor replaces the slider value each frame while the entropy-coupling
control remains adjustable.

Sensor reads are validated as finite Celsius values; missing access or a
zero/missing sensor is an error rather than a fabricated fallback. The
installed hardware library requires an elevated parent process on this
machine. Launch the complete temperature-following effect from an ordinary
PowerShell with:

```powershell
Start-Process powershell.exe -Verb RunAs -ArgumentList '-NoExit -NoProfile -ExecutionPolicy Bypass -File "C:\dev\Powershell\speaktome\tools\openrgb_sdk_shell\run_temperature_three_body.ps1"'
```

`--trace-rejections` is an opt-in diagnostic which writes from inside every
attempted dt subdivision before
rollback.  Each rejection names the failing channel, attempted dt and limits,
all body coordinates and coordinate velocities, metric surface speeds, and all
three pair Green-potential values.  When subdivision finds an acceptable step it
prints `dt-accepted-after-reject` with the accepted dt and preceding rejection
count.  The Pygame status area retains the latest rejection summary.

`eager` evaluates the exact CSE'd SymPy equations while retaining the same
managed PieceState, rollback, and dt-controller graph.  `--backend llvm`
selects the final AbstractTensor-to-LLVM lowering after the visible behavior
has been checked.

The ten-second live run achieved 59.999 Hz with zero deadline misses. Total
work was 3.028 ms at p95 and 5.685 ms maximum; SDK send time was 0.395 ms at
p95, and the Python process used 16.72% of one core.

Those live and native timings predate the Laplace--Beltrami Green update.  A
120-frame eager dry run of the Green law crossed the old frame-90 stall with
all 120 full `1/60 s` attempts accepted: 59.998 Hz, zero deadline misses,
4.206 ms median work, 4.627 ms p95, 5.790 ms maximum, maximum relative energy
error `1.301e-13`, and maximum cyclic-momentum error `2.944e-13`.  LLVM
lowering for this new law has not yet been timed or asserted here.

Both effects can be selected through one entry point without changing their
managed-dt engines:

```powershell
py .\openrgb_dt_field.py spirograph <spirograph options>
py .\openrgb_dt_field.py three-body <three-body options>
py .\openrgb_dt_field.py studio <studio options>
```

## Interactive Pygame field studio

`openrgb_field_studio.py` places the eager intrinsic-physics, spirograph,
vectorscope, and manually placed emitter engines behind one live Pygame
control surface. It edits their
registered parameter columns directly; switching engines or moving controls
does not rebuild the symbolic laws.

Controls include:

- continuous distance-falloff power and center contribution;
- per-physics-body spectral hue, hue angular velocity, and mass;
- one to three active compiled mass slots;
- an arbitrary eager collection of spirograph sources, with add/remove and
  previous/next selection controls; each source owns its spectral hue, hue
  velocity, snapped integer `(R-r):r` ratio, angular offset, and independent
  angular-offset velocity;
- spectral accumulation of every spirograph source before the shared display
  transform, so overlapping sources blend under the same field algebra as the
  physics mode;
- spirograph angular speed and physics/spirograph engine switching. Adding a
  source reuses the existing eager managed law and does not compile a new law;
- the active empirically discovered spirograph repeat in the status panel;
- a manual mode whose numbered emitters are click-dragged to arbitrary field
  positions and remain where they are released; its integer deployment control
  can wipe the manual field and replace it with up to 128 emitters spaced
  uniformly around both the spatial ring and the spectral hue wheel;
- a refresh-rate-derived speed range. For the eight-site ring, the studio
  bounds the highest active path harmonic against the ring's fourth spatial
  mode temporal Nyquist limit;
- forward/reverse LED ordering;
- a live whole-frame wire-order toggle between direct RGB and the measured
  Header 3 GRB convention. This is shared by physics and spirograph modes and
  sits only at the final hardware boundary;
- a draggable LED ring which rotates continuously under the pointer and snaps
  to one of eight integer offsets on release; and
- manually adjustable or optional processor-driven entropic heat-bath coupling.

The receiver surface is separate from the source engine. The top tabs select
either the eight-address cooler ring or the four-address motherboard `PCB`
strip. By default each surface samples its own independently evolving field;
both are merged into every full-controller frame, so changing tabs affects
only the view and the hidden system continues updating. The ring starts with
controller identity 0 at the directly leftmost receiver and retains its
draggable clockwise/reverse calibration. The motherboard view uses a vertical
receiver line with identity 0 at the top followed by identities 1 through 3
downward.

### Persistent arrays and independent fields

`openrgb_field_studio_config.json` is the studio's persistent configuration.
The launcher passes it explicitly. Button changes, slider releases, tab
selection, ring calibration, source add/remove operations, and channel-order
changes are written atomically back to that file.

The JSON keeps three concerns separate:

- `arrays` describe OpenRGB zones, address counts, receiver geometry,
  identity calibration, per-array `RGB`/`GRB` channel order, and relative
  placement (`x`, `y`, rotation, and scale);
- `fields` describe independently advancing physics/spirograph systems,
  sources, hues, rates, masses, falloff, brightness, and exposure; and
- each array's `field_id` assigns that hardware geometry to a field.

The defaults bind `ring` to `ring-field` and `motherboard` to
`motherboard-field`. These are separate managed physics systems with separate
spirograph collections and controls; both advance even when their tab is not
visible. Assigning multiple arrays the same `field_id` is reserved as the
configuration-level mechanism for a future shared emitter scene. Their
relative `placement` values are already applied during receiver sampling, but
the scene-placement editor is not part of the current UI.

Each field has six engine modes: intrinsic physics, spirograph, a live stereo
vectorscope, manual emitter placement, `uniform`, and `sites`. Physics switches among a torus,
sphere, and Cartesian plane without replacing the managed-dt pipeline. The
torus uses the existing numerical Laplace--Beltrami Green operator; the sphere
uses the closed-form spherical Laplace--Beltrami Green kernel in longitude and
latitude coordinates; and Cartesian mode uses the softened planar inverse-
distance kernel. The vectorscope captures the default Windows speaker
render endpoint in the native C++ shell through shared-mode WASAPI loopback,
which does not require administrator rights. That shell loads the existing
`C:\dev\Powershell\fftfree\build\Release\fft_cffi.dll` C ABI and owns one
power-of-two real FFT plan for each stereo channel. The FFT window and temporal
overlap are powers of two; the requested light-field node count is deliberately
arbitrary. FFT bins are accumulated into that many logarithmically spaced
musical-frequency bands between the selected low and high limits.

Each band is one simultaneous light-field emitter. Its position uses the
band-limited stereo vectorscope convention `x=(L+R)/sqrt(2)`,
`y=(L-R)/sqrt(2)`, its hue follows its logarithmic center frequency, and its
field strength follows the measured FFT power. This keeps stereo placement and
spectral brightness coupled instead of reducing the input to either a single
trace or a volume meter. Hue offset, hue velocity, XY gain, field level,
log-band node count, FFT power, overlap, and frequency bounds are independent
for every field and autosave with the rest of the JSON.

Every colour source has HSL saturation and lightness beside hue and hue rate.
The light-field engines convert that HSL colour into the existing nonnegative
spectral basis before accumulation; the two direct engines use the same HSL
colour as their byte-level drive value without constructing a field.

`uniform` has one source object for a configured light array. It submits a
versioned `ASRockHIDUniformColor` ctypes/C message which becomes one 65-byte
`SET_ZONE` native-mode transaction, so one RGB value and one selected hardware
effect are applied by the controller to that selected array. The Uniform UI's
hardware-mode button cycles the known `0x00` through `0x0E` effects—including
Static, Strobe, and Audio—independently of the host-side source strobe control.
Its `allzone` byte remains zero:
setting it to one means every physical controller region, not every address in
the selected ARGB chain. `sites` creates one source object per configured
receiver address—eight for the cooler ring and four for the motherboard strip—
and writes those colours directly into the established packed `E3`/`E4` frame.
Neither mode performs spatial falloff or spectral field accumulation. Their UI
includes a compact offset/value/meaning table for the actual HID packing;
“site” always means an addressable light here, while the controller's physical
selector remains an internal protocol field.

Controller output frequency is a global live slider shared by every mode. It
changes the producer clock rather than becoming a Uniform-source property, so
it can be lowered while checking whether rapid mode writes interrupt an ARGB
chain update. Uniform compares the selected region, native mode, and packed RGB
at the Python boundary and submits no request while all three are unchanged.
Native per-region coalescing remains only as a defensive ABI safeguard.

Two rotations belong to the common field geometry rather than to the audio
engine. `sources angle` and its rate rotate the active physics bodies,
spirograph sources, or band-vectorscope emitters together. `lights angle` and
its rate rotate the selected array's receiver geometry through that same
field. The Pygame view and hardware sampling consume the same transformed
coordinates, so neither rotation is display-only. Sites restores the existing
receiver-light angle/rate/multiplier controls; on a ring, their visible result
is quantized to exact `360/N` address steps while a private phase accumulator
preserves sub-step motion between advances. Each step permutes the direct RGB
payload addresses using the configured ring direction, so both the HID image
and viewer advance together. Uniform ignores spatial rotation entirely and
does not advance hidden source or receiver rotation state.

All angular-rate controls retain their fine base-rate sliders and now have
independent `1x` through `100x` multipliers. This applies to whole-field
source rotation, receiver-light rotation, the spirograph path speed, each
spiro source's additional angular rate, and every physics, spirograph, manual,
or vectorscope hue velocity. The base spiro control still displays the derived
Nyquist range, while its multiplier deliberately permits aliasing and
phase-loss experiments beyond that range.

Strobe modulation is part of each source's colour controls. Physics bodies and
spirograph sources each own an independent waveform and frequency; manual
emitters now own the same controls, while the
vectorscope colour source owns one envelope which multiplies every band after
its individual FFT power has been measured. Clicking the strobe button cycles
`off`, `sine`, `square`, `triangle`, `saw`, `beat`, `audio`, and `timed`. `off` is exactly a
constant unit multiplier. Frequencies are limited by the current display-rate
Nyquist bound, and the source dots and emitted spectral field use the same
envelope.

`audio` is continuous band-level modulation rather than onset detection. Each
source selects one fftfree logarithmic band and an averaging-window duration;
its source envelope is the mean measured level in that band over the requested
window. The live status line reports the resulting per-source values. It uses
the same non-privileged WASAPI loopback capture as vectorscope and `beat`.

`timed` bypasses frame-rate strobe sampling. The native deadline dispatcher
orders full ON and OFF HID transactions by absolute `steady_clock` deadlines
expressed in nanoseconds. One native worker exclusively owns the HID handle,
so a transaction that runs past the next deadline makes that next dispatch
late instead of racing a second Windows overlapped operation on the same
handle. A coarse wait followed by a short yield loop approaches an available
deadline, and the UI reports measured native dispatch lateness in nanoseconds.
This is clock-resolution scheduling, not a claim that USB, the motherboard
controller, or the LEDs have nanosecond physical response.

The 30 Hz producer remains a latest-wins mailbox. Byte-identical pending,
in-flight, and already-resident frames are coalesced, and uniform requests are
coalesced independently per configured array. This removes redundant HID work
without changing the producer clock, adding another I/O thread, or delaying a
new state behind obsolete frames.

Timed mode has a predelay used to compensate observed transport/device latency,
an exact requested ON duration, a phase within the selected revolution, and a
fallback repetition rate. ON and OFF retain independent requested deadlines,
while their device writes are serialized. Continuous-throughput measurements
never quantize, lengthen, or otherwise rewrite either requested deadline: any
inability of the HID/controller to begin a transaction on time is reported as
dispatch lateness. ON events use a phase-locked integer divider. From the
guaranteed transport reset interval and the current revolution duration, the
studio suppresses complete revolution opportunities and schedules the next ON
at the selected phase of the first strictly eligible later revolution. OFF is
submitted at its requested absolute deadline without an eligibility, reset,
phase, or queue-state rule. Only the next ON is constrained: its divider
reserves `max(requested duration, guaranteed reset) + guaranteed reset`, so the
independently timed OFF has completed before that later ON. No queue-state poll
participates in the divider.

Run the launcher from an Administrator PowerShell to use the first fan sensor
reported by LibreHardwareMonitor:

```powershell
.\run_field_studio.ps1 -FanRpm
```

Select a specific LibreHardwareMonitor fan identifier when needed:

```powershell
.\run_field_studio.ps1 -FanRpm -FanSensor '/lpc/nct6798d/fan/0'
```

RPM supplies revolution duration but no physical tach-edge timestamp. The
resulting phase is therefore a chained estimate from the instantaneous RPM;
true rotor phase lock would require a timestamped tach edge from hardware.

`beat` is onset matching, not tempo prediction. Positive spectral flux across
the logarithmic FFT bands is compared with a 1.5-second adaptive floor and a
0.75-second decaying peak, then held visibly by an 80-millisecond release
envelope. The adjacent control becomes `beat +/-`: `+1` is dark between
onsets and bright on a strong onset, `-1` is bright between onsets and blinks
out on a strong onset, and `0` disables beat modulation. Intermediate values
blend continuously. Selecting `beat` in physics or spirograph lazily starts the
same native speaker capture used by vectorscope mode; it does not require
administrator rights.

Vectorscope tails are temporal again. Every logarithmic band retains its own
sequence of past mid/side positions, and Pygame draws a fading path through
those positions rather than joining unrelated adjacent frequency bands. The
same `tail decay ms` value drives a receiver-side peak-hold exponential fade,
so the physical LEDs retain the solved light field as the sound moves. This is
not additive normalization: new drive values replace weaker retained values,
then retained RGB decays naturally. History is discarded after 6.25 decay
constants, when an 8-bit peak is below half a code value; setting the slider to
zero restores instantaneous output.

Source-count normalization is independently selectable in every accumulated
field engine: Physics, Spirograph, Manual, and Vectorscope. With it on, source
strengths are divided by `sqrt(N)` so changing the active body, emitter, or
log-band count does not dominate overall field level. With it off, every
source is accumulated at full strength, preserving genuine dynamic range.
Uniform and Sites expose no normalization switch because they directly pack
one color per target instead of accumulating a multi-source field.

Every source now enters one shared Helmholtz colour law. The studio-facing
chooser cycles three authored systems: HSL, a Planck blackbody spectrum with a
1000--40000 K control, and a white-point chooser for equal-energy E, D50, and
D65. The same law also accepts HSV, encoded sRGB, linear RGB, XYZ, and an arbitrary
eight-band spectral distribution for compiler/native callers. Each route
produces linear RGB plus the common spectral representation; field addition
and exposure remain linear, and exact IEC sRGB encoding occurs only at device
output. Blackbody sources retain the Planck spectral distribution rather than
being reduced to a hue. Saturation mixes toward the equal-display-energy white
spectrum, while lightness is the source power control.

The symbolic laws are catalogued as the `Helmholtz` honorary-equation group in
`C:\dev\Powershell\engine_toy\honorary_engine_equation_catalogue.py`: Planck
radiance, general SPD-to-CIE-XYZ integration, xyY white-point construction,
XYZ-to-linear-sRGB, and the final piecewise IEC transfer. This keeps the
colour-system selector a parameter choice rather than a collection of
unrelated rendering paths.

`run_field_studio.ps1` builds `native_audio_scope\openrgb_audio_scope.cpp`
when its DLL is absent or stale, then starts the ordinary studio. The build uses
the installed CMake/MSVC toolchain and does not rebuild `fftfree` itself.

Every engine exposes the same three persisted projection radii: inner, outer,
and light-array ring. Their floating-point ratios control where sources land
relative to the receiver ring, including outside it. Torus projection uses
both inner and outer radii; sphere, Cartesian, spirograph, vectorscope, and
manual modes use the outer-to-light ratio because those geometries have no
inner hole. These are projection controls only; they do not alter any
intrinsic physics metric or Green kernel.

OpenRGB's hardware `Direct` mode remains controller-wide because both zones
belong to the same ASRock controller. Application engine mode and every source
state are field-local, while channel order is array-local.

The ring display reads the selected zone's actual full-controller frame slots,
not positions zero through seven of the whole device. Each spot is filled with
its latest calculated color and labelled with the corresponding integer RGB
tuple. The same snapped offset and direction mapping is applied to the hardware
address geometry before either engine evaluates its light field. The GUI then
places each address and its already-solved color at that same real-world site;
the hardware frame remains address-indexed and is not permuted a second time.
While the ring is being dragged, its continuous rotation is also included in
the field geometry, then both geometry and display snap together on release.

Run without processor temperature—and therefore without administrator rights:

```powershell
cd C:\dev\Powershell\speaktome\tools\openrgb_sdk_shell
.\run_field_studio.ps1
```

Raise the managed/display refresh rate—and therefore the derived spirograph
speed limit—at launch:

```powershell
.\run_field_studio.ps1 -Fps 240
```

On every hardware-enabled exit, including Escape, window close, or an error
after Direct mode is selected, the studio sends one full-controller black
frame before closing.

Run with processor-temperature kinetic coupling:

```powershell
Start-Process powershell.exe -Verb RunAs -ArgumentList '-NoExit -NoProfile -ExecutionPolicy Bypass -File "C:\dev\Powershell\speaktome\tools\openrgb_sdk_shell\run_field_studio.ps1" -CpuTemperature'
```

The geometry control deliberately remains torus-only. A correct sphere mode
needs a separate spherical state chart, Laplace--Beltrami Green operator, and
constraint-preserving integrator; changing only the projection would falsely
label torus dynamics as sphere dynamics.

AbstractUI was inspected but not used as the Pygame widget layer. Its current
portable contract carries HTML, CSS, JavaScript, JSON, Canvas, and WebGL-style
backend packets. It has no established Pygame packet, renderer, or event
adapter. Creating those as part of this hardware studio would be a new
cross-runtime backend rather than reuse of an existing robust path.
