# Turing mobile reference product: Windows-to-phone handoff

Date: 2026-10-24

## Result

The canonical `C:/dev/Powershell/turing` repository now has one mobile
reference application authored in Turing source and consumed by Android,
iOS, and browser shells. The authored calculation is not copied into a host.
All hosts consume the compiled `mobile_main__mobile_main` public entry and its
published buffer order.

The application shape is now concrete:

1. shell starts and owns device/window/input/frame timing;
2. shell calls the compiled public program entry;
3. the program publishes RGB values through its generated ABI;
4. the shell supplies those values to the `main-output` shader component; and
5. the shader alone owns presentation.

## Android product (verified on Windows)

The build stages Android SDK platform 37.0, build-tools 37.0.0,
platform-tools 37.0.1, and NDK 30.0.16248370 under ignored build storage.
`examples/mobile_reference_app/android/main.cpp` is a real NativeActivity
shell. It creates a Vulkan instance, Android surface, graphics/present device,
FIFO swapchain, render pass, SPIR-V shader modules, pipeline, framebuffers,
command buffer, and frame synchronization. Touch coordinates enter the Turing
program and its RGB output becomes a fragment push constant.

`build_mobile.py --android` completed. It compiled the sanctioned source entry,
emitted LLVM, compiled the program with NDK clang for
`aarch64-linux-android24`, linked `libturing_mobile.so`, assembled an APK,
zip-aligned it, generated an ignored debug keystore, and signed the APK.

Observed verification:

- APK: `build/mobile-reference/android/turing-mobile-debug.apk`
- SHA-256: `76fa8580b5b4127c463328546592e0a8f2d6f05a98216af58ed22670d5c72a03`
- APK Signature Scheme v2: true
- APK Signature Scheme v3: true
- package: `org.turingcompiler.mobile`
- launchable activity: `android.app.NativeActivity`
- min SDK: 24; target/compile SDK: 37
- native code: `arm64-v8a`
- shared library: ELF64, AArch64, DYN
- required feature: Vulkan 1.0 and feature level 1
- both SPIR-V modules passed `spirv-val`

`adb devices` started the local daemon and reported no attached device. The APK
therefore has not been physically launched here. `install_android.ps1`
performs install and launch once a phone is attached and authorizes USB debug.

## iOS compiler and shader product (verified on Windows)

The explicit `ios-arm64` target produced
`build/mobile-reference/ios/mobile_main.o`. Its first bytes are
`cf fa ed fe 0c 00 00 01`, confirming a 64-bit ARM Mach-O object. The product
also contains its generated ABI header and target receipt.

The same GLSL shader sources were compiled to validated SPIR-V and then
translated by SPIRV-Cross to Metal Shading Language 2.0. The generated vertex
and fragment sources are separate libraries and expose `main0`, matching the
Objective-C++ `MTKView` host in `ios/main.mm`. That host compiles the MSL from
bundle resources, calls the compiled Turing program each frame, and supplies
its RGB output at fragment buffer index zero.

MoltenVK 1.4.2 is independently hash-pinned and staged for the Vulkan-over-
Metal route. The reference iOS host selects direct Metal after SPIR-V-to-MSL;
the application-product report retains MoltenVK as the verified alternative
runtime dependency rather than pretending it is used by this host.

The final executable link was not attempted because this Windows machine has
no iPhoneOS SDK. `ios/package_ios.py` makes that boundary explicit: it requires
an SDK root, links UIKit/Metal/MetalKit/QuartzCore plus the already-produced
Mach-O object, assembles `Payload/TuringMobile.app`, and writes an unsigned
IPA. Sideloadly or AltStore then owns Apple-account signing and installation.
They are not treated as compilers or SDK linkers.

## Browser/iPhone product (executable now)

The PWA under `build/mobile-reference/web` calls the same public compiler
entry. A Node execution proof with inputs `(0.5, 0.25, 0.75)` returned
`(0.42500000000000004, 0.36250000000000004, 0.7375)`, the authored program's
exact result. The page feeds those values into a WebGL2 main-output shader and
is installable as a standalone Home Screen app. `serve_phone.ps1` binds the
product to the LAN and prints an address for Safari.

The PWA is emitted by the independent whole-repository Wasm backend; it does
not pass through LLVM. During implementation that backend exposed an exact
identity loss: inlining computed the callee's three results but discarded the
public wrapper's `ssa.aggregate` result record. Its `GetElementPtr` instructions
already carried the authoritative `source_output_id`, but the emitter had no
aggregate-member table to consult. The emitter now retains the Call's exact
`output_ids` to callee-local mapping and resolves each wrapper load by its
`source_output_id`. No private planned-region entry or positional guess is
used. The focused public-entry execution test returns the expected RGB values.

WebAssembly remains its own independent repository-SSA backend. Its artifact
must be as available and versatile as a C shell artifact: any page shell may
place it, feed it, schedule it, and select presentation. The Wasm backend must
not own a page, DOM structure, animation loop, input policy, shader, or
application-specific behavior. The page is one replaceable shell consumer.

## Architecture and build-target work included

`NativeTargetContext` is now the registry-owned `turing.build-target.v1`
contract. Explicit entries cover `ios-arm64`, `android-arm64-v8a`, and
`x86-64-portable`, along with the existing Windows/internal contexts. The
contract travels through source lowering and LLVM emission, supplies the LLVM
triple and toolchain target, and is checked against tampering. Portable x86 is
defined as machine-state SSA execution/re-emission, not a false claim that one
PE/ELF/Mach-O binary runs everywhere.

`turing.application-product.v1` references the already-owned program API,
execution artifact, shader deployment/artifact, and runtime dependencies. It
does not introduce another computation ABI. Site bundles publish the same
reference-only record rather than keeping web application identity in a
bespoke page path.

## Tests and evidence

Focused target tests previously passed 19/19. The wider touched target suite
passed 45 tests with one documented pre-existing cosine-literal expectation.
Android build, APK signature verification, `aapt dump badging`, ELF inspection,
SPIR-V validation, iOS object emission, SPIR-V-to-MSL translation, and the Node
Wasm execution proof all completed. The repository's known full site-bundle
test remains blocked earlier in planner specialization; it was not masked.

The final combined touched suite passed 53 tests with 3 platform skips across
the LLVM backend, mobile reference, application product, target registry/native
retention, and repository-SSA module tests.
The generic browser binding also passed JavaScript syntax validation and the
guestbook validator reported that all filenames conform. The continuation stub
names the remaining physical-device and backend breadth work.

## Delivery and Windows signing preparation

The implementation was committed as Turing `31e0fdf7` and pushed to
`origin/main` on 2026-10-24. Ignored build products, downloaded toolchains,
the generated Android debug key, and unrelated `.claude/` and `out/` scratch
directories were not committed.

The official Apple desktop iTunes and iCloud installers and the current
Sideloadly 64-bit installer were staged under ignored build storage. The Apple
installers have valid Apple Inc. Authenticode signatures. Windows refused both
silent Apple installs because the current process is not elevated; completing
them requires the operator to approve the Administrator/UAC prompt. The
Sideloadly installer and its extracted executables are not Authenticode-signed,
so that executable boundary is retained for explicit operator approval rather
than silently treating an HTTPS download as publisher verification.

Apple distributes the iPhoneOS SDK inside Xcode, not as a supported standalone
Windows SDK. The machine currently has about 34.5 GiB free on C: and 22.6 GiB
on E:, so a complete current Xcode download plus extraction cannot be assumed
to fit. The operator explicitly cancelled the iPhone/Sideloadly installation
path before any installer was run interactively or any Apple authentication
was entered. The Apple sign-in tab was closed. No Apple credential, password,
2FA code, or provisioning material was requested from or stored by the
compiler tooling. The staged installers remain ignored build inputs only. The
operator will use a Mac with Xcode for final SDK linking, signing,
provisioning, and device installation; no Windows sideload tooling is part of
the continuation.
