# Turing mobile native shell and application-product continuation

- Migrate the web HTML runtime to consume `turing.application-product.v1`
  instead of duplicating embedded execution structures.
- Publish the primary presentation shader as a complete shader deployment and
  component record before referencing it from the application product.
- Install or discover the Android SDK and NDK; none was present in this visit.
- Add the common Vulkan C/C++ shell consuming CompiledProgramAPI and ShellIO:
  Android GameActivity plus the NDK Vulkan loader, and iOS CAMetalLayer plus the
  pinned MoltenVK static XCFramework.
- Compile the native program's existing LLVM/C ABI for Android arm64-v8a and
  iOS arm64 rather than creating a mobile-specific program ABI.
- Validate the iOS product on macOS with Xcode and a real/simulated iOS target.
- The real site-bundle path is currently blocked in planner specialization by
  a missing caller-owned source cell. Repair that compiler ownership issue;
  do not paper over it in the application publisher.
