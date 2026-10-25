# Turing target-aware backend emission continuation

- Link the already verified `ios-arm64` Mach-O object on an Apple/Xcode host;
  do not claim an iOS dylib or signed application from Windows.
- Make C/Fortran build drivers consume the same target record where their
  toolchains support cross compilation and refuse unsupported target/SDK
  combinations by name.
- Publish the verified build-target receipt in native application-product
  target records.
- Route Nodus import/execution through the same serialized receipt rather than
  maintaining a second platform-name table.
