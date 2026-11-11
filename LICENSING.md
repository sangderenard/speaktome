# Licensing

SpeakToMe retains its existing [MIT license](LICENSE), including the
`Copyright (c) 2025 SpeakToMe Authors` notice. Its root and first-party package
metadata already declare MIT. The owner's 2026-11-11 direction confirms MIT
for the first-party work returning to SpeakToMe; it does not relicense
third-party software or claim rights to independently authored material.

## Repository boundaries

Turing, Nodus, nogodsnomasters and spectral-analyzer maintain their own
repository license documents. Preserve source paths, revisions and original
notices when moving first-party code between them. Shared imports or a shared
parent directory do not replace third-party terms.

The HTTP route extraction in commit
`91add50378fae73baeefe4505a57007fc9a9b0a3` came from Turing and follows the
owner's first-party MIT policy. Its matching Turing adapter is `763582a0`.
That source relationship does not extend the MIT grant to backend dependencies.

## Pinned submodules

The following terms were read at the checked-out gitlink revisions. This is a
component inventory, not a claim that every optional module or combined build
has the same terms.

| Submodule | Revision | Observed notices |
| --- | --- | --- |
| codex | b96b63ecb863a7fa78252b39409852e068ca3e29 | Apache-2.0; retain LICENSE and NOTICE |
| wheelhouse | 0c4c7a3768e7c282fffd43728f33050d7e9efeac | No tracked license/notice found; separate review needed |
| eigen | db8bd5b825918e20212d29b45d3fbd2b015b616b | Primarily MPL-2.0; retain all applicable COPYING files |
| libigl | 73a2e0de9b0bcfac83dc5bb661049c516a73650f | MPL2 modules, GPL copyleft modules, and dependency-specific restrictions |
| imgui | 0dc2885f3e0890585a0fa61a1ea041df9609b0ab | MIT |
| glfw | e7ea71be039836da3a98cea55ae5569cb5eb885c | zlib-style license |
| glad | 431786d8126e4f383a81e36f47b61a5d52a1c20d | MIT generator, Apache-2.0 Khronos specifications, additional header terms |
| stb | f58f558c120e9b32c217290b80bad1a0729fbb2c | MIT or public-domain alternative |
| tinydnn | c0f576f5cb7b35893f62127cb7aec18f77a3bcc5 | BSD-3-Clause and nested third-party notices |
| onnxruntime | c98a0e014b641e289ed25f42b792bca1893ccb03 | MIT and component-specific notices |

The graphics submodules reside below
`asciioscilloscope/asciioscilloscope/third_party/`. Keep their license and
copyright files with the components. Do not replace them with this repository's
MIT notice.

## Bundled references and unresolved provenance

- `DataProcessor/phpseclib/` contains third-party author and MIT references.
  No separate full license file was found in the scoped inventory. The exact
  imported release and its complete notice set still need reconciliation;
  its authors' terms and attribution must remain.
- Garmin FIT SDK reference files under
  `DataProcessor/nonuploading resources/SDK/` were removed from this branch's
  tracked tree by `2c7adc2078adc55361cc6a74424a011e47f05411` and are ignored.
  All 1,973 local files were preserved. They retain Garmin's FIT Protocol
  License and are not covered by SpeakToMe's MIT grant. Other branches and
  old commits still contain them; no history rewrite was performed.
- Downloaded models, datasets, media, binary dependencies and transitive
  dependencies retain their own terms. This document is not an exhaustive
  release or rights audit.
