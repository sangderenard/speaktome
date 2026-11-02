# Documentation Report

**Date:** 1793629497
**Title:** Visual register tokens conceptual flag

## Overview

Added a sourced conceptual flag about explicit non-spatial workspace/register
nodes for visual neural networks. The note separates published evidence from the
user's motivation and from proposed, unimplemented implications for SpeakToMe's
neural-network graph work.

## Steps Taken

1. Read the root and `AGENTS/` guidance, guestbook instructions, documentation
   template, and long-document guidance.
2. Searched the current tree and open-work context for an existing register-token
   proposal.
3. Read primary literature and inspected the official DINOv2 implementation and
   DINOv2/DINOv3 model documentation.
4. Inspected `graph_capsule.py`, `graph_runner.py`, and `demo_cifar10.py` before
   describing them as future integration anchors.
5. Added the conceptual note and its index entry.
6. Ran the guestbook validator and documentation checks recorded below.

## Observed Behaviour

The repository's graph capsule exposes control nodes, but the inspected neural
network files do not implement patch-token transformers or learned register
nodes. The literature supports explicit register tokens as one tested remedy for
spatial-token artifacts in several ViTs, with important later qualifications
about generality, localization, and interpretability.

## Lessons Learned

“Discarded registers” should mean excluded from the final patch/class readout,
not absent during inference. DINOv2 register states remain in the sequence
through every transformer block. Clean attention maps are also insufficient as
evidence that local features faithfully determine global outputs.

## Verification

- `python AGENTS/validate_guestbook.py` — passed; all filenames conform. The
  validator reported older entries it would archive only with `--apply`; no
  unrelated archive cleanup was applied.
- `git diff --check` — passed.
- Confirmed the note's three SpeakToMe anchor paths exist and inspected their
  contents; no neural-network tests were run because this change is docs-only.
- Inspected all eight currently open pull requests through the public GitHub
  page and searched the repository tree for related terminology; no duplicate
  register-token proposal was found.
- Checked the primary source pages and the official DINOv2 implementation/model
  documentation used for the tensor-flow description.

## Next Steps

No implementation is authorized by this documentation-only task. If a later
proposal proceeds, first run the controlled ablations and diagnostics listed in
the conceptual flag; do not infer a new compiler contract from this note.

## Prompt History

> "\"register tokens\" research the literature and give us in speaktome I guess or a neural network flagged note in turing, the need for visual systems to contain register nodes to avoid placing them in inert regions"

> "Publish scoped proposals as draft PRs with actual verification results and explicit limits; do not merge or deploy."
