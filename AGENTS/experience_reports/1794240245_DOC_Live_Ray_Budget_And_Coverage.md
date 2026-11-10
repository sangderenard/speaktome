# Live ray batch and sampled-pixel transparency

Implemented source changes in the canonical Turing and Spectral repositories.
The API budget is a strict integer 128..65536, default 8192 total primary rays
per complete sensor sweep. Transport revisions apply at sweep boundaries and
clear ring/checkpoint history. Display-only revisions do not clear history.
The forward field remains additive; the budget is not an exposure-invariant
quality control.

Live PNG alpha is taken from actual sensor sample-validity alpha, preserving
sampled black as opaque and unvisited/reset pixels as transparent. Still PNGs
remain RGB. Partial live snapshots now carry their current applied revision
atomically with the complete status.

CPU verification: 40 Turing tests passed; 137 Spectral tests passed and one
opt-in GPU test skipped. System Python is the working Spectral test environment.
No live worker, GPU job, server, export, publication or user exposure was touched.
The receipt is in Turing docs/hazards/20261109T160349Z_live-ray-budget-coverage.md.

## Prompt History

User: "also give us like, live ray bathch size"

User: "honestly i think i and users would appreciate if unexposed pixels remained clear, so the ray trace grows like moss over the user experience"
