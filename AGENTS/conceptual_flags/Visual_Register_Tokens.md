# Visual register tokens: explicit non-spatial workspace

## Status and scope

This is a research-backed **conceptual flag**, not a compiler contract or an
implementation claim. It asks future visual graph designs to consider explicit
non-spatial workspace nodes when spatial nodes might otherwise be repurposed for
global computation. The evidence is strongest for particular Vision
Transformers (ViTs); it does **not** prove that every visual model needs
registers, that every high-norm token is harmful, or that registers solve every
dense-feature problem.

Here, “low-information” or “redundant” means that an image patch contributes
little distinct local evidence relative to nearby patches. It does not mean the
token is computationally inactive. In fact, the concern is precisely that a
model may make such a spatial token computationally important while weakening
its spatial meaning.

## Finding in the literature

Darcet et al. identified sparse, high-norm patch-token artifacts in trained
supervised, text-supervised, and self-supervised ViTs [1]. Their detailed study
focused on DINOv2. In a large DINOv2 model, the outliers emerged around middle
layers, were concentrated in patches whose initial embeddings resembled their
neighbors, retained less recoverable position and pixel information, and
carried more global classification information than ordinary patch tokens.
This supports a specific interpretation: sufficiently capable models can reuse
redundant spatial tokens as internal workspace, mixing a global computational
role into tokens that downstream code expects to remain local.

Their intervention was to add learned, input-independent tokens to the ViT
sequence. These register tokens have no image-grid coordinate. They participate
in every self-attention/MLP block during both training and inference, so their
hidden states become input-dependent. Only at the output readout are register
positions excluded from the normal class-token and spatial-patch outputs. They
are not persistent memory across images, hardware registers, or tokens removed
after pretraining.

The official DINOv2 implementation makes the tensor flow concrete [2]:

```text
patches = patch_embed(image)                         # B x N x D
spatial = concat(CLS, patches) + positional_encoding
sequence = concat(spatial[:, :1], expand(REG, B), spatial[:, 1:])
sequence = transformer_blocks(sequence)              # B x (N + 1 + R) x D
normalized = final_layer_norm(sequence)
patch_output = normalized[:, 1 + R:]                 # registers skipped here
```

`REG` is a learned parameter of shape `1 x R x D`, expanded across a batch and
inserted after positional encoding has been applied to the class and patch
tokens. It is therefore shared at the input but not a fixed per-image state
inside the network. Each block processes the full sequence. The same repository
returns normalized register states separately while patch-oriented intermediate
readouts slice them away [2]. The official DINOv2 model card documents four
optional learned register tokens [3], and DINOv3's official model card likewise
describes transformer checkpoints with one class token, four registers, and the
patch grid [4]. DINOv3 also introduces Gram anchoring for dense-feature quality;
that separate mechanism is a reminder not to treat registers as a complete
answer to representation quality [5].

The original norm result also needs measurement discipline. Darcet et al.
analyzed feature norms at model outputs and across intermediate layers [1]. In a
pre-norm ViT, measuring only the final LayerNorm output can hide magnitude
structure. A reproduction should record residual-stream token norms immediately
before the final LayerNorm as well as normalized features, attention statistics,
and downstream behavior. Large norm, high attention, semantic information, and
causal importance are related questions, not interchangeable labels.

## Important qualifications

Later work narrows the safe conclusion.

- Jiang et al. found sparse “register neurons” involved in concentrating
  high-norm activations and created test-time registers without retraining [6].
  Their intervention is not merely appending an arbitrary zero token: it
  appends a token and deliberately activates the identified neurons at that
  position. It supports the workspace interpretation while showing that
  trained register parameters are not the only possible intervention.
- Lappe and Giese report that, in large DINOv2 variants, class-token outputs can
  become dominated by information flowing through registers; cleaner attention
  maps then need not faithfully explain how local patches form the global
  representation [7]. Artifact suppression is not equivalent to
  interpretability.
- Shi et al. find that registers can suppress high-norm artifacts while leaving
  localization and feature-alignment problems, and propose a different account
  based on background-driven “lazy aggregation” [8]. This CVPR 2026 result is a
  direct warning that visual systems may need more than dedicated workspace.
- A March 2026 cross-architecture reassessment reproduces parts of the original
  result but finds that some claims do not generalize across models [9]. It is a
  preprint submitted to TMLR, not an accepted TMLR paper. Register/no-register
  comparisons are also constrained by which architectures provide comparable
  checkpoints.

The design signal is consequently conditional: explicit workspace is worth
testing when a visual architecture couples spatial identity to a globally
mixing computation and exhibits evidence of spatial-token repurposing. It is
not a universal architectural law.

## Proposal for future visual graph systems

For a future visual graph path, represent candidate workspace as explicit
non-spatial nodes rather than silently allowing image-region nodes to carry two
contracts. A minimal design would give each sample a finite `R`-node workspace,
initialize it from learned values (or a separately justified rule), allow
bidirectional interaction with spatial nodes through the same graph layers, and
exclude it from spatial reshaping. The workspace state must reset or be derived
per sample unless a separate, explicit persistent-memory design is intended.

This is a capacity choice, not magical semantics. Non-spatial nodes may still
influence every output indirectly, so a compiler or optimizer must not prune
them merely because they have no grid coordinate or direct spatial readout. A
full-attention layer grows from an `(N + 1) x (N + 1)` score matrix to
`(N + 1 + R) x (N + 1 + R)`; that is bookkeeping, not a measured runtime claim.
Token order, masks, gradients, checkpoint compatibility, and downstream slicing
must all be explicit before implementation.

The existing neural-network graph files are anchors for later investigation,
not evidence that this feature exists. `tensors/abstract_nn/graph_capsule.py`
builds generic computation graphs and adds explicit **control** nodes;
`tensors/abstract_nn/graph_runner.py` schedules and trains those graphs; and
`tensors/abstract_nn/demo_cifar10.py` demonstrates a flattened-image MLP. None
currently implements a patch-token ViT or learned non-spatial workspace. Any
future implementation should extend the applicable model/graph representation
rather than reclassify the capsule's control nodes as registers.

## Proposed tests (not performed)

1. Compare `R = 0, 1, 4, 8` under controlled parameter and compute budgets.
2. Record per-layer residual-stream norms before the final LayerNorm and the
   corresponding normalized features; map patch outliers back to images and
   quantify concentration in locally redundant/background regions.
3. Evaluate both global classification and genuinely local tasks such as
   segmentation, depth, correspondence, or localization. A cleaner attention
   map alone is not a success criterion.
4. Probe where local, global, and positional information is stored, and use
   causal ablations or interventions in addition to attention visualization.
5. Verify that readout code removes only non-spatial positions from spatial
   reshaping while retaining their influence through preceding layers.
6. Test checkpoint loading, batching/expansion, positional encoding, masks,
   gradient flow, and inference behavior explicitly.

No benchmark, ablation, or runtime result in this section has been run for this
repository. The tests are decision gates for a later implementation proposal,
not evidence that registers will improve the current graph stack.

## References

1. T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski, “Vision Transformers
   Need Registers,” ICLR 2024; arXiv v2, 12 April 2024.
   <https://openreview.net/forum?id=2dnO3LLiJ1>
2. Meta AI, DINOv2 official `vision_transformer.py`, accessed 2 November 2026.
   <https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py>
3. Meta AI, “DINOv2 Model Card,” accessed 2 November 2026.
   <https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md>
4. Meta AI, “DINOv3 ViT-B/16 model card,” accessed 2 November 2026.
   <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
5. O. Siméoni et al., “DINOv3,” arXiv:2508.10104, 13 August 2025.
   <https://arxiv.org/abs/2508.10104>
6. N. Jiang, A. Dravid, A. A. Efros, and Y. Gandelsman, “Vision Transformers
   Don't Need Trained Registers,” arXiv:2506.08010, revised 2025.
   <https://arxiv.org/abs/2506.08010>
7. A. Lappe and M. A. Giese, “Register and [CLS] tokens induce a decoupling of
   local and global features in large ViTs,” NeurIPS 2025; arXiv:2505.05892.
   <https://arxiv.org/abs/2505.05892>
8. C. Shi, Y. Yu, and S. Yang, “Vision Transformers Need More Than Registers,”
   CVPR 2026, pp. 26328–26337. <https://openaccess.thecvf.com/content/CVPR2026/html/Shi_Vision_Transformers_Need_More_Than_Registers_CVPR_2026_paper.html>
9. S. Baxevanakis, P. Karageorgis, I. Dravilas, and K. Szewczyk, “Do All Vision
   Transformers Need Registers? A Cross-Architectural Reassessment,”
   arXiv:2603.25803, 26 March 2026 (preprint; submitted to TMLR).
   <https://arxiv.org/abs/2603.25803>
