# Concurrent native ray dispatch saturation trial

**Date:** 2026-11-13 UTC
**Title:** Spectral T1–T3 CPU/AMD OpenGL concurrent saturation, with Nodus boundary

## Overview

The owner asked for a demonstration of concurrent ray tracing jobs saturating GPU and CPU dispatch. This trial uses Spectral Analyzer's existing native `RayTracer` T1 intersection → T2 refinement → T3 material pipeline, with its existing CPU and OpenGL GPU workers competing for real ray batches. It is a dispatch throughput trial, not a finished photograph or a Nodus three-GPU market trial.

## Steps Taken

- Added `spectral-analyzer/profile_ray_dispatch_saturation.py`, which submits bounded batches through the original pipeline, drains its output queue, and records stage CPU/GPU counts and times. The same triangle and material inputs are used for CPU, GPU, and mixed modes. Its rays intersect the scene; a 100,000-ray, 8-band, 32×32-grid qualification produced 100,000 `STRIKE` records with valid triangle IDs and 100,000 T1/T2/T3 processed rays in aggregate.
- Added `spectral-analyzer/profile_ray_dispatch_concurrent.ps1` to launch and sample 1–8 independent native jobs. The matched demand series used `--rays 10000 --batches 2000 --window 30000 --grid 1 --bands 4 --mode mixed` per job. Each job therefore submitted 20 million rays. Record storage was bounded to 100 million floats per process. The runner writes separate output/error files and samples Windows GPU Engine and NVIDIA utilization counters.
- A previous attempt to enqueue ten 100,000-ray batches without a finite window failed with `std::bad_alloc`; the bounded 30,000-ray in-flight window corrected the queue pressure. The failed trial was not included in throughput figures.
- The original Spectral GL renderer string identifies **AMD Radeon(TM) Graphics**. This explains why Windows counts the OpenGL work on the AMD `engtype_3D` engine while `nvidia-smi` stayed at zero. Counter samples outside 0–100% and samples with invalid status were excluded.

## Observed Behavior

| Concurrent mixed jobs | Submitted and drained rays | Longest job (s) | Aggregate rays/s | Mean AMD 3D engine during simultaneous valid samples |
|---:|---:|---:|---:|---:|
| 4 | 80 million | 31.58 | 2.53 million | ~76% |
| 6 | 120 million | 43.00 | 2.79 million | ~81% |
| 8 | 160 million | 57.76 | 2.77 million | ~78% |

All 18 jobs in this matched series exited 0 with `in_flight=0`. The eight-job run produced exactly 20 million valid `STRIKE` records per job. Across the eight jobs, each of T1, T2, and T3 counted **97,332,016 CPU** and **62,667,984 GPU** rays, totaling 160 million per stage. Combined process CPU time was 892.6 CPU seconds during a 57.76-second longest job, about 15.45 logical CPU cores on a 24-logical-core machine. Thus CPU and GPU native workers did run concurrently. The six-to-eight throughput change is within ~1%, indicating a dispatch throughput plateau for this specific scene, queue window, and machine. It does not prove 100% physical GPU or CPU occupancy.

Separate lane curves used the same scene and 10 million rays per job (`--rays 10000 --batches 1000 --window 30000`, one pipeline mode per run):

| Lane | Concurrent jobs | Longest job (s) | Aggregate rays/s | Other measured load |
|---|---:|---:|---:|---:|
| CPU only | 2 | 10.53 | 1.90 million | 12.50 process CPU seconds/wall second |
| CPU only | 4 | 15.39 | 2.60 million | 14.96 process CPU seconds/wall second |
| CPU only | 6 | 24.36 | 2.46 million | 15.03 process CPU seconds/wall second |
| GPU only | 2 | 14.15 | 1.41 million | ~74.6% AMD 3D engine |
| GPU only | 4 | 23.85 | 1.68 million | ~83.9% AMD 3D engine |
| GPU only | 6 | 35.05 | 1.71 million | ~84.5% AMD 3D engine |
| GPU only | 8 | 44.85 | 1.78 million | ~89.4% AMD 3D engine |

Every lane job exited 0, drained to `in_flight=0`, and emitted one valid `STRIKE` per submitted ray. CPU throughput declined from four to six jobs, while GPU throughput gained only 6% from four to eight jobs despite twice as many GPU contexts. The eight-GPU-job point still gained about 4% over six; it is near sustained engine saturation, not a mathematically proven maximum.

The CPU-only qualification emitted one terminal record per ray; the GPU-only qualification emitted a strike record for every ray but no terminal output records, although T3's GPU processed counter reached the full input count. This is an output-contract difference to investigate before using this benchmark as evidence of final exposure parity. The benchmark uses zero-reflectance material and `max_bounces=1`; it does not exercise T4/T5, BDPT, a film image, or compound lens traversal.

The Nodus `KernelPipelineTool` and original Spectral shader adapters already execute sensor raygen→T1→exact-conic T2 in a shared tensor state library. T3 has a validated Vulkan artifact but Nodus's generic shader row currently leases only storage buffers and cannot bind its image resource. The earlier four-worker Nodus canvas trial used fabric plus T1, not full T1–T3 ray jobs. The present saturation numbers therefore belong to Spectral's original native CPU/AMD OpenGL pipeline. They must not be reported as Nodus three-GPU ray-market results. See `nodus/docs/KERNEL_PIPELINE_ADOPTION_2026-11-13.md` and `nodus/docs/TENSOR_STATE_LIBRARY_GPU_DISPATCH_2026-11-12.md`.

## Next Steps

1. Investigate GPU terminal readback/output semantics and assert CPU/GPU result parity for a bounded fixture, including sensor image output if terminal records are intentionally omitted on GPU.
2. Extend Nodus's existing Vulkan resource lease/descriptor path to support T3's original image binding, then run sensor raygen→T1→T2→T3 with a real output tensor on all three Vulkan devices and CPU where the same source supports it.
3. Repeat the matched 4/6/8-job saturation series through the Nodus canvas/market with all available GPU dispatchers and CPU workers. Record per-device resident state, transfer, active time, and end-to-end output parity before claiming that integration.

## Prompt History

> “Can you continue to finish the work”

> “Success condition right now is demonstrating saturating the gpu and cpu dispatches with concurrent ray tracing jobs”
