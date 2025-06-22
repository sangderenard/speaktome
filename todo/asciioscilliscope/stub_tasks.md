# asciioscilliscope Stub Tasks

The C++ oscilliscope experiment contains several stub blocks. The list below breaks down the expectations for implementation.

## Renderer
- connect to image or signal sources
- implement phosphor downsampling in `exciteFromImage`
- run a continuous render loop in `start`
- emit output through `flushDisplay`

## PixelFrameBuffer
- allocate buffers with correct memory layout
- add locking for thread safety
- compute diffs efficiently and swap buffers

## CharClassifier
- configurable brightness ramp (implemented)
- variable channel classification method (implemented)
- document arbitrary channel counts and aggregated sample sites
- add batched classification support
- implement per-channel timing offsets

## BeamFocus
- validate parameters in constructor
- generate convolution kernels
- apply kernels in `processMask`

## Interpolator
- precompute coefficients for selected mode
- implement Eigen-based resampling

## SampleSiteGrid
- aggregate HD tensors into sample sites
- compute exact centers for each site in `initMetadata`

## PhosphorScreen
- allocate event queues and buffers
- process queued excitations with decay and envelope
- apply diffusion to rendered buffers

## Diffusion Kernels
- finish custom, dynamic, and complex kernel logic

These tasks provide a starting point for incremental implementation and testing.

### Stub Catalog

Individual stub markers in the `src/` directory mapped to their suggested work
items:

- **BeamFocus.cpp**
  - Constructor logic
  - `setGridParameters`
  - `processMask`
- **CharClassifier.cpp**
  - Static `classify` helper
  - Member `classify` method
- **ComplexKernel.cpp**
  - Prepare spline interpolation data
  - Implement location dependent diffusion
- **ConicProjector3D.cpp**
  - `projectCone`
- **CustomKernel.cpp**
  - Precompute custom features
  - Implement custom diffusion
- **DynamicKernel.cpp**
  - Initialize dynamic behavior
  - Timestamp-aware diffusion
- **Interpolator.cpp**
  - Constructor implementation
  - `resample`
- **PhosphorScreen.cpp**
  - Constructor setup
  - `excite`
  - `renderBuffer`
  - `applyDiffusion`
- **PixelFrameBuffer.cpp**
  - Constructor
  - `updateRender`
  - `getDiffAndSwap`
- **Renderer.cpp**
  - Constructor tasks
  - `exciteFromImage`
  - `Renderer::start`
  - `processDiffs`
  - `flushDisplay`
- **SampleSiteGrid.cpp**
  - `reduceHdTensor`
  - `initMetadata`
