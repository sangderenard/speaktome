#pragma once
#include <vector>
#include <deque>
#include <tuple>
#include <mutex>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "DiffusionKernel.h"

namespace asciioscilliscope {

/**
 * High-Definition Electron Gun and Phosphor Simulation
 * ---------------------------------------------------
 * This module simulates a CRT electron gun delivering an arbitrary-resolution
 * high-definition tensor of energy deposition events across the display plane.
 * Unlike traditional pixel-based systems, all stages operate on the original
 * high-definition tensor until final sampling:
 *
 * 1. Electron Gun Stage
 *    - Input: HD tensor [channels, HD_rows, HD_cols] representing beam intensity.
 *    - Beam can be focused or unfocused via mathematical confinement parameters:
 *        • ConfinementRadius, FieldCurvature, BeamAngle, Gain
 *    - Emits a diff mask tensor of the same HD shape, representing incremental
 *      energy delivery per time slice.
 *
 * 2. Sample Site Grid Reduction
 *    - A lower-resolution grid of sampling sites is defined, one per color channel.
 *    - Each site has metadata: (siteRow, siteCol, sampleRadius) mapping HD coords → site.
 *    - Reduction: average/intensify all HD beam contributions within each sampleRadius
 *      to produce a SampleTensor [channels, sites], preserving position metadata.
 *
 * 3. Phosphor Excitation
 *    - Using SampleTensor, apply per-channel offsets and envelope profiles
 *      (riseTime, peakTime, fallTime) to generate time-varying intensity events.
 *    - Events are enqueued in a PixelFrameBuffer<DataType> for double-buffered
 *      causal diffs, enabling full vs differential output modes.
 *
 * 4. Spatial Diffusion and High-Definition Reconstruction
 *    - If useDiffusionGrid enabled:
 *        • DiffusionKernel projects SampleTensor back to an intermediate HD grid
 *          via a circular or custom kernel to simulate phosphor spread.
 *    - Otherwise, tightly focus per-site energy onto a pixel grid of configurable
 *      resolution using interpolation routines (TODO: insert Interpolator module).
 *
 * 5. Output Modes
 *    - Full Mode: renderBuffer() returns full current buffer tensor [channels, rows, cols]
 *    - Diff Mode: getDiffAndSwap(threshold) returns only changed cells for fast updates.
 *
 * 6. ASCII Conversion (Separate Module)
 *    - Takes final low-res or HD tensor and maps intensities to ASCII via CharClassifier.
 *    - Supports channel-blend or discrete-channel rendering per user configuration.
 *    - Outputs ANSI control sequences for location and color per diff or full-print.
 *
 * Thread Safety & Pipeline
 *   - Each stage buffers via PixelFrameBuffer<T> with std::mutex locking.
 *   - Pipeline executes as a bucket-brigade of time-sliced tensors,
 *     handing off between electron gun, sampler, phosphor, diffusion, and ASCII.
 *
 * This design ensures full high-definition fidelity through each transformation,
 * deferring resolution reduction until explicit sampling, while enabling both
 * full-frame and sparse updates for performance. All extension points (e.g. Interpolator,
 * custom decay curves, beam shape parameters) are marked TODO for customer implementation.
 */

/**
 * EnvelopeSettings
 * ----------------
 * Defines a customizable excitation envelope for phosphor channels,
 * including rise time, hold time, and decay time (seconds).
 */
struct EnvelopeSettings {
    double riseTime;   // Seconds to ramp from 0 to peak
    double peakTime;   // Seconds to hold at peak intensity
    double fallTime;   // Seconds to decay from peak to 0
};

/**
 * PhosphorScreen<DataType>
 * -------------------------
 * Simulates a multi-channel phosphor screen with per-channel deposition offsets
 * and envelope-based excitation/decay profiles.
 * Integrates a PixelFrameBuffer<DataType> for event-driven state management.
 *
 * Responsibilities:
 *   - Maintain an event queue of energy depositions with timestamps
 *   - Apply per-channel time offsets and envelope shapes
 *   - Compute time-varying intensity for each pixel and channel
 *   - Provide full buffer or diffs for rendering (tensor of [channels, rows, cols])
 *
 * Thread Safety:
 *   - All public methods are protected by a mutex for concurrent access
 *
 * @tparam DataType  Numeric type for energy/intensity (e.g., float or uint8_t)
 */
template<typename DataType = float>
class PhosphorScreen {
public:
    /**
     * Constructor
     *
     * @param rows              Screen height (pixels)
     * @param cols              Screen width (pixels)
     * @param channels          Number of phosphor channels
     * @param decayRate         Exponential decay rate fallback (per second)
     * @param channelOffsets    Per-channel time offsets (seconds)
     * @param envelope          Per-channel excitation envelope settings
     * @param useDiffusionGrid  If true, applies spatial diffusion using DiffusionKernel
     * @param diffusionRadius   Radius for diffusion kernel (pixels)
     * @param diffusionStrength Strength factor for diffusion kernel [0..1]
     */
    PhosphorScreen(int rows,
                   int cols,
                   int channels,
                   double decayRate,
                   const std::vector<double>& channelOffsets,
                   const std::vector<EnvelopeSettings>& envelope,
                   bool useDiffusionGrid = false,
                   int diffusionRadius = 1,
                   DataType diffusionStrength = static_cast<DataType>(0.5));

    /**
     * excite
     * ------
     * Enqueue an excitation event at pixel (x,y) for a specific channel.
     * The event time is adjusted by the channel offset and envelope start.
     *
     * @param x         Column index [0, cols)
     * @param y         Row index [0, rows)
     * @param channel   Channel index [0, channels)
     * @param value     Energy value (normalized or raw)
     * @param timestamp Event timestamp (seconds)
     */
    void excite(int x,
                int y,
                int channel,
                DataType value,
                double timestamp);

    /**
     * renderBuffer
     * ------------
     * Compute the current screen state by applying decay to all queued events
     * relative to the provided current time. Returns a tensor of intensities.
     *
     * @param currentTime  Reference time (seconds) for decay computation
     * @return Eigen::Tensor<DataType,3> with shape [channels, rows, cols]
     *
     * Steps:
     *   1. Lock mutex and copy & clear event queue
     *   2. For each event: compute decayed value = value * exp(-decayRate * dt)
     *      where dt = currentTime - (timestamp + channelOffset)
     *   3. Accumulate decayed intensities into output tensor
     */
    Eigen::Tensor<DataType,3>
    renderBuffer(double currentTime) const;

    /**
     * applyDiffusion
     * --------------
     * Applies the spatial diffusion kernel to the rendered buffer.
     * Called internally if useDiffusionGrid_ is true.
     *
     * @param buffer  Tensor<DataType,3> [channels, rows, cols]
     */
    void applyDiffusion(Eigen::Tensor<DataType,3>& buffer) const;

private:
    int rows_, cols_, channels_;
    double decayRate_;
    std::vector<double> channelOffsets_;
    std::vector<EnvelopeSettings> envelopeSettings_;
    bool useDiffusionGrid_;
    DiffusionKernel<DataType> diffusionKernel_;
    mutable std::deque<std::tuple<int, int, int, DataType, double>> eventQueue_;
    mutable std::mutex mutex_;
};

} // namespace asciioscilliscope
