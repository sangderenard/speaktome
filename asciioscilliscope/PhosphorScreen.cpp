#include "PhosphorScreen.h"
#include <cmath>
#include <memory>

namespace asciioscilliscope {

using kernels::KernelMode;
using kernels::makeDiffusionKernel;

template<typename T>
PhosphorScreen<T>::PhosphorScreen(int rows,
                                  int cols,
                                  int channels,
                                  double decayRate,
                                  const std::vector<double>& channelOffsets,
                                  const std::vector<EnvelopeSettings>& envelope,
                                  bool useDiffusionGrid,
                                  int diffusionRadius,
                                  T diffusionStrength)
    : rows_(rows), cols_(cols), channels_(channels),
      decayRate_(decayRate), channelOffsets_(channelOffsets),
      envelopeSettings_(envelope), useDiffusionGrid_(useDiffusionGrid)
{
    // Set up diffusion kernel based on mode
    diffusionKernel_ = makeDiffusionKernel<T>(
        useDiffusionGrid_ ? KernelMode::Simple : KernelMode::FullComplex,
        diffusionRadius, diffusionStrength
    );
    // Initialize event queue
}

template<typename T>
void PhosphorScreen<T>::excite(int x, int y, int channel, T value, double timestamp) {
    // STUB: enqueue excitation
    std::lock_guard<std::mutex> lock(mutex_);
    eventQueue_.emplace_back(x, y, channel, value, timestamp);
}

template<typename T>
Eigen::Tensor<T,3> PhosphorScreen<T>::renderBuffer(double currentTime) const {
    // STUB: compute decayed intensities and apply envelope
    Eigen::Tensor<T,3> buffer(channels_, rows_, cols_);
    buffer.setZero();
    return buffer;
}

template<typename T>
void PhosphorScreen<T>::applyDiffusion(Eigen::Tensor<T,3>& buffer) const {
    // STUB: apply diffusion per channel
    if (!useDiffusionGrid_) return;
    for (int c = 0; c < channels_; ++c) {
        Eigen::Tensor<T,2> slice(rows_, cols_);
        diffusionKernel_->apply(buffer.chip(c, 0), slice);
        buffer.chip(c, 0) = slice;
    }
}

// Explicit instantiation
template class PhosphorScreen<float>;
template class PhosphorScreen<double>;

} // namespace asciioscilliscope
