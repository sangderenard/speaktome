#include "../include/asciioscilliscope/PhosphorScreen.h"

namespace asciioscilliscope {

template<typename DataType>
PhosphorScreen<DataType>::PhosphorScreen(int rows,
                                         int cols,
                                         int channels,
                                         double decayRate,
                                         const std::vector<double>& channelOffsets,
                                         const std::vector<EnvelopeSettings>& envelope,
                                         bool useDiffusionGrid,
                                         int diffusionRadius,
                                         DataType diffusionStrength)
    : rows_(rows), cols_(cols), channels_(channels), decayRate_(decayRate),
      channelOffsets_(channelOffsets), envelopeSettings_(envelope),
      useDiffusionGrid_(useDiffusionGrid),
      diffusionKernel_(diffusionRadius, diffusionStrength) {}

template<typename DataType>
void PhosphorScreen<DataType>::excite(int x, int y, int channel, DataType value, double timestamp) {
    (void)x; (void)y; (void)channel; (void)value; (void)timestamp; // STUB
}

template<typename DataType>
Eigen::Tensor<DataType,3> PhosphorScreen<DataType>::renderBuffer(double currentTime) const {
    (void)currentTime;
    Eigen::Tensor<DataType,3> out(channels_, rows_, cols_);
    out.setZero();
    return out;
}

template<typename DataType>
void PhosphorScreen<DataType>::applyDiffusion(Eigen::Tensor<DataType,3>& buffer) const {
    (void)buffer; // STUB
}

template class PhosphorScreen<float>;

} // namespace asciioscilliscope
