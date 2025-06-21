#include "../include/asciioscilliscope/PhosphorScreen.h"

namespace asciioscilliscope {

template<typename DataType>
PhosphorScreen<DataType>::PhosphorScreen(
    int rows,
    int cols,
    int channels,
    double decayRate,
    const std::vector<double>& channelOffsets,
    const std::vector<EnvelopeSettings>& envelope,
    bool useDiffusionGrid,
    int diffusionRadius,
    DataType diffusionStrength)
    : rows_(rows), cols_(cols), channels_(channels),
      decayRate_(decayRate), channelOffsets_(channelOffsets),
      envelopeSettings_(envelope), useDiffusionGrid_(useDiffusionGrid),
      diffusionKernel_(diffusionRadius, diffusionStrength) {}

// ########## STUB: PhosphorScreen::excite ##########
template<typename DataType>
void PhosphorScreen<DataType>::excite(int x,
                                      int y,
                                      int channel,
                                      DataType value,
                                      double timestamp) {
    (void)x; (void)y; (void)channel; (void)value; (void)timestamp;
}

// ########## STUB: PhosphorScreen::renderBuffer ##########
template<typename DataType>
Eigen::Tensor<DataType,3>
PhosphorScreen<DataType>::renderBuffer(double currentTime) const {
    (void)currentTime;
    return Eigen::Tensor<DataType,3>(channels_, rows_, cols_);
}

// ########## STUB: PhosphorScreen::applyDiffusion ##########
template<typename DataType>
void PhosphorScreen<DataType>::applyDiffusion(Eigen::Tensor<DataType,3>& buffer) const {
    if (!useDiffusionGrid_) return;
    for(int c=0;c<channels_;++c) {
        Eigen::Tensor<DataType,2> tmp(rows_, cols_);
        diffusionKernel_.apply(buffer.chip(c,0), tmp);
        buffer.chip(c,0) = tmp;
    }
}

template class PhosphorScreen<float>;

} // namespace asciioscilliscope
