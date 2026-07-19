#pragma once
#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include <vector>
#include <mutex>
#include "kernels/DiffusionKernel.h"
#include "EnvelopeSettings.h"

namespace asciioscilliscope {

template<typename T>
class PhosphorScreen {
public:
    PhosphorScreen(int rows,
                   int cols,
                   int channels,
                   double decayRate,
                   const std::vector<double>& channelOffsets,
                   const std::vector<EnvelopeSettings>& envelope,
                   bool useDiffusionGrid,
                   int diffusionRadius,
                   T diffusionStrength);

    // Queue an excitation event
    void excite(int x, int y, int channel, T value, double timestamp);

    // Render current buffer with decay and envelope
    Eigen::Tensor<T,3> renderBuffer(double currentTime) const;

    // Apply spatial diffusion across channels
    void applyDiffusion(Eigen::Tensor<T,3>& buffer) const;

private:
    int rows_, cols_, channels_;
    double decayRate_;
    std::vector<double> channelOffsets_;
    std::vector<EnvelopeSettings> envelopeSettings_;
    bool useDiffusionGrid_;
    kernels::IDiffusionKernel<T>* diffusionKernel_;
    std::vector<std::tuple<int,int,int,T,double>> eventQueue_;
    mutable std::mutex mutex_;
};

} // namespace asciioscilliscope
