#pragma once
#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include <memory>

namespace asciioscilliscope {

// The mode selector
enum class KernelMode {
    Simple,       // fixed convolution
    Custom,       // stride & feature dims
    Dynamic,      // timestamp-aware
    FullComplex   // spline-based dynamic
};

// Interface
template<typename T>
class IDiffusionKernel {
public:
    virtual ~IDiffusionKernel() = default;
    virtual void apply(const Eigen::Tensor<T,2>& input,
                       Eigen::Tensor<T,2>& output,
                       double timestamp = 0.0) const = 0;
};

// Factory
template<typename T>
std::unique_ptr<IDiffusionKernel<T>>
makeDiffusionKernel(KernelMode mode, int radius, T strength);

} // namespace asciioscilliscope
