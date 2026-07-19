#pragma once
#include "TensorAliases.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace asciioscilliscope::kernels {

template<typename T>
class TemporalLinearKernel {
public:
    TemporalLinearKernel(int ts);
    // Build a simple linear interpolation kernel along time dimension
    PhosphorBuffer4D<T> build(int c, int h, int w) const;

private:
    int ts_;
};

} // namespace asciioscilliscope::kernels
