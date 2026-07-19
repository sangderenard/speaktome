#pragma once
#include "DiffusionKernel.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace asciioscilliscope {

template<typename T>
class SimpleKernel : public IDiffusionKernel<T> {
public:
    SimpleKernel(int radius, T strength);
    void apply(const Eigen::Tensor<T,2>& input,
               Eigen::Tensor<T,2>& output,
               double timestamp = 0.0) const override;
private:
    int radius_;
    T strength_;
    Eigen::Tensor<T,2> kernel_;
};

} // namespace asciioscilliscope
