#pragma once
#include "DiffusionKernel.h"

namespace asciioscilliscope {

template<typename T>
class ComplexKernel : public IDiffusionKernel<T> {
public:
    ComplexKernel(int radius, T strength);
    void apply(const Eigen::Tensor<T,2>& input,
               Eigen::Tensor<T,2>& output,
               double timestamp) const override;
private:
    int radius_;
    T strength_;
    // STUB: spline data members
};

} // namespace asciioscilliscope
