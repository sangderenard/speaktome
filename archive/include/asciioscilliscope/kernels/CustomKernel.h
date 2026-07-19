#pragma once
#include "DiffusionKernel.h"

namespace asciioscilliscope {

template<typename T>
class CustomKernel : public IDiffusionKernel<T> {
public:
    CustomKernel(int radius, T strength, int stride, int featureDims);
    void apply(const Eigen::Tensor<T,2>& input,
               Eigen::Tensor<T,2>& output,
               double timestamp = 0.0) const override;
private:
    int radius_, stride_, featureDims_;
    T strength_;
};

} // namespace asciioscilliscope
