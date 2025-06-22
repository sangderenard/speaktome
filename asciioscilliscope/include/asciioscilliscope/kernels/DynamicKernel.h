#pragma once
#include "DiffusionKernel.h"

namespace asciioscilliscope {

template<typename T>
class DynamicKernel : public IDiffusionKernel<T> {
public:
    DynamicKernel(int radius, T strength);
    void apply(const Eigen::Tensor<T,2>& input,
               Eigen::Tensor<T,2>& output,
               double timestamp) const override;
private:
    int radius_;
    T strength_;
};

} // namespace asciioscilliscope
