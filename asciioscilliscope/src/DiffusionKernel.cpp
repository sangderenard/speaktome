#include "../include/asciioscilliscope/DiffusionKernel.h"

namespace asciioscilliscope {

template<typename DataType>
DiffusionKernel<DataType>::DiffusionKernel(int radius, DataType strength)
    : radius_(radius), strength_(strength) {}

template<typename DataType>
void DiffusionKernel<DataType>::apply(const Eigen::Tensor<DataType,2>& input,
                                      Eigen::Tensor<DataType,2>& output) const {
    // ########## STUB ########## copy input to output
    output = input;
    (void)radius_;
    (void)strength_;
}

template class DiffusionKernel<float>;

} // namespace asciioscilliscope
