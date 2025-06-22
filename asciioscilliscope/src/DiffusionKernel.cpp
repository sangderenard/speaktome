#include "../include/asciioscilliscope/DiffusionKernel.h"

namespace asciioscilliscope {

template<typename DataType>
DiffusionKernel<DataType>::DiffusionKernel(int radius, DataType strength)

    : radius_(radius), strength_(strength) {}

// ########## STUB: DiffusionKernel::apply ##########
template<typename DataType>
void DiffusionKernel<DataType>::apply(const Eigen::Tensor<DataType,2>& input,
                                      Eigen::Tensor<DataType,2>& output) const {
    output = input;
}


template class DiffusionKernel<float>;

} // namespace asciioscilliscope
