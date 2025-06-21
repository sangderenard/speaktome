#include "../include/asciioscilliscope/DiffusionKernel.h"

namespace asciioscilliscope {

template<typename DataType>
DiffusionKernel<DataType>::DiffusionKernel(int radius, DataType strength)
    : radius_(radius), strength_(strength) {
    // ########## STUB: DiffusionKernel Constructor ##########
    // PURPOSE: store kernel parameters.
    // EXPECTED BEHAVIOR: precompute convolution weights.
    // TODO: implement kernel weight generation.
    // ########################################################
}

template<typename DataType>
void DiffusionKernel<DataType>::apply(const Eigen::Tensor<DataType,2>& input,
                                      Eigen::Tensor<DataType,2>& output) const {
    // ########## STUB: apply ##########
    // PURPOSE: diffuse values of input onto output using circular kernel.
    // EXPECTED BEHAVIOR: produce smoothed tensor based on radius and strength.
    // TODO: implement convolution loop with Eigen.
    // #################################
    output = input;
}

// explicit instantiation
template class DiffusionKernel<float>;

} // namespace asciioscilliscope
