#include "../include/asciioscilliscope/Interpolator.h"

namespace asciioscilliscope {

template<typename DataType>
Interpolator<DataType>::Interpolator(Mode mode)
    : mode_(mode) {
    // ########## STUB: Interpolator Constructor ##########
    // PURPOSE: store chosen interpolation mode.
    // TODO: Precompute coefficients for fast resampling.
    // ###############################################
}

template<typename DataType>
Eigen::Tensor<DataType,3> Interpolator<DataType>::resample(const Eigen::Tensor<DataType,3>& input,
                                                           int outRows,
                                                           int outCols) const {
    // ########## STUB: resample ##########
    // PURPOSE: resample input tensor to new dimensions.
    // EXPECTED BEHAVIOR: apply mode-specific interpolation.
    // TODO: implement efficient Eigen-based resampling.
    // ######################################
    Eigen::Tensor<DataType,3> out(input.dimension(0), outRows, outCols);
    out.setZero();
    return out;
}

// explicit instantiation
template class Interpolator<float>;

} // namespace asciioscilliscope
