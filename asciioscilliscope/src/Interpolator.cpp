#include "../include/asciioscilliscope/Interpolator.h"

namespace asciioscilliscope {

template<typename DataType>
Interpolator<DataType>::Interpolator(Mode mode)
    : mode_(mode) {}

template<typename DataType>
Eigen::Tensor<DataType,3> Interpolator<DataType>::resample(const Eigen::Tensor<DataType,3>& input,
                                                           int outRows,
                                                           int outCols) const {
    // ########## STUB ########## naive resize using setZero and cropping
    Eigen::Tensor<DataType,3> out(input.dimension(0), outRows, outCols);
    out.setZero();
    (void)input;
    return out;
}

template class Interpolator<float>;

} // namespace asciioscilliscope
