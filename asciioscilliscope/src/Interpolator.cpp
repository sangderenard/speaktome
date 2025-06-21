#include "../include/asciioscilliscope/Interpolator.h"

namespace asciioscilliscope {

template<typename DataType>
Interpolator<DataType>::Interpolator(Mode mode) : mode_(mode) {}

// ########## STUB: Interpolator::resample ##########
template<typename DataType>
Eigen::Tensor<DataType,3>
Interpolator<DataType>::resample(const Eigen::Tensor<DataType,3>& input,
                                 int outRows,
                                 int outCols) const {
    return Eigen::Tensor<DataType,3>(input.dimension(0), outRows, outCols);
}

template class Interpolator<float>;

} // namespace asciioscilliscope
