#include "../include/asciioscilliscope/BeamFocus.h"

namespace asciioscilliscope {

template<typename DataType>
BeamFocus<DataType>::BeamFocus(int hdRows, int hdCols, int channels)
    : hdRows_(hdRows), hdCols_(hdCols), channels_(channels),
      useCircularKernel_(false), kernelRadius_(1) {}

// ########## STUB: BeamFocus::setGridParameters ##########
template<typename DataType>
void BeamFocus<DataType>::setGridParameters(bool useCircularKernel, int kernelRadius) {
    useCircularKernel_ = useCircularKernel;
    kernelRadius_ = kernelRadius;
}

// ########## STUB: BeamFocus::processMask ##########
template<typename DataType>
Eigen::Tensor<DataType,3> BeamFocus<DataType>::processMask(const Eigen::Tensor<DataType,3>& inputMask) const {
    return inputMask;
}

template class BeamFocus<float>;

} // namespace asciioscilliscope
