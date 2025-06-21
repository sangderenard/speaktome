#include "../include/asciioscilliscope/ElectronGun.h"

namespace asciioscilliscope {

template<typename DataType>
ElectronGun<DataType>::ElectronGun(int hdRows, int hdCols, int channels)
    : hdRows_(hdRows), hdCols_(hdCols), channels_(channels),
      confinementRadius_(0), fieldCurvature_(0),
      beamAngle_(0), gain_(1), currentStep_(0) {}

// ########## STUB: ElectronGun::setFocusParameters ##########
template<typename DataType>
void ElectronGun<DataType>::setFocusParameters(DataType confinementRadius,
                                               DataType fieldCurvature,
                                               DataType beamAngle,
                                               DataType gain) {
    confinementRadius_ = confinementRadius;
    fieldCurvature_ = fieldCurvature;
    beamAngle_ = beamAngle;
    gain_ = gain;
}

// ########## STUB: ElectronGun::emitDiffMask ##########
template<typename DataType>
Eigen::Tensor<DataType,3> ElectronGun<DataType>::emitDiffMask(int timeStep) {
    currentStep_ = timeStep;
    return Eigen::Tensor<DataType,3>(channels_, hdRows_, hdCols_);
}

template<typename DataType>
void ElectronGun<DataType>::reset() {
    currentStep_ = 0;
}

template class ElectronGun<float>;

} // namespace asciioscilliscope
