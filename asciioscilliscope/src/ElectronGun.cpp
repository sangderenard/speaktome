#include "asciioscilliscope/ElectronGun.h"
#include "Projection2D.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace asciioscilliscope {

template<typename T>
ElectronGun<T>::ElectronGun(int hdRows, int hdCols, int channels, const TrapezoidalPyramid& geom)
 : hdRows_(hdRows), hdCols_(hdCols), channels_(channels),
   geometry_(geom), confinementRadius_(0), fieldCurvature_(0),
   beamAngle_(0), gain_(1), currentStep_(0) {}

// Set beam focusing parameters
template<typename T>
void ElectronGun<T>::setFocusParameters(T confinementRadius, T fieldCurvature, T beamAngle, T gain) {
    confinementRadius_ = confinementRadius;
    fieldCurvature_ = fieldCurvature;
    beamAngle_ = beamAngle;
    gain_ = gain;
}

// Stub: attach a periodic signal per channel (not implemented)
template<typename T>
void ElectronGun<T>::attachSignalGenerator(int channel, double frequency) {
    // Would store generator info
}

// Emit a 3D mask [channels x rows x cols] of beam intensity diffs
template<typename T>
Eigen::Tensor<T,3> ElectronGun<T>::emitDiffMask(int timeStep) {
    // 1) Project tube geometry to 2D mask
    auto mask2d = Projection2D<T>::projectTrapezoid(hdRows_, hdCols_, geometry_);
    // 2) Initialize 3D tensor
    Eigen::Tensor<T,3> tensor(channels_, hdRows_, hdCols_);
    tensor.setZero();
    // 3) For each channel, offset mask by beam-angle and curvature
    for(int c=0; c<channels_; ++c) {
        // Compute per-channel shift
        int dx = int(std::round(confinementRadius_ * std::sin(beamAngle_ + c)));
        int dy = int(std::round(confinementRadius_ * std::cos(beamAngle_ + c)));
        for(int r=0; r<hdRows_; ++r) {
            for(int col=0; col<hdCols_; ++col) {
                int srcR = r + dy;
                int srcC = col + dx;
                if (srcR>=0 && srcR<hdRows_ && srcC>=0 && srcC<hdCols_) {
                    tensor(c, r, col) = mask2d(srcR, srcC) ? gain_ : T(0);
                }
            }
        }
    }
    currentStep_ = timeStep;
    return tensor;
}

// Simulate a single-bounce reflection by inverting mask intensities
template<typename T>
void ElectronGun<T>::simulateReflection(int timeStep) {
    // Reflectivity attenuation per step
    T factor = T(std::pow(geometry_.reflectivity, timeStep - currentStep_));
    // Could apply to last mask, not stored here
    currentStep_ = timeStep;
}

template<typename T>
void ElectronGun<T>::reset() {
    currentStep_ = 0;
}

template class ElectronGun<float>;
template class ElectronGun<double>;

} // namespace asciioscilliscope
