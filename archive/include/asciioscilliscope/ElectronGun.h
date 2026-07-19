#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include "Geometry3D.h"

namespace asciioscilliscope {

template<typename T=float>
class ElectronGun {
public:
    ElectronGun(int hdRows, int hdCols, int channels, const TrapezoidalPyramid& geom);
    void setFocusParameters(T confinementRadius, T fieldCurvature, T beamAngle, T gain);
    void attachSignalGenerator(int channel, double frequency);
    Eigen::Tensor<T,3> emitDiffMask(int timeStep);
    void simulateReflection(int timeStep);
    void reset();
private:
    int hdRows_, hdCols_, channels_;
    TrapezoidalPyramid geometry_;
    T confinementRadius_, fieldCurvature_, beamAngle_, gain_;
    int currentStep_;
};

} // namespace asciioscilliscope
