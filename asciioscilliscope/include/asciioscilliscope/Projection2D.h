#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include "Geometry3D.h"

namespace asciioscilliscope {

/**
 * Maps 3D trapezoidal pyramid interior to a continuous 2D weight map.
 * Outputs attenuation values [0..1] based on geometry, not binary mask.
 */
template<typename T=float>
class Projection2D {
public:
    // rows x cols output tensor of weights
    static Eigen::Tensor<T,2> projectTrapezoid(
        int rows, int cols,
        const TrapezoidalPyramid& geom
    );
};

} // namespace asciioscilliscope
