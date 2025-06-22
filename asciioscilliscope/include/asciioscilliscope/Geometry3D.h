#pragma once
#include "eigen/unsupported/Eigen/CXX11/Tensor"

/**
 * Geometry3D
 * ----------
 * Defines a trapezoidal pyramid for CRT tube interior.
 * Represents near and far rectangle dimensions and depth.
 */
namespace asciioscilliscope {

struct TrapezoidalPyramid {
    float nearWidth, nearHeight;
    float farWidth, farHeight;
    float depth;
    float reflectivity;
};

} // namespace asciioscilliscope
