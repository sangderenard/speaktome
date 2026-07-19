#pragma once
#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include "Geometry3D.h"

namespace asciioscilliscope {

/**
 * ConicProjector3D
 * ----------------
 * Models projection of an electron beam exiting the yolk after
 * magnetic deflection and confinement focus. Produces a 3D tensor
 * describing beam intensity within a conic volume.
 */
template<typename T=float>
class ConicProjector3D {
public:
    /**
     * projectCone
     * -----------
     * Compute conic projection weights.
     *
     * @param radialSamples  Resolution along cone radius
     * @param angularSamples Number of angular divisions
     * @param depthSamples   Samples along beam depth
     * @param exitAngle      Direction of cone center in radians
     * @return Tensor<T,3>   [depth, radial, angle] weights
     */
    static Eigen::Tensor<T,3> projectCone(
        int radialSamples,
        int angularSamples,
        int depthSamples,
        T exitAngle);
};

} // namespace asciioscilliscope
