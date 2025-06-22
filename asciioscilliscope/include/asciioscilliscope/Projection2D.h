#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

namespace asciioscilliscope {

/**
 * Projection2D<DataType>
 * -----------------------
 * Computes the 2D projection mask of a beam path through a trapezoidal tube cross-section.
 * Utilizes piecewise linear inequalities to determine if each pixel lies within the beam's trapezoid.
 *
 * Responsibilities:
 *   - Given beam parameters (nearWidth, farWidth, depth), compute intersection region mask
 *   - Support batch processing via Eigen tensor operations
 *
 * @tparam DataType Numeric type for parameter and computations (e.g., float)
 */
template<typename DataType = float>
class Projection2D {
public:
    /**
     * projectTrapezoid
     * ----------------
     * Generates a boolean mask tensor [rows, cols], where true indicates the beam intersects.
     * The trapezoid is defined by a nearWidth at row 0, a farWidth at row depthRows-1,
     * and straight sides connecting the edges.
     *
     * @param rows       Number of rows in the output mask
     * @param cols       Number of columns in the output mask
     * @param nearWidth  Beam width at the near plane (top)
     * @param farWidth   Beam width at the far plane (bottom)
     * @return Eigen::Tensor<bool,2> mask of shape [rows, cols]
     */
    static Eigen::Tensor<bool,2> projectTrapezoid(
        int rows,
        int cols,
        DataType nearWidth,
        DataType farWidth
    );
};

} // namespace asciioscilliscope
