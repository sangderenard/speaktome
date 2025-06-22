#include "asciioscilliscope/Projection2D.h"
#include <cmath>

namespace asciioscilliscope {

// Project trapezoidal pyramid interior to a binary mask
template<typename T>
Eigen::Tensor<bool,2> Projection2D<T>::projectTrapezoid(
    int rows, int cols, const TrapezoidalPyramid& geom) {
    Eigen::Tensor<bool,2> mask(rows, cols);
    // Compute scale factors per row for trapezoid interpolation
    for(int r=0; r<rows; ++r) {
        // Linear interpolate width/height at this depth
        T t = T(r) / T(rows - 1);
        T halfW = (1 - t) * (geom.nearWidth/2) + t * (geom.farWidth/2);
        T halfH = (1 - t) * (geom.nearHeight/2) + t * (geom.farHeight/2);
        for(int c=0; c<cols; ++c) {
            // Center coordinates
            T x = (T(c) - cols/2);
            T y = (T(r) - rows/2);
            // Inside trapezoid if within halfW and halfH
            mask(r, c) = (std::abs(x) <= halfW && std::abs(y) <= halfH);
        }
    }
    return mask;
}

template class Projection2D<float>;
template class Projection2D<double>;

} // namespace asciioscilliscope
