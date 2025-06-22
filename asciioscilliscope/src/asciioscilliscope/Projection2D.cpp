#include "asciioscilliscope/Projection2D.h"
#include <cmath>

namespace asciioscilliscope {

template<typename T>
Eigen::Tensor<T,2> Projection2D<T>::projectTrapezoid(
    int rows, int cols,
    const TrapezoidalPyramid& geom
) {
    Eigen::Tensor<T,2> weights(rows, cols);
    // Center coords
    T cx = cols / 2.0;
    T cy = rows / 2.0;
    for(int r=0; r<rows; ++r) {
        // depth interpolation factor
        T t = T(r) / T(rows - 1);
        T halfW = (1 - t) * (geom.nearWidth/2) + t * (geom.farWidth/2);
        T halfH = (1 - t) * (geom.nearHeight/2) + t * (geom.farHeight/2);
        for(int c=0; c<cols; ++c) {
            T x = (c - cx);
            T y = (r - cy);
            // Distance to center of trapezoid cross-section
            T dx = std::abs(x) / halfW;
            T dy = std::abs(y) / halfH;
            T norm = std::max(dx, dy);
            // Weight: 1 inside, falling to 0 at edge + margin
            if(norm <= 1) {
                weights(r, c) = T(1);
            } else {
                // Linear falloff beyond edge up to margin (10% of half)
                T margin = 0.1;
                T v = T(1) - ((norm - 1) / margin);
                weights(r, c) = std::max(T(0), v);
            }
            // Apply reflectivity attenuation
            weights(r, c) *= geom.reflectivity;
        }
    }
    return weights;
}

// Explicit instantiation
template class Projection2D<float>;
template class Projection2D<double>;

} // namespace asciioscilliscope
