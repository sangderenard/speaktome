#include "asciioscilliscope/Projection2D.h"
#include <cmath>

namespace asciioscilliscope {

template<typename DataType>
Eigen::Tensor<bool,2> Projection2D<DataType>::projectTrapezoid(
    int rows,
    int cols,
    DataType nearWidth,
    DataType farWidth
) {
    Eigen::Tensor<bool,2> mask(rows, cols);
    mask.setConstant(false);
    if (rows < 1 || cols < 1) return mask;
    for (int i = 0; i < rows; ++i) {
        DataType t = (rows == 1) ? DataType(0) : static_cast<DataType>(i) / static_cast<DataType>(rows - 1);
        DataType width = nearWidth + (farWidth - nearWidth) * t;
        DataType leftDouble = (static_cast<DataType>(cols) - width) / DataType(2);
        int left = static_cast<int>(std::round(leftDouble));
        int w = static_cast<int>(std::round(width));
        int right = left + w;
        for (int j = 0; j < cols; ++j) {
            if (j >= left && j < right) {
                mask(i, j) = true;
            }
        }
    }
    return mask;
}

// Explicit template instantiation
template Eigen::Tensor<bool,2> Projection2D<float>::projectTrapezoid(
    int, int, float, float);
template Eigen::Tensor<bool,2> Projection2D<double>::projectTrapezoid(
    int, int, double, double);

} // namespace asciioscilliscope
