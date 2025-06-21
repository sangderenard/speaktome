#pragma once
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace asciioscilliscope {

/**
 * Interpolator
 * ------------
 * Provides utilities to resample high-definition tensors onto different grids
 * (e.g., downsampling to a pixel grid or upsampling for HD reconstruction).
 * Supports various interpolation methods (nearest, bilinear, bicubic).
 *
 * Template Parameters:
 *   - DataType: Numeric type of tensor elements (e.g., float)
 */
template<typename DataType = float>
class Interpolator {
public:
    /**
     * Supported interpolation modes.
     */
    enum class Mode { Nearest, Bilinear, Bicubic };

    /**
     * Constructor
     *
     * @param mode        Interpolation mode
     */
    explicit Interpolator(Mode mode = Mode::Bilinear);

    /**
     * resample
     * --------
     * Resamples an input tensor to new spatial dimensions.
     *
     * @param input     Eigen::Tensor<DataType,3> of shape [channels, inRows, inCols]
     * @param outRows   Desired output rows
     * @param outCols   Desired output cols
     * @return Eigen::Tensor<DataType,3> of shape [channels, outRows, outCols]
     *
     * Notes:
     *   - Channels are preserved; only spatial dims are interpolated.
     *   - Use SIMD-friendly Eigen operations when possible.
     */
    Eigen::Tensor<DataType,3>
    resample(const Eigen::Tensor<DataType,3>& input,
             int outRows,
             int outCols) const;

private:
    Mode mode_;
};

} // namespace asciioscilliscope
