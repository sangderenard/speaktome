#pragma once
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace asciioscilliscope {

/**
 * BeamFocus<DataType>
 * --------------------
 * Wraps a PixelFrameBuffer<DataType> to apply beam shaping and spatial focusing
 * onto a high-definition buffer before phosphor sampling.
 *
 * Responsibilities:
 *   - Accept ElectronGun diff masks
 *   - Apply mathematical transforms for focus grid or pixel grid mapping
 *   - Output a focused HD tensor for downstream sample-site reduction
 *
 * API Methods:
 *   - setGridParameters(...) : choose circular kernel vs grid sampling
 *   - processMask(inputMask) : Eigen::Tensor<DataType,3> focused buffer
 */
template<typename DataType = float>
class BeamFocus {
public:
    /**
     * Constructor
     *
     * @param hdRows     HD plane height
     * @param hdCols     HD plane width
     * @param channels   Number of channels
     */
    BeamFocus(int hdRows, int hdCols, int channels);

    /**
     * setGridParameters
     * -----------------
     * Configure whether to use a circular diffusion kernel or pixel grid mapping.
     *
     * @param useCircularKernel  If true, use circular kernel; else use grid mapping
     * @param kernelRadius       Radius for circular kernel (pixels)
     */
    void setGridParameters(bool useCircularKernel, int kernelRadius);

    /**
     * processMask
     * -----------
     * Apply focus or diffusion over the input diff mask.
     *
     * @param inputMask  Tensor<DataType,3> [channels, hdRows, hdCols]
     * @return Focused HD tensor [channels, hdRows, hdCols]
     */
    Eigen::Tensor<DataType,3> processMask(const Eigen::Tensor<DataType,3>& inputMask) const;

private:
    int hdRows_, hdCols_, channels_;
    bool useCircularKernel_;
    int kernelRadius_;
};

} // namespace asciioscilliscope
