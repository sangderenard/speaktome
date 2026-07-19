#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace asciioscilliscope {

/**
 * DiffusionKernel
 * ---------------
 * Provides utilities to apply spatial diffusion (circular kernel) over a 2D grid.
 * Can diffuse energy values across neighboring pixels based on a radius and strength.
 *
 * @tparam DataType  Numeric type of the grid (e.g., float)
 */
template<typename DataType = float>
class DiffusionKernel {
public:
    /**
     * Constructor
     *
     * @param radius   Diffusion radius in pixels
     * @param strength Diffusion strength factor [0..1]
     */
    DiffusionKernel(int radius, DataType strength);

    /**
     * apply
     * -----
     * Applies circular diffusion over the input grid, producing an output grid.
     *
     * @param input   Eigen::Tensor<DataType,2> input [rows, cols]
     * @param output  Eigen::Tensor<DataType,2> output [rows, cols]
     */
    void apply(const Eigen::Tensor<DataType,2>& input,
               Eigen::Tensor<DataType,2>& output) const;

private:
    int radius_;
    DataType strength_;
};

} // namespace asciioscilliscope
