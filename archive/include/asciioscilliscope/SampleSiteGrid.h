#pragma once
#include <vector>
#include <tuple>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace asciioscilliscope {

/**
 * SampleSiteGrid
 * --------------
 * Defines a reduced-resolution set of sampling sites over a high-definition plane.
 * Aggregates HD tensor values into per-site intensities via spatial mapping.
 *
 * Template Parameters:
 *   - DataType: Numeric type for HD and sample values (e.g., float)
 *
 * Responsibilities:
 *   - Define sampling grid dimensions (numRows x numCols) and per-site radius
 *   - Store metadata for each site: physical plane coordinates and radius
 *   - Map HD tensor [channels, HD_rows, HD_cols] into SampleTensor [channels, numSites]
 *   - Provide inverse mapping for reconstruction if needed
 *
 * Use Cases:
 *   - Channel-specific sampling for CRT phosphor grids
 *   - SIMD-accelerated aggregation of HD energy masks
 *   - Feeding reduced tensor into PixelFrameBuffer or phosphor simulation
 */
template<typename DataType = float>
class SampleSiteGrid {
public:
    /**
     * Constructor
     *
     * @param hdRows      Full-resolution height
     * @param hdCols      Full-resolution width
     * @param siteRows    Number of sample rows
     * @param siteCols    Number of sample columns
     * @param radius      Sampling radius in HD pixels per site
     */
    SampleSiteGrid(int hdRows,
                   int hdCols,
                   int siteRows,
                   int siteCols,
                   int radius);

    /**
     * reduceHdTensor
     * --------------
     * Aggregates an HD tensor [channels, HD_rows, HD_cols] into
     * a sample tensor [channels, siteRows*siteCols], computing
     * the average intensity within each site's radius.
     *
     * @param hdTensor  Input HD tensor
     * @return Eigen::Tensor<DataType,2> SampleTensor [channels, numSites]
     */
    Eigen::Tensor<DataType,2>
    reduceHdTensor(const Eigen::Tensor<DataType,3>& hdTensor) const;

    /**
     * getSiteMetadata
     * ---------------
     * Returns a vector of tuples containing (siteRow, siteCol, hdRowCenter, hdColCenter, radius)
     * for each sampling site.
     */
    std::vector<std::tuple<int,int,int,int,int>>
    getSiteMetadata() const;

private:
    int hdRows_, hdCols_, siteRows_, siteCols_, radius_;
    std::vector<std::tuple<int,int,int,int,int>> siteMetadata_;

    void initMetadata();
};

} // namespace asciioscilliscope
