#include "../include/asciioscilliscope/SampleSiteGrid.h"

namespace asciioscilliscope {

template<typename DataType>
SampleSiteGrid<DataType>::SampleSiteGrid(int hdRows,
                                         int hdCols,
                                         int siteRows,
                                         int siteCols,
                                         int radius)

    : hdRows_(hdRows), hdCols_(hdCols), siteRows_(siteRows), siteCols_(siteCols), radius_(radius) {
    initMetadata();
}

template<typename DataType>
Eigen::Tensor<DataType,2> SampleSiteGrid<DataType>::reduceHdTensor(const Eigen::Tensor<DataType,3>& hdTensor) const {
    // ########## STUB: reduceHdTensor ##########
    // PURPOSE: aggregate HD tensor values into sample sites.
    // EXPECTED BEHAVIOR: average intensities within each site's radius.
    // TODO: implement spatial aggregation using Eigen.
    // ##########################################
    Eigen::Tensor<DataType,2> out(hdTensor.dimension(0), siteRows_ * siteCols_);
    out.setZero();
    return out;
}

template<typename DataType>
std::vector<std::tuple<int,int,int,int,int>> SampleSiteGrid<DataType>::getSiteMetadata() const {

    return siteMetadata_;
}

template<typename DataType>
void SampleSiteGrid<DataType>::initMetadata() {

    // ########## STUB: initMetadata ##########
    // PURPOSE: populate site metadata mapping HD regions to sites.
    // TODO: compute exact hdRowCenter/hdColCenter per site.
    // ########################################
    siteMetadata_.clear();
    for (int r=0; r<siteRows_; ++r) {
        for (int c=0; c<siteCols_; ++c) {
            siteMetadata_.emplace_back(r, c, 0, 0, radius_);
        }
    }
}

template class SampleSiteGrid<float>;

} // namespace asciioscilliscope
