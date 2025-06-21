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
void SampleSiteGrid<DataType>::initMetadata() {
    siteMetadata_.clear();
    for (int r=0; r<siteRows_; ++r) {
        for (int c=0; c<siteCols_; ++c) {
            siteMetadata_.emplace_back(r, c, r*radius_, c*radius_, radius_);
        }
    }
}

template<typename DataType>
Eigen::Tensor<DataType,2>
SampleSiteGrid<DataType>::reduceHdTensor(const Eigen::Tensor<DataType,3>& hdTensor) const {
    // ########## STUB ########## return zero tensor of expected shape
    Eigen::Tensor<DataType,2> out( hdTensor.dimension(0), siteRows_*siteCols_ );
    out.setZero();
    (void)hdTensor;
    return out;
}

template<typename DataType>
std::vector<std::tuple<int,int,int,int,int>> SampleSiteGrid<DataType>::getSiteMetadata() const {
    return siteMetadata_;
}

template class SampleSiteGrid<float>;

} // namespace asciioscilliscope
