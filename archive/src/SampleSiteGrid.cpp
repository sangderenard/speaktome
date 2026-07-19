#include "../include/asciioscilliscope/SampleSiteGrid.h"
#include <algorithm>

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
    // PURPOSE: aggregate HD tensor values into sample sites using IsoShell
    //          weighting derived from the conic beam projection.
    // EXPECTED BEHAVIOR: Each site's intensity is computed by intersecting the
    //          beam cone with the trapezoidal CRT volume and applying a
    //          CharClassifier-generated kernel. The function must support
    //          off-axis steering and magnetic curvature.
    // INPUTS: hdTensor with shape [channels, hdRows, hdCols].
    // OUTPUTS: Sample tensor with shape [channels, siteRows*siteCols].
    // KEY ASSUMPTIONS/DEPENDENCIES:
    //   - ConicProjector3D provides beam geometry.
    //   - CharClassifier supplies per-channel weighting kernels.
    // TODO:
    //   - Integrate IsoShell sampling from ConicProjector3D.
    //   - Apply CharClassifier kernels for weighted reduction.
    //   - Record offsets caused by magnetic curvature.
    // NOTES: This placeholder simply returns a zero tensor so other
    //         components compile.
    // ######################################################################

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
    // PURPOSE: prepare site metadata informed by IsoShell geometry.
    // EXPECTED BEHAVIOR: compute accurate centers and radii for each sample
    //          site using the beam's conic projection and trapezoidal bounds.
    // TODO:
    //   - Calculate true intersection centers based on IsoShell sampling.
    //   - Support non-uniform layouts and beam steering offsets.
    // ###########################################

    siteMetadata_.clear();
    // Placeholder metadata so callers have predictable structure.
    // Centers are set to (0,0) and should be recomputed when real
    // projection logic is implemented.
    for (int r = 0; r < siteRows_; ++r) {
        for (int c = 0; c < siteCols_; ++c) {
            siteMetadata_.emplace_back(r, c, 0, 0, radius_);
        }
    }
}

template class SampleSiteGrid<float>;

} // namespace asciioscilliscope
