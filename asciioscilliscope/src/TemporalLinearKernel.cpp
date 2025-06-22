#include "TemporalLinearKernel.h"

namespace asciioscilliscope::kernels {

template<typename T>
TemporalLinearKernel<T>::TemporalLinearKernel(int ts)
  : ts_(ts) {}

template<typename T>
PhosphorBuffer4D<T> TemporalLinearKernel<T>::build(int c, int h, int w) const {
    // Kernel shape: [ts, 1, 1, 1] broadcastable to [ts, c, h, w]
    PhosphorBuffer4D<T> kernel(ts_, c, h, w);
    // Linear weights from 0 to 1 across time
    for(int t=0; t<ts_; ++t) {
        T wgt = T(t) / T(ts_-1);
        for(int cc=0; cc<c; ++cc)
            for(int yy=0; yy<h; ++yy)
                for(int xx=0; xx<w; ++xx)
                    kernel(t,cc,yy,xx) = wgt;
    }
    // Normalize across time dimension so sum of weights = 1 at each (c,h,w)
    kernel = kernel / kernel.sum(Eigen::array<int,3>{0});
    return kernel;
}

// Explicit instantiation
template class TemporalLinearKernel<float>;
template class TemporalLinearKernel<double>;

} // namespace asciioscilliscope::kernels
