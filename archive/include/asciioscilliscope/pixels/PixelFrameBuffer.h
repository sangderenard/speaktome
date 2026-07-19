#pragma once
#include "../kernels/TensorAliases.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace asciioscilliscope::pixels {

template<typename T>
class PixelFrameBuffer {
public:
  PixelFrameBuffer(int ts, int c, int h, int w)
    : ts_(ts), c_(c), h_(h), w_(w),
      data_(ts, c, h, w) {
    data_.setZero();
  }

  void batchExcite(
    const Eigen::Tensor<int,2>& offsets,
    const Eigen::Tensor<T,2>& values
  ) {
    Eigen::array<int,4> bcastOffs{1, c_, 1, 1};
    auto xOff = offsets.chip(0,1).reshape(Eigen::array<int,4>{ts_,1,h_,w_})
                  .broadcast(bcastOffs);
    auto yOff = offsets.chip(1,1).reshape(Eigen::array<int,4>{ts_,1,h_,w_})
                  .broadcast(bcastOffs);

    Eigen::array<int,4> bcastVal{1,1,h_,w_};
    auto val4 = values.reshape(Eigen::array<int,4>{ts_,c_,1,1})
                   .broadcast(bcastVal);

    // Create coordinate grids
    Eigen::Tensor<int,4> coordX(ts_, c_, h_, w_);
    Eigen::Tensor<int,4> coordY(ts_, c_, h_, w_);
    for(int t=0;t<ts_;++t) for(int cc=0;cc<c_;++cc)
      for(int yy=0;yy<h_;++yy) for(int xx=0;xx<w_;++xx) {
        coordX(t,cc,yy,xx) = xx;
        coordY(t,cc,yy,xx) = yy;
      }

    auto mask = (coordX == xOff).template cast<T>() * (coordY == yOff).template cast<T>();
    data_ = data_ + mask * val4;
  }

  void diffuse(const kernels::PhosphorBuffer4D<T>& kernel4d,
               const Eigen::array<long,4>& pad) {
    data_ = data_.convolution(kernel4d, pad);
  }

  const kernels::PhosphorBuffer4D<T>& data() const { return data_; }

private:
  int ts_, c_, h_, w_;
  kernels::PhosphorBuffer4D<T> data_;
};

} // namespace asciioscilliscope::pixels
