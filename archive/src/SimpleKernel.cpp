#include "SimpleKernel.h"
#include <cmath>

namespace asciioscilliscope {

template<typename T>
SimpleKernel<T>::SimpleKernel(int radius, T strength)
 : radius_(radius), strength_(strength) {
    int sz = 2*radius_ + 1;
    kernel_ = Eigen::Tensor<T,2>(sz, sz);
    kernel_.setConstant(T(1)/(sz*sz)); // average kernel stub
}

template<typename T>
void SimpleKernel<T>::apply(const Eigen::Tensor<T,2>& input,
                            Eigen::Tensor<T,2>& output,
                            double) const {
    auto kr = kernel_.reverse(Eigen::array<bool,2>{true,true});
    Eigen::array<long,2> pad{radius_, radius_};
    output = input.convolution(kr, pad);
}

template class SimpleKernel<float>;
template class SimpleKernel<double>;

} // namespace asciioscilliscope
