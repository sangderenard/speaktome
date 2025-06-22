#include "ComplexKernel.h"

namespace asciioscilliscope {

template<typename T>
ComplexKernel<T>::ComplexKernel(int radius, T strength)
 : radius_(radius), strength_(strength) {
    // STUB: prepare spline interpolation data
}

template<typename T>
void ComplexKernel<T>::apply(const Eigen::Tensor<T,2>& input,
                             Eigen::Tensor<T,2>& output,
                             double timestamp) const {
    // STUB: implement location-dependent spline diffusion
    output = input;
}

template class ComplexKernel<float>;
template class ComplexKernel<double>;

} // namespace asciioscilliscope
