#include "CustomKernel.h"

namespace asciioscilliscope {

template<typename T>
CustomKernel<T>::CustomKernel(int radius, T strength, int stride, int featureDims)
 : radius_(radius), stride_(stride), featureDims_(featureDims), strength_(strength) {
    // STUB: precompute custom features
}

template<typename T>
void CustomKernel<T>::apply(const Eigen::Tensor<T,2>& input,
                            Eigen::Tensor<T,2>& output,
                            double) const {
    // STUB: implement custom diffusion
    output = input;
}

template class CustomKernel<float>;
template class CustomKernel<double>;

} // namespace asciioscilliscope
