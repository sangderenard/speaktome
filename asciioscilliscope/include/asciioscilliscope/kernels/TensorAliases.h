#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

namespace asciioscilliscope::kernels {
  // Time × Channels × Height × Width
  template<typename T>
  using PhosphorBuffer4D = Eigen::Tensor<T,4>;
}
