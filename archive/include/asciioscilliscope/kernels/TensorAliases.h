#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
namespace asciioscilliscope::kernels {
  template<typename T> using PhosphorBuffer4D = Eigen::Tensor<T,4>;
  template<typename T> using BeamMask3D = Eigen::Tensor<T,4>;
}
