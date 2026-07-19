#include "DiffusionKernel.h"
#include "SimpleKernel.h"
#include "CustomKernel.h"
#include "DynamicKernel.h"
#include "ComplexKernel.h"

namespace asciioscilliscope {
template<typename T>
std::unique_ptr<IDiffusionKernel<T>> makeDiffusionKernel(KernelMode mode, int radius, T strength) {
    switch (mode) {
        case KernelMode::Simple:
            return std::make_unique<SimpleKernel<T>>(radius, strength);
        case KernelMode::Custom:
            return std::make_unique<CustomKernel<T>>(radius, strength, 1, 4);
        case KernelMode::Dynamic:
            return std::make_unique<DynamicKernel<T>>(radius, strength);
        case KernelMode::FullComplex:
            return std::make_unique<ComplexKernel<T>>(radius, strength);
        default:
            return nullptr;
    }
}
template class std::unique_ptr<IDiffusionKernel<float>> makeDiffusionKernel<float>(KernelMode, int, float);
template class std::unique_ptr<IDiffusionKernel<double>> makeDiffusionKernel<double>(KernelMode, int, double);
} // namespace asciioscilliscope
