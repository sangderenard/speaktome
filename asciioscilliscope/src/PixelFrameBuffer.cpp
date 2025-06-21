#include "../include/asciioscilliscope/PixelFrameBuffer.h"

namespace asciioscilliscope {

// ########## STUB: PixelFrameBuffer Constructor ##########
// PURPOSE: allocate curr_/prev_ tensors and store dimensions.
// EXPECTED BEHAVIOR: maintain double-buffered state for diff calculations.
// INPUTS: batch, timeSteps, channels, rows, cols
// OUTPUTS: internal tensors ready for updates
// TODO:
//   - Initialize mutex once multithreading is enabled
// ########################################################
template<typename DataType>
PixelFrameBuffer<DataType>::PixelFrameBuffer(int batch, int timeSteps, int channels, int rows, int cols)
    : batch_(batch), timeSteps_(timeSteps), channels_(channels), rows_(rows), cols_(cols),
      curr_(batch, timeSteps, channels, rows, cols),
      prev_(batch, timeSteps, channels, rows, cols) {}

// ########## STUB: updateRender ##########
// PURPOSE: copy incoming tensor into current buffer.
// EXPECTED BEHAVIOR: queueing and locking will be added later.
// ########################################################
template<typename DataType>
void PixelFrameBuffer<DataType>::updateRender(const Eigen::Tensor<DataType,5>& data) {
    curr_ = data;
}

// ########## STUB: getDiffAndSwap ##########
// PURPOSE: compute sparse diffs above threshold.
// EXPECTED BEHAVIOR: produce tuples of changed locations.
// ########################################################
template<typename DataType>
std::vector<std::tuple<int,int,int,int,int,DataType>>
PixelFrameBuffer<DataType>::getDiffAndSwap(DataType threshold) {
    (void)threshold; // placeholder until diff logic implemented
    std::vector<std::tuple<int,int,int,int,int,DataType>> result;
    prev_ = curr_;
    return result;
}

// explicit instantiation for float
template class PixelFrameBuffer<float>;

} // namespace asciioscilliscope
