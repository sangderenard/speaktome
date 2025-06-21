#include "../include/asciioscilliscope/PixelFrameBuffer.h"

namespace asciioscilliscope {

template<typename DataType>
PixelFrameBuffer<DataType>::PixelFrameBuffer(int batch,
                                             int timeSteps,
                                             int channels,
                                             int rows,
                                             int cols)
    : batch_(batch), timeSteps_(timeSteps), channels_(channels),
      rows_(rows), cols_(cols),
      curr_(batch, timeSteps, channels, rows, cols),
      prev_(batch, timeSteps, channels, rows, cols) {
    curr_.setZero();
    prev_.setZero();
}

// ########## STUB: PixelFrameBuffer::updateRender ##########
// PURPOSE: ingest a new 5D tensor into the current buffer.
// EXPECTED BEHAVIOR: lock and update internal state.
// ###########################################################################
template<typename DataType>
void PixelFrameBuffer<DataType>::updateRender(const Eigen::Tensor<DataType,5>& data) {
    curr_ = data;
}

// ########## STUB: PixelFrameBuffer::getDiffAndSwap ##########
// PURPOSE: compute diff between current and previous buffer.
// EXPECTED BEHAVIOR: return sparse events and swap buffers.
// ###########################################################################
template<typename DataType>
std::vector<std::tuple<int,int,int,int,int,DataType>>
PixelFrameBuffer<DataType>::getDiffAndSwap(DataType threshold) {
    (void)threshold;
    prev_ = curr_;
    return {};
}

template class PixelFrameBuffer<float>;

} // namespace asciioscilliscope
