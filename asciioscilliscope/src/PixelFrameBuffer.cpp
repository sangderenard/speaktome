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
    // ########## STUB: PixelFrameBuffer Constructor ##########
    // PURPOSE: allocate and zero-initialize buffers.
    // EXPECTED BEHAVIOR: prepare double-buffered tensors for diffs.
    // TODO: handle memory initialization and threading primitives.
    // ########################################################
    curr_.setZero();
    prev_.setZero();
}

template<typename DataType>
void PixelFrameBuffer<DataType>::updateRender(const Eigen::Tensor<DataType,5>& data) {
    // ########## STUB: updateRender ##########
    // PURPOSE: ingest next time-slice tensor.
    // TODO: enforce size checks and thread safety.
    // ########################################
    curr_ = data;
}

template<typename DataType>
std::vector<std::tuple<int,int,int,int,int,DataType>>
PixelFrameBuffer<DataType>::getDiffAndSwap(DataType threshold) {
    // ########## STUB: getDiffAndSwap ##########
    // PURPOSE: compute diff between curr_ and prev_.
    // EXPECTED BEHAVIOR: return sparse events above threshold.
    // TODO: implement diff computation using Eigen operations.
    // ########################################
    (void)threshold;
    std::vector<std::tuple<int,int,int,int,int,DataType>> diff;
    prev_ = curr_;
    return diff;
}

// explicit instantiation
template class PixelFrameBuffer<float>;

} // namespace asciioscilliscope
