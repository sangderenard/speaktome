#include "../include/asciioscilliscope/PixelFrameBuffer.h"
#include <algorithm>

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

// ########## STUB: PixelFrameBuffer::updateRender ##########
// PURPOSE: ingest a new 5D tensor into the current buffer.
// EXPECTED BEHAVIOR: lock and update internal state.
// ###########################################################################
template<typename DataType>
void PixelFrameBuffer<DataType>::updateRender(const Eigen::Tensor<DataType,5>& data) {
    // Basic shape verification. Eigen::Tensor::dimensions() returns an array
    // of fixed size. Only perform checks in debug builds to avoid runtime
    // overhead.
#ifndef NDEBUG
    assert(data.dimension(0) == batch_);
    assert(data.dimension(1) == timeSteps_);
    assert(data.dimension(2) == channels_);
    assert(data.dimension(3) == rows_);
    assert(data.dimension(4) == cols_);
#endif
    curr_ = data;
}

// ########## STUB: PixelFrameBuffer::getDiffAndSwap ##########
// PURPOSE: compute diff between current and previous buffer.
// EXPECTED BEHAVIOR: return sparse events and swap buffers.
// ###########################################################################
template<typename DataType>
std::vector<std::tuple<int,int,int,int,int,DataType>>
PixelFrameBuffer<DataType>::getDiffAndSwap(DataType threshold) {
    Eigen::Tensor<DataType,5> diff = curr_ - prev_;
    Eigen::Tensor<DataType,5> absDiff = diff.abs();
    std::vector<std::tuple<int,int,int,int,int,DataType>> events;
    for (int b = 0; b < batch_; ++b) {
        for (int t = 0; t < timeSteps_; ++t) {
            for (int c = 0; c < channels_; ++c) {
                for (int r = 0; r < rows_; ++r) {
                    for (int col = 0; col < cols_; ++col) {
                        DataType delta = absDiff(b,t,c,r,col);
                        if (delta > threshold) {
                            events.emplace_back(
                                b, t, c, r, col,
                                curr_(b,t,c,r,col) - prev_(b,t,c,r,col));
                        }
                    }
                }
            }
        }
    }
    std::swap(prev_, curr_);
    return events;
}

template class PixelFrameBuffer<float>;

} // namespace asciioscilliscope
