#include "../include/asciioscilliscope/PixelFrameBuffer.h"
#include <cmath>

namespace asciioscilliscope {

// ########## STUB: PixelFrameBuffer Implementation ##########
// PURPOSE: placeholder implementation matching the header expectations.
// EXPECTED BEHAVIOR: manage double-buffered Eigen tensors and produce
// sparse diffs for downstream consumers.
// INPUTS: constructor parameters define tensor dimensions. updateRender()
// receives a 5D tensor. getDiffAndSwap() returns diff events above a
// threshold.
// TODO:
//   - integrate mutex for thread safety
//   - optimize diff computation using Eigen broadcasting
// NOTES: current implementation performs naive element-wise operations
// for compilation only.
// ######################################################################

template<typename DataType>
PixelFrameBuffer<DataType>::PixelFrameBuffer(int batch, int timeSteps,
                                             int channels, int rows, int cols)
    : batch_(batch), timeSteps_(timeSteps), channels_(channels),
      rows_(rows), cols_(cols),
      curr_(batch, timeSteps, channels, rows, cols),
      prev_(batch, timeSteps, channels, rows, cols) {
    curr_.setZero();
    prev_.setZero();
}

template<typename DataType>
void PixelFrameBuffer<DataType>::updateRender(
    const Eigen::Tensor<DataType,5>& data) {
    // Direct assignment for now; no locking
    curr_ = data;
}

template<typename DataType>
std::vector<std::tuple<int,int,int,int,int,DataType>>
PixelFrameBuffer<DataType>::getDiffAndSwap(DataType threshold) {
    std::vector<std::tuple<int,int,int,int,int,DataType>> events;
    for (int b=0; b<batch_; ++b) {
        for (int t=0; t<timeSteps_; ++t) {
            for (int c=0; c<channels_; ++c) {
                for (int r=0; r<rows_; ++r) {
                    for (int col=0; col<cols_; ++col) {
                        DataType currVal = curr_(b,t,c,r,col);
                        DataType prevVal = prev_(b,t,c,r,col);
                        DataType delta = currVal - prevVal;
                        if (std::abs(delta) > threshold) {
                            events.emplace_back(b,t,c,r,col,delta);
                        }
                        prev_(b,t,c,r,col) = currVal;
                    }
                }
            }
        }
    }
    return events;
}

// Explicit template instantiation for float
template class PixelFrameBuffer<float>;

} // namespace asciioscilliscope
