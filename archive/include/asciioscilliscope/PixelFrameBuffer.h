#pragma once
#include <vector>
#include <cstdint>
#include <tuple>
#include "eigen/Eigen/Core"
#include "eigen/unsupported/Eigen/CXX11/Tensor"

namespace asciioscilliscope {

/**
 * PixelFrameBuffer<T>
 * -------------------
 * Generic 5D spatiotemporal buffer manager using Eigen::Tensor<DataType,5>.
 * Manages double-buffered curr_/prev_ states for batched streams of time slices.
 * Each stage can handle float-normalized frames or integer-encoded control data.
 *
 * Dimensions:
 *   - batch       : parallel streams
 *   - timeSteps   : causal history window length
 *   - channels    : data channels per element (e.g. color planes)
 *   - rows, cols  : spatial dimensions
 *
 * Responsibilities:
 *   - Allocate and initialize curr_ and prev_ Eigen tensors
 *   - Provide updateRender(): ingest next time-slice package
 *   - Provide getDiffAndSwap(): compute sparse diffs above threshold
 *   - Maintain strict causal ordering via buffer swapping
 *
 * Thread Safety:
 *   - Protect updateRender() and getDiffAndSwap() with a mutex (TODO)
 *   - Designed for multi-threaded producer/consumer pipelines
 */
template<typename DataType = float>
class PixelFrameBuffer {
public:
    /**
     * Constructor
     * @param batch     Number of streams
     * @param timeSteps Temporal window length
     * @param channels  Number of channels per element
     * @param rows      Height dimension
     * @param cols      Width dimension
     */
    PixelFrameBuffer(int batch, int timeSteps, int channels, int rows, int cols);

    /**
     * updateRender
     * @param data 5D Eigen::Tensor<DataType,5> with [batch, timeSteps, channels, rows, cols]
     */
    void updateRender(const Eigen::Tensor<DataType,5>& data);

    /**
     * getDiffAndSwap
     * @param threshold Minimum delta of DataType to report
     * @return Vector of (batchIdx, timeIdx, channelIdx, rowIdx, colIdx, delta)
     */
    std::vector<std::tuple<int,int,int,int,int,DataType>>
    getDiffAndSwap(DataType threshold = static_cast<DataType>(1e-5f));

private:
    int batch_, timeSteps_, channels_, rows_, cols_;
    Eigen::Tensor<DataType,5> curr_, prev_;
    // std::mutex mutex_; // TODO: enable when multithreading
};

/**
 * Algorithm Outline (Broadcasted Diff Pipeline):
 *   1. Producer threads enqueue incoming 5D tensor chunks (time-slice packages) into a lock-protected queue.
 *   2. Consumer thread dequeues the next tensor and loads it into curr_ buffer.
 *   3. Compute broadcasted diff: diffTensor = (curr_ - prev_).abs() via Eigen.
 *   4. Threshold diffTensor to create a mask of changed elements.
 *   5. Collect sparse events (batch, time, channel, row, col, delta).
 *   6. Swap buffers: prev_.swap(curr_) under mutex.
 *   7. Emit diff events downstream and repeat for subsequent time-slices.
 *
 * Performance Notes:
 *   - Eigen Tensor broadcasts leverage SIMD acceleration
 *   - Minimal locking for high-throughput pipelines
 *   - Optional sparse output formats (CSR, COO) may be added
 */

} // namespace asciioscilliscope
