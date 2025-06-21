#include "../include/asciioscilliscope/CharDisplay.h"

namespace asciioscilliscope {

CharDisplay::CharDisplay(int rows, int cols, bool fullPrintMode)
    : rows_(rows), cols_(cols), fullPrintMode_(fullPrintMode),
      sliceQueue_(), activeBuffer_(0) {
    buffers_[0] = Eigen::Tensor<char,2>(rows, cols);
    buffers_[1] = Eigen::Tensor<char,2>(rows, cols);
    buffers_[0].setZero();
    buffers_[1].setZero();
}

// ########## STUB: CharDisplay::stageSlice ##########
// PURPOSE: enqueue a character slice for future display.
// EXPECTED BEHAVIOR: thread-safe insertion into the slice queue.
// ###########################################################################
void CharDisplay::stageSlice(const Eigen::Tensor<char,2>& slice, double timestamp) {
    sliceQueue_.emplace_back(timestamp, slice);
}

// ########## STUB: CharDisplay::getNextDiff ##########
// PURPOSE: compute diffs between queued slice and current buffer.
// EXPECTED BEHAVIOR: produce per-cell updates; this stub only swaps buffers.
// ###########################################################################
std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>>
CharDisplay::getNextDiff() {
    if (sliceQueue_.empty()) return {};
    buffers_[activeBuffer_ ^ 1] = sliceQueue_.front().second;
    sliceQueue_.pop_front();
    activeBuffer_ ^= 1;
    return {};
}

const Eigen::Tensor<char,2>& CharDisplay::getFullBuffer() const {
    return buffers_[activeBuffer_];
}

} // namespace asciioscilliscope
