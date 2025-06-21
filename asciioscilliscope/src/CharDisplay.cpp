#include "../include/asciioscilliscope/CharDisplay.h"
#include <algorithm>

namespace asciioscilliscope {

CharDisplay::CharDisplay(int rows, int cols, bool fullPrint)
    : rows_(rows), cols_(cols), fullPrintMode_(fullPrint),
      buffers_{Eigen::Tensor<char,2>(rows, cols), Eigen::Tensor<char,2>(rows, cols)} {}

// ########## STUB: stageSlice ##########
// PURPOSE: enqueue next character slice for display
void CharDisplay::stageSlice(const Eigen::Tensor<char,2>& slice, double timestamp) {
    (void)timestamp;
    std::lock_guard<std::mutex> lock(mutex_);
    sliceQueue_.emplace_back(timestamp, slice);
}

// ########## STUB: getNextDiff ##########
// PURPOSE: compute diffs between queued slice and active buffer
std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>> CharDisplay::getNextDiff() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>> result;
    if (sliceQueue_.empty()) return result;
    Eigen::Tensor<char,2> slice = sliceQueue_.front().second;
    sliceQueue_.pop_front();
    buffers_[activeBuffer_] = slice;
    activeBuffer_ ^= 1;
    return result;
}

const Eigen::Tensor<char,2>& CharDisplay::getFullBuffer() const {
    return buffers_[activeBuffer_];
}

} // namespace asciioscilliscope
