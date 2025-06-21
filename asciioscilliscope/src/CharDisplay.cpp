#include "../include/asciioscilliscope/CharDisplay.h"
#include <algorithm>

namespace asciioscilliscope {

CharDisplay::CharDisplay(int rows, int cols, bool fullPrintMode)
    : rows_(rows), cols_(cols), fullPrintMode_(fullPrintMode),
      buffers_{Eigen::Tensor<char,2>(rows, cols), Eigen::Tensor<char,2>(rows, cols)} {
    // ########## STUB: CharDisplay Constructor ##########
    // PURPOSE: initialize double buffers and mode flag.
    // TODO: allocate buffers with zeros and prepare queue.
    // #########################################
    buffers_[0].setZero();
    buffers_[1].setZero();
}

void CharDisplay::stageSlice(const Eigen::Tensor<char,2>& slice, double timestamp) {
    // ########## STUB: stageSlice ##########
    // PURPOSE: enqueue a slice for later display.
    // TODO: manage queue ordering and capacity.
    // ########################################
    std::lock_guard<std::mutex> lock(mutex_);
    sliceQueue_.emplace_back(timestamp, slice);
}

std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>> CharDisplay::getNextDiff() {
    // ########## STUB: getNextDiff ##########
    // PURPOSE: diff next queued slice against active buffer.
    // EXPECTED BEHAVIOR: return list of changed cells.
    // TODO: implement real diff computation.
    // ######################################
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>> diff;
    if (sliceQueue_.empty()) return diff;
    auto next = sliceQueue_.front().second;
    sliceQueue_.pop_front();
    buffers_[activeBuffer_] = next;
    activeBuffer_ = 1 - activeBuffer_;
    return diff;
}

const Eigen::Tensor<char,2>& CharDisplay::getFullBuffer() const {
    return buffers_[activeBuffer_];
}

} // namespace asciioscilliscope
