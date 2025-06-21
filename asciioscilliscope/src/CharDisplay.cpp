#include "../include/asciioscilliscope/CharDisplay.h"
#include <algorithm>

namespace asciioscilliscope {

// ########## STUB: CharDisplay Implementation ##########
// PURPOSE: Simplified placeholder adhering to the header contract.
// EXPECTED BEHAVIOR: queue slices, compute diffs against double buffers
// and allow retrieval of either diffs or full buffers.
// TODO:
//   - implement efficient Eigen based diffing
//   - support timed playback of queued slices
// ######################################################################

CharDisplay::CharDisplay(int rows, int cols, bool fullPrint)
    : rows_(rows), cols_(cols), fullPrintMode_(fullPrint),
      sliceQueue_(), buffers_{Eigen::Tensor<char,2>(rows,cols),
                              Eigen::Tensor<char,2>(rows,cols)} {
    buffers_[0].setZero();
    buffers_[1].setZero();
}

void CharDisplay::stageSlice(const Eigen::Tensor<char,2>& slice, double timestamp) {
    std::lock_guard<std::mutex> lk(mutex_);
    sliceQueue_.emplace_back(timestamp, slice);
}

std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>> CharDisplay::getNextDiff() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (sliceQueue_.empty()) return {};
    auto pair = sliceQueue_.front();
    sliceQueue_.pop_front();
    const Eigen::Tensor<char,2>& next = pair.second;
    Eigen::Tensor<char,2>& curr = buffers_[activeBuffer_];

    std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>> diffs;
    for (int r=0; r<rows_; ++r) {
        for (int c=0; c<cols_; ++c) {
            char n = next(r,c);
            char cur = curr(r,c);
            if (fullPrintMode_ || n != cur) {
                diffs.emplace_back(r,c,n,0,0,0); // color values unused for now
                curr(r,c) = n;
            }
        }
    }
    activeBuffer_ = 1 - activeBuffer_;
    buffers_[activeBuffer_] = next;
    return diffs;
}

const Eigen::Tensor<char,2>& CharDisplay::getFullBuffer() const {
    return buffers_[activeBuffer_];
}

} // namespace asciioscilliscope
