#include "../include/asciioscilliscope/CharDisplay.h"

namespace asciioscilliscope {

CharDisplay::CharDisplay(int rows, int cols, bool fullPrintMode)
  : rows_(rows),
    cols_(cols),
    fullPrintMode_(fullPrintMode),
    sliceQueue_(),
    activeBuffer_(0),
    buffers_{ Eigen::Tensor<char,2>(rows, cols),
              Eigen::Tensor<char,2>(rows, cols) }
{
  buffers_[0].setZero();
  buffers_[1].setZero();
}

void CharDisplay::stageSlice(const Eigen::Tensor<char,2>& slice, double timestamp) {
    std::lock_guard lock(mutex_);
    sliceQueue_.emplace_back(timestamp, slice);
}

std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>>
CharDisplay::getNextDiff() {
    std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>> diff;
    std::lock_guard lock(mutex_);
    if (sliceQueue_.empty()) return diff;

    // Compute diffs between current activeBuffer_ and next slice
    const auto& oldBuf = buffers_[activeBuffer_];
    const auto& newBuf = sliceQueue_.front().second;
    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < cols_; ++c) {
        char newCh = newBuf(r, c);
        if (oldBuf(r, c) != newCh) {
          // TODO: compute actual colors if needed
          diff.emplace_back(r, c, newCh, 0, 0, 0);
        }
      }
    }

    // Swap buffers
    buffers_[activeBuffer_ ^ 1] = newBuf;
    sliceQueue_.pop_front();
    activeBuffer_ ^= 1;
    return diff;
}

const Eigen::Tensor<char,2>& CharDisplay::getFullBuffer() const {
    return buffers_[activeBuffer_];
}

const Eigen::Tensor<char,2>& CharDisplay::getFullBuffer() const {
    return buffers_[activeBuffer_];
}

} // namespace asciioscilliscope
