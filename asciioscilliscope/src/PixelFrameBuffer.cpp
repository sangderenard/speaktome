#include "../include/asciioscilliscope/PixelFrameBuffer.h"
#include <mutex>

namespace asciioscilliscope {

PixelFrameBuffer::PixelFrameBuffer(int rows, int cols)
    : rows_(rows), cols_(cols), size_(rows*cols*3), curr_(size_), prev_(size_) {}

void PixelFrameBuffer::updateRender(const std::vector<uint8_t>& data) {
    if (data.size() != size_) return;
    // TODO: lock if multithreaded
    curr_ = data;
}

std::vector<std::tuple<int,int,uint8_t,uint8_t,uint8_t>> PixelFrameBuffer::getDiffAndSwap() {
    std::vector<std::tuple<int,int,uint8_t,uint8_t,uint8_t>> diff;
    // TODO: lock if multithreaded
    for (int i=0, idx=0; i<size_; i+=3, ++idx) {
        uint8_t r = curr_[i], g = curr_[i+1], b = curr_[i+2];
        uint8_t pr = prev_[i], pg = prev_[i+1], pb = prev_[i+2];
        if (r!=pr || g!=pg || b!=pb) {
            int y = idx / cols_, x = idx % cols_;
            diff.emplace_back(y, x, r, g, b);
        }
    }
    prev_.swap(curr_);
    return diff;
}

} // namespace asciioscilliscope
