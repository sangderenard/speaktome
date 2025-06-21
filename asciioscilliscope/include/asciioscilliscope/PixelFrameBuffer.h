#pragma once
#include <vector>
#include <cstdint>
#include <tuple>

namespace asciioscilliscope {

// Manages double buffering for raw pixel data and computes diffs.
class PixelFrameBuffer {
public:
    PixelFrameBuffer(int rows, int cols);
    // Update the current frame with new raw RGB data
    void updateRender(const std::vector<uint8_t>& data);
    // Compute the diff between previous and current buffer, then swap
    std::vector<std::tuple<int,int,uint8_t,uint8_t,uint8_t>> getDiffAndSwap();

private:
    int rows_, cols_, size_;
    std::vector<uint8_t> curr_, prev_;
    // TODO: add synchronization primitives if needed
};

} // namespace asciioscilliscope
