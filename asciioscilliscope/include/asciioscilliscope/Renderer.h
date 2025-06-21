#pragma once
#include <vector>
#include <atomic>
#include <tuple>
#include "PixelFrameBuffer.h"
#include "CharDisplay.h"
#include "CharClassifier.h"

namespace asciioscilliscope {

// Renders pixel data to terminal by diffing buffers and mapping to ASCII
class Renderer {
public:
    Renderer(int imgWidth, int imgHeight);
    // Feed raw image buffer (RGB) at full resolution
    void exciteFromImage(const std::vector<uint8_t>& img);
    // Start rendering loop; terminates when running_ is false
    void start();

private:
    int imgW_, imgH_, phosphorW_, phosphorH_, charW_, charH_;
    PixelFrameBuffer fb_;
    CharDisplay display_;
    CharClassifier classifier_;
    std::vector<uint8_t> imgPrev_, phPrev_;
    std::atomic<bool> running_;

    void draw(const std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>>& diffs);
};

} // namespace asciioscilliscope
