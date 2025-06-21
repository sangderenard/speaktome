#include "../include/asciioscilliscope/Renderer.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace std::chrono;

namespace asciioscilliscope {

Renderer::Renderer(int imgWidth, int imgHeight)
    : imgW_(imgWidth), imgH_(imgHeight),
      phosphorW_(imgWidth/4), phosphorH_(imgHeight/4),
      charW_(imgWidth/16), charH_(imgHeight/16),
      fb_(charH_, charW_), display_(charH_, charW_), running_(true) {}

void Renderer::exciteFromImage(const std::vector<uint8_t>& img) {
    // Downsample image into phosphor grid
    std::vector<uint8_t> phosphorBuf(phosphorW_*phosphorH_*3);
    for (int py=0; py<phosphorH_; ++py) {
        for (int px=0; px<phosphorW_; ++px) {
            int sumR=0,sumG=0,sumB=0;
            for (int by=0; by<4; ++by) for (int bx=0; bx<4; ++bx) {
                int ix = px*4 + bx;
                int iy = py*4 + by;
                int idx = (iy*imgW_ + ix)*3;
                sumR += img[idx]; sumG += img[idx+1]; sumB += img[idx+2];
            }
            phosphorBuf[(py*phosphorW_ + px)*3]   = sumR/16;
            phosphorBuf[(py*phosphorW_ + px)*3+1] = sumG/16;
            phosphorBuf[(py*phosphorW_ + px)*3+2] = sumB/16;
        }
    }
    fb_.updateRender(phosphorBuf);
}

void Renderer::start() {
    std::thread input([&]{ std::cin.get(); running_.store(false); });
    std::cout << "\x1B[?25l"; // hide cursor
    while (running_.load()) {
        auto diffs = fb_.getDiffAndSwap();
        for (auto &t : diffs) {
            int y,x; uint8_t r,g,b;
            std::tie(y,x,r,g,b) = t;
            char ch = classifier_.classify(r,g,b);
            display_.set(y,x,ch,r,g,b);
        }
        auto charDiffs = display_.diffAndSwap();
        draw(charDiffs);
        std::this_thread::sleep_for(milliseconds(33));
    }
    std::cout << "\x1B[?25h"; // show cursor
    input.join();
}

void Renderer::draw(const std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>>& diffs) {
    for (auto &t : diffs) {
        int y,x; char ch; uint8_t r,g,b;
        std::tie(y,x,ch,r,g,b) = t;
        // Move cursor and set color then print char
        std::printf("\x1B[%d;%dH", y+1, x+1);
        std::printf("\x1B[38;2;%d;%d;%dm%c", r,g,b,ch);
    }
    std::fflush(stdout);
}

} // namespace asciioscilliscope
