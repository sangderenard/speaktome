#include "../include/asciioscilliscope/Renderer.h"
#include <thread>
#include <iostream>

namespace asciioscilliscope {

Renderer::Renderer(int imgWidth, int imgHeight)
    : imgW_(imgWidth), imgH_(imgHeight), phosphorW_(imgWidth/4), phosphorH_(imgHeight/4),
      charW_(imgWidth/16), charH_(imgHeight/16),
      pfb_(1,1,1,1,1), classifier_(), display_(charH_, charW_), running_(false) {}

void Renderer::exciteFromImage(const std::vector<float>& img) {
    (void)img; // ########## STUB ##########
}

void Renderer::start() {
    running_.store(true);
    // ########## STUB: main loop ##########
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    running_.store(false);
}

void Renderer::stop() {
    running_.store(false);
}

void Renderer::processDiffs(float threshold) {
    (void)threshold; // STUB
}

void Renderer::flushDisplay() {
    // STUB
}

} // namespace asciioscilliscope
