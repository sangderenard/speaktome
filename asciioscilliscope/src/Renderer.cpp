#include "../include/asciioscilliscope/Renderer.h"
#include <iostream>

namespace asciioscilliscope {

Renderer::Renderer(int imgWidth, int imgHeight)
    : imgW_(imgWidth), imgH_(imgHeight),
      phosphorW_(imgWidth/4), phosphorH_(imgHeight/4),
      charW_(imgWidth/16), charH_(imgHeight/16),
      pfb_(1,1,1,1,1),
      classifier_(),
      display_(charH_, charW_),
      running_(false) {}

// ########## STUB: Renderer::exciteFromImage ##########
// PURPOSE: downsample an image into the phosphor grid.
// CURRENTLY: no-op placeholder.
void Renderer::exciteFromImage(const std::vector<float>& img) {
    (void)img;
}

// ########## STUB: Renderer::start ##########
// PURPOSE: run rendering loop until stop() is called.
void Renderer::start() {
    running_.store(true);
    std::cout << "Renderer start stub" << std::endl;
}

void Renderer::stop() {
    running_.store(false);
}

void Renderer::processDiffs(float threshold) {
    (void)threshold;
}

void Renderer::flushDisplay() {}

} // namespace asciioscilliscope
