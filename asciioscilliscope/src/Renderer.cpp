#include "../include/asciioscilliscope/Renderer.h"
#include <iostream>

namespace asciioscilliscope {

Renderer::Renderer(int imgWidth, int imgHeight)
    : imgW_(imgWidth), imgH_(imgHeight),
      phosphorW_(imgWidth/4), phosphorH_(imgHeight/4),
      charW_(imgWidth/16), charH_(imgHeight/16),
      pfb_(1,1,3,charH_,charW_),
      display_(charH_, charW_),
      running_(false) {
    // ########## STUB: Renderer Constructor ##########
    // PURPOSE: initialize internal buffers and state.
    // TODO: connect to signal sources and allocate resources.
    // ###############################################
}

void Renderer::exciteFromImage(const std::vector<float>& img) {
    // ########## STUB: exciteFromImage ##########
    // PURPOSE: downsample image into phosphor grid and store in PixelFrameBuffer.
    // TODO: implement image processing and buffer update.
    // ###########################################
    (void)img;
}

void Renderer::start() {
    // ########## STUB: start ##########
    // PURPOSE: run render loop until stop() called.
    // For now simply process one empty frame.
    // ##################################
    running_.store(true);
    processDiffs(0.f);
    flushDisplay();
}

void Renderer::stop() {
    running_.store(false);
}

void Renderer::processDiffs(float threshold) {
    // ########## STUB: processDiffs ##########
    // PURPOSE: convert PixelFrameBuffer diffs into CharDisplay updates.
    // TODO: implement diff handling and classification.
    // #########################################
    (void)threshold;
}

void Renderer::flushDisplay() {
    // ########## STUB: flushDisplay ##########
    // PURPOSE: output current display state to terminal or buffer.
    // TODO: send ANSI sequences or integrate with GUI.
    // ########################################
    auto& buf = display_.getFullBuffer();
    (void)buf;
    std::cout << "Renderer flushDisplay stub" << std::endl;
}

} // namespace asciioscilliscope
