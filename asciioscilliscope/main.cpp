#include "include/asciioscilliscope/Renderer.h"
#include <iostream>

int main() {
    asciioscilliscope::Renderer r(64, 64);
    std::vector<float> img(64*64*3, 0.0f);
    r.exciteFromImage(img);
    r.start();
    r.stop();
    std::cout << "Stub renderer executed" << std::endl;
    return 0;
}
