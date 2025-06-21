#include "include/asciioscilliscope/Renderer.h"
#include <vector>

int main() {
    asciioscilliscope::Renderer renderer(16, 16);
    std::vector<float> img(16 * 16 * 3, 0.0f);
    renderer.exciteFromImage(img);
    renderer.start();
    return 0;
}
