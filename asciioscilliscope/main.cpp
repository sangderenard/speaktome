#include "include/asciioscilliscope/Renderer.h"

int main() {
    asciioscilliscope::Renderer renderer(64, 32);
    renderer.start();
    renderer.stop();
    return 0;
}
