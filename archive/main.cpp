#include "include/asciioscilliscope/Renderer.h"

int main() {
    asciioscilliscope::Renderer renderer(80, 60);
    renderer.start();
    renderer.stop();
    return 0;
}
