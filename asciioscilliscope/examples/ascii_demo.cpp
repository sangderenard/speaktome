#include "asciioscilliscope/CharClassifier.h"
#include "asciioscilliscope/CharDisplay.h"
#include <thread>

int main() {
    // Simple example: display a hello world message in ASCII at center
    const int rows = 10, cols = 40;
    asciioscilliscope::CharDisplay display(rows, cols, true);
    // Prepare a buffer of spaces
    Eigen::Tensor<char,2> buffer(rows, cols);
    buffer.setConstant(' ');
    std::string msg = "Hello, AsciiOscilloscope!";
    int startCol = (cols - msg.size()) / 2;
    for (int i = 0; i < msg.size(); ++i) {
        buffer(5, startCol + i) = msg[i];
    }
    display.stageSlice(buffer, 0.0);
    auto full = display.getFullBuffer();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << full(r, c);
        }
        std::cout << "\n";
    }
    return 0;
}
