#pragma once
#include <vector>
#include <tuple>
#include <cstdint>

namespace asciioscilliscope {

struct CharCell {
    char ch;
    uint8_t r, g, b;
};

// Manages double buffering for character grid and computes diffs
class CharDisplay {
public:
    CharDisplay(int rows, int cols);
    void set(int y, int x, char ch, uint8_t r, uint8_t g, uint8_t b);
    std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>> diffAndSwap();

private:
    int rows_, cols_;
    std::vector<CharCell> curr_, next_;
};

} // namespace asciioscilliscope
