#include "../include/asciioscilliscope/CharDisplay.h"
#include <algorithm>

namespace asciioscilliscope {

CharDisplay::CharDisplay(int rows, int cols)
    : rows_(rows), cols_(cols), curr_(rows*cols), next_(rows*cols) {}

void CharDisplay::set(int y, int x, char ch, uint8_t r, uint8_t g, uint8_t b) {
    if (x<0||x>=cols_||y<0||y>=rows_) return;
    next_[y*cols_ + x] = CharCell{ch, r, g, b};
}

std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>> CharDisplay::diffAndSwap() {
    std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>> diff;
    for (int i=0; i<rows_*cols_; ++i) {
        const CharCell &n = next_[i];
        CharCell &c = curr_[i];
        if (n.ch!=c.ch || n.r!=c.r || n.g!=c.g || n.b!=c.b) {
            diff.emplace_back(i/cols_, i%cols_, n.ch, n.r, n.g, n.b);
            c = n;
        }
    }
    std::fill(next_.begin(), next_.end(), CharCell{' ',0,0,0});
    return diff;
}

} // namespace asciioscilliscope
