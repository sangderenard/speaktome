#include "../include/asciioscilliscope/CharClassifier.h"

namespace asciioscilliscope {

char CharClassifier::classify(uint8_t r, uint8_t g, uint8_t b) const {
    static const std::string ramp = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
    float brightness = 0.2126f*r + 0.7152f*g + 0.0722f*b;
    size_t idx = static_cast<size_t>((brightness/255.0f)*(ramp.size()-1));
    return ramp[idx];
}

} // namespace asciioscilliscope
