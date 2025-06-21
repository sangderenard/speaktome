#include "../include/asciioscilliscope/CharClassifier.h"

namespace asciioscilliscope {

// ########## STUB: CharClassifier::classify ##########
// PURPOSE: Map RGB values to an ASCII symbol using a brightness ramp.
// EXPECTED BEHAVIOR: Select a character corresponding to the computed
// brightness. This stub uses a fixed ramp and ignores advanced options.
// ###########################################################################
char CharClassifier::classify(uint8_t r, uint8_t g, uint8_t b) const {
    static const std::string ramp = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
    float brightness = 0.2126f*r + 0.7152f*g + 0.0722f*b;
    size_t idx = static_cast<size_t>((brightness/255.0f)*(ramp.size()-1));
    return ramp[idx];
}

} // namespace asciioscilliscope
