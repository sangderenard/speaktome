#pragma once
#include <cstdint>
#include <string>

namespace asciioscilliscope {

// Classifies RGB values to ASCII characters based on brightness ramp
class CharClassifier {
public:
    // Map a color triplet to an ASCII character
    char classify(uint8_t r, uint8_t g, uint8_t b) const;
};

} // namespace asciioscilliscope
