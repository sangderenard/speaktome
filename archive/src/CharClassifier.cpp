#include "../include/asciioscilliscope/CharClassifier.h"
#include <algorithm>
#include <vector>

namespace asciioscilliscope {

CharClassifier::CharClassifier(const std::string& ramp)
{
    if (ramp.empty()) {
        ramp_ = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
    } else {
        ramp_ = ramp;
    }
}

// ########## STUB: CharClassifier::classify ##########
// PURPOSE: Map RGB values to an ASCII symbol using a brightness ramp.
// EXPECTED BEHAVIOR: Select a character corresponding to the computed
// brightness. This stub uses a fixed ramp and ignores advanced options.
// ###########################################################################
char CharClassifier::classify(uint8_t r, uint8_t g, uint8_t b) const {
    // Delegate to the vector-based variant so the logic stays in one place.
    std::vector<uint8_t> tmp = {r, g, b};
    return classify(tmp);
}

char CharClassifier::classify(const std::vector<uint8_t>& channels) const {
    // ########## STUB: classify(const vector<uint8_t>&) ##########
    // PURPOSE: map an arbitrary set of channel intensities to an ASCII char.
    // EXPECTED BEHAVIOR: average all channels to obtain a brightness value,
    // then map that brightness via the configured ramp.
    if (channels.empty()) {
        return ' ';
    }
    float sum = 0.0f;
    for (uint8_t v : channels) {
        sum += static_cast<float>(v);
    }
    float brightness = sum / channels.size();
    size_t idx = static_cast<size_t>((brightness / 255.0f) * std::max<size_t>(1, ramp_.size() - 1));
    idx = std::min(idx, ramp_.size() - 1);
    return ramp_[idx];
}

void CharClassifier::setRamp(const std::string& ramp) {
    if (!ramp.empty()) {
        ramp_ = ramp;
    }
}

} // namespace asciioscilliscope
