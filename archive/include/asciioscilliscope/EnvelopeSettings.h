#pragma once
#include <vector>

namespace asciioscilliscope {

/**
 * EnvelopeSettings
 * ----------------
 * Simple struct describing an excitation decay envelope for a pixel site.
 * Represents amplitude coefficients over time that will be sampled during
 * phosphor decay. This header is referenced by PhosphorScreen but had no
 * implementation.
 */
struct EnvelopeSettings {
    std::vector<float> coefficients;  ///< envelope amplitude samples
    float sampleRate = 1.0f;          ///< samples per second
};

} // namespace asciioscilliscope
